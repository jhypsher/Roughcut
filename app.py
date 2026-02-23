"""
app.py — Flask UI for Video Highlight Extractor

Run with:
    python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import json
import queue
import threading
import time
import subprocess
import re
import csv
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, Response, jsonify, stream_with_context

app = Flask(__name__)

# Global job state (single-user local tool)
job = {
    "running": False,
    "queue": queue.Queue(),
    "thread": None,
}


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────
def run_pipeline(cfg: dict, q: queue.Queue):
    def emit(kind, **data):
        data["kind"] = kind
        data["ts"] = datetime.now().strftime("%H:%M:%S")
        q.put(data)

    def log(msg, level="info"):
        emit("log", msg=msg, level=level)

    def stage(s, pct):
        emit("stage", name=s, pct=pct)

    def metric(k, v):
        emit("metric", key=k, val=v)

    try:
        import cv2
        import numpy as np
        from dotenv import load_dotenv
        import google.generativeai as genai
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor, as_completed

        video_path  = cfg["video_path"]
        output_dir  = Path(cfg["output_dir"])
        dotenv_path = cfg["dotenv_path"]
        frame_step  = int(cfg["frame_step"])
        motion_thresh = float(cfg["motion_thresh"])
        top_k       = int(cfg["top_k"])
        batch_size  = int(cfg["batch_size"])
        num_workers = int(cfg["num_workers"])
        padding_sec = float(cfg["padding_seconds"])
        make_concat = cfg.get("make_concat", True)
        model_name  = cfg.get("gemini_model", "gemini-2.5-flash-lite")

        output_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Validate ───────────────────────────────────────
        stage("Validating", 2)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        size_mb = os.path.getsize(video_path) / (1024 ** 2)
        log(f"Video OK: {Path(video_path).name}  ({size_mb:.1f} MB)")

        # ── 2. Extract frames ─────────────────────────────────
        stage("Extracting Frames", 5)
        existing = [f for f in output_dir.iterdir()
                    if f.name.startswith("frame_") and f.suffix == ".jpg"]
        if existing:
            saved_count = len(existing)
            log(f"Frames already exist ({saved_count}) — skipping extraction.")
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            log(f"Opened video — {total_vid} total frames, saving every {frame_step}th.")
            frame_count = saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % frame_step == 0:
                    cv2.imwrite(str(output_dir / f"frame_{saved_count:05d}.jpg"), frame)
                    saved_count += 1
                    if saved_count % 100 == 0:
                        pct = min(5 + int((frame_count / total_vid) * 15), 20)
                        stage("Extracting Frames", pct)
                        log(f"  Saved {saved_count} frames...")
            cap.release()
            log(f"Extraction done — {saved_count} frames saved.", "ok")

        metric("frames_extracted", saved_count)
        stage("Motion Filter", 22)

        # ── 3. Motion prefilter ───────────────────────────────
        image_paths = sorted([
            p for p in output_dir.iterdir()
            if p.is_file() and p.name.startswith("frame_") and p.suffix.lower() == ".jpg"
        ])
        log(f"Motion-filtering {len(image_paths)} frames (threshold={motion_thresh})...")

        scores = []
        prev = None
        for p in image_paths:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                scores.append((p, 0.0)); prev = None; continue
            scores.append((p, 0.0 if prev is None else float(cv2.absdiff(im, prev).mean())))
            prev = im

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        filtered = [p for p, _ in sorted_scores[:top_k]] if top_k else [p for p, s in scores if s >= motion_thresh]
        if not filtered:
            filtered = [p for p, _ in sorted_scores[:max(1, top_k or 5)]]

        log(f"Filter kept {len(filtered)} / {len(image_paths)} frames.", "ok")
        metric("frames_filtered", len(filtered))
        image_paths = filtered
        stage("Gemini Vision", 28)

        # ── 4. Gemini setup ───────────────────────────────────
        load_dotenv(dotenv_path=str(dotenv_path))
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(f"GOOGLE_API_KEY not found in {dotenv_path}")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name)
        log("Gemini API ready.", "ok")

        # ── 5. Batch processing ───────────────────────────────
        ts = time.strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"gemini_results_{ts}.jsonl"

        PROMPT = """You will be given multiple images. For each image return one JSON object in a JSON array.
Images in order:
{files_list}

Return ONLY a JSON array with one object per image:
[
  {{"file":"frame_00010.jpg","is_shooting":true,"confidence":0.87,"shooter":"player #30","shooter_team":"Warriors","notes":"Clear release"}},
  {{"file":"frame_00011.jpg","is_shooting":false,"confidence":0.15,"shooter":null,"shooter_team":null,"notes":"No motion"}}
]
Rules: is_shooting=true only if clearly releasing the ball; confidence 0-1; shooter/shooter_team null if unknown."""

        def gemini_call(parts, retries=4, backoff=2.0):
            for attempt in range(1, retries + 2):
                try:
                    return model.generate_content(parts)
                except Exception as e:
                    if attempt > retries: raise
                    wait = backoff ** attempt
                    log(f"  Retry {attempt}/{retries} ({e}) — waiting {wait:.1f}s", "warn")
                    time.sleep(wait)

        def parse_resp(resp, filenames):
            text = (getattr(resp, "text", None) or str(resp or "")).strip()
            parsed = None
            try: parsed = json.loads(text)
            except Exception:
                s = text.find("["); e = text.rfind("]")
                if s != -1 and e > s:
                    try: parsed = json.loads(text[s:e+1])
                    except: pass
                if parsed is None:
                    objs = []
                    for ln in text.splitlines():
                        try: objs.append(json.loads(ln.strip()))
                        except: pass
                    if objs: parsed = objs
            if parsed is None:
                return [{"file": f, "is_shooting": False, "confidence": None} for f in filenames]
            if isinstance(parsed, list):
                if len(parsed) == len(filenames): return parsed
                by_f = {o.get("file"): o for o in parsed if isinstance(o, dict) and o.get("file")}
                return [by_f.get(f, {"file": f, "is_shooting": False, "confidence": None}) for f in filenames]
            if isinstance(parsed, dict) and len(filenames) == 1: return [parsed]
            return [{"file": f, "is_shooting": False, "confidence": None} for f in filenames]

        all_batches = [(i, image_paths[i:i+batch_size]) for i in range(0, len(image_paths), batch_size)]
        total_b = len(all_batches)
        log(f"Sending {len(image_paths)} frames in {total_b} batches ({num_workers} workers)...")

        results_by_idx: dict = {}
        lock = threading.Lock()
        done_count = [0]

        def process_batch(bidx, batch_imgs):
            filenames = [p.name for p in batch_imgs]
            flist = "\n".join(f"{j+1}. {n}" for j, n in enumerate(filenames))
            prompt = PROMPT.format(files_list=flist)
            pil_imgs = []
            for p in batch_imgs:
                try: pil_imgs.append(Image.open(p).convert("RGB"))
                except Exception as e: log(f"  Cannot open {p.name}: {e}", "warn")
            try:
                resp = gemini_call([prompt] + pil_imgs)
                recs = parse_resp(resp, filenames)
            except Exception as e:
                log(f"  Batch {bidx} failed: {e}", "error")
                recs = [{"file": f, "is_shooting": False, "confidence": None} for f in filenames]
            with lock:
                results_by_idx[bidx] = recs
                done_count[0] += 1
                pct = 28 + int((done_count[0] / total_b) * 52)
                stage("Gemini Vision", pct)
                log(f"  Batch {done_count[0]}/{total_b} done ({len(recs)} frames)", "ok")
            return recs

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(process_batch, idx, batch): idx for idx, batch in all_batches}
            for f in as_completed(futures):
                try: f.result()
                except Exception as e: log(f"Batch future error: {e}", "error")

        all_results = []
        seen: set = set()
        with open(result_file, "w", encoding="utf-8") as out_f:
            for idx in sorted(results_by_idx):
                for rec in results_by_idx[idx]:
                    fn = rec.get("file", "")
                    if fn and fn not in seen:
                        seen.add(fn)
                        all_results.append(rec)
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        shooting = [r for r in all_results if r.get("is_shooting")]
        log(f"Gemini complete — {len(all_results)} frames, {len(shooting)} shooting detections.", "ok")
        metric("frames_processed", len(all_results))
        metric("shooting_detected", len(shooting))
        stage("Event Detection", 82)

        # ── 6. Event detection ────────────────────────────────
        fname_re = re.compile(r"frame_(\d+)\.jpg", re.IGNORECASE)

        cap2 = cv2.VideoCapture(video_path)
        fps = cap2.get(cv2.CAP_PROP_FPS)
        total_frames_v = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.release()
        video_dur = (total_frames_v - 1) / fps if total_frames_v > 0 else 0

        def fmt_ts(sec):
            h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
            return f"{h}:{m:02d}:{s:06.3f}"

        frames_data = []
        for rec in all_results:
            fname = rec.get("file", "")
            m = fname_re.search(fname)
            if not m: continue
            saved_idx = int(m.group(1))
            orig_frame = frame_step + saved_idx * frame_step
            ts_sec = (orig_frame - 1) / fps
            is_s = rec.get("is_shooting", False)
            if isinstance(is_s, str): is_s = is_s.lower() in ("true","1","yes")
            conf = rec.get("confidence")
            try: conf = max(0.0, min(1.0, float(conf))) if conf is not None else None
            except: conf = None
            frames_data.append({"fname": fname, "saved_idx": saved_idx,
                                 "ts_seconds": ts_sec, "is_shooting": bool(is_s), "confidence": conf})

        frames_data.sort(key=lambda x: x["saved_idx"])
        idx_sorted = [f["saved_idx"] for f in frames_data]
        idx_map = {f["saved_idx"]: f for f in frames_data}
        used: set = set()
        event_list = []

        for pos, idx in enumerate(idx_sorted):
            if idx in used: continue
            frame = idx_map[idx]
            if not frame["is_shooting"]: continue
            nxt = range(pos+1, min(pos+6, len(idx_sorted)))
            look = [idx_map[idx_sorted[p]] for p in nxt]
            if sum(1 for f in look if f["is_shooting"]) >= 2:
                ev_frames = [frame] + [f for f in look if f["is_shooting"]]
                for f in ev_frames: used.add(f["saved_idx"])
                ev_frames.sort(key=lambda x: x["saved_idx"])
                start = ev_frames[0]; end = ev_frames[-1]
                peak = max(ev_frames, key=lambda x: x["confidence"] if x["confidence"] is not None else -1)
                confs = [f["confidence"] for f in ev_frames if f["confidence"] is not None]
                mean_c = round(sum(confs)/len(confs), 3) if confs else None
                pad_s = max(0.0, start["ts_seconds"] - padding_sec)
                pad_e = min(video_dur, end["ts_seconds"] + padding_sec)
                event_list.append({
                    "start_time": fmt_ts(pad_s), "start_time_s": pad_s,
                    "end_time": fmt_ts(pad_e), "end_time_s": pad_e,
                    "peak_time": fmt_ts(peak["ts_seconds"]),
                    "num_frames": len(ev_frames),
                    "mean_confidence": mean_c,
                    "padded_duration_s": round(pad_e - pad_s, 3),
                    "detection_start": fmt_ts(start["ts_seconds"]),
                    "detection_end": fmt_ts(end["ts_seconds"]),
                    "raw_fnames": [f["fname"] for f in ev_frames],
                })

        events_json_path = output_dir / (result_file.stem + "_events.json")
        with open(events_json_path, "w", encoding="utf-8") as jf:
            json.dump({"video_path": str(video_path), "fps": fps, "events": event_list}, jf, indent=2)
        events_csv_path = output_dir / (result_file.stem + "_events.csv")
        with open(events_csv_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf)
            w.writerow(["start_time","start_time_s","end_time","end_time_s",
                        "peak_time","num_frames","mean_confidence","padded_duration_s"])
            for e in event_list:
                w.writerow([e["start_time"],e["start_time_s"],e["end_time"],e["end_time_s"],
                            e["peak_time"],e["num_frames"],e["mean_confidence"],e["padded_duration_s"]])

        log(f"Detected {len(event_list)} events. Wrote JSON + CSV.", "ok")
        metric("events_detected", len(event_list))
        emit("events", events=event_list)
        emit("paths", result_jsonl=str(result_file), events_json=str(events_json_path))
        stage("Trimming Clips", 88)

        # ── 7. Clip trimming ──────────────────────────────────
        import shutil
        clips_dir = output_dir / "Clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        concat_name = clips_dir / "roughcut.mp4"

        if not shutil.which("ffmpeg"):
            log("ffmpeg not found on PATH — skipping clip trimming.", "warn")
            log("Install ffmpeg: https://ffmpeg.org/download.html", "warn")
        else:
            clips_made = []
            for i, ev in enumerate(event_list, start=1):
                s = ev["start_time_s"]; e = ev["end_time_s"]; dur = e - s
                out_p = clips_dir / f"clip_{i:03d}_{int(s)}s_{int(dur)}s.mp4"
                cmd = ["ffmpeg","-y","-ss",f"{s:.3f}","-i",str(video_path),
                       "-t",f"{dur:.3f}","-c","copy",str(out_p)]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and out_p.exists() and out_p.stat().st_size > 1000:
                    clips_made.append(out_p)
                    log(f"  ✓ {out_p.name}", "ok")
                else:
                    cmd2 = ["ffmpeg","-y","-ss",f"{s:.3f}","-i",str(video_path),
                            "-t",f"{dur:.3f}","-c:v","libx264","-preset","fast",
                            "-crf","18","-c:a","aac",str(out_p)]
                    res2 = subprocess.run(cmd2, capture_output=True, text=True)
                    if res2.returncode == 0:
                        clips_made.append(out_p); log(f"  ✓ {out_p.name} (re-encoded)", "ok")
                    else:
                        log(f"  ✗ Failed: {out_p.name}", "error")
                pct = 88 + int((i / max(len(event_list),1)) * 10)
                stage("Trimming Clips", pct)

            metric("clips_saved", len(clips_made))

            if make_concat and clips_made:
                log("Concatenating clips into rough cut...")
                cl = clips_dir / "concat_list.txt"
                with open(cl,"w") as cf2:
                    for p in clips_made: cf2.write(f"file '{p.as_posix()}'\n")
                rc = subprocess.run(["ffmpeg","-y","-f","concat","-safe","0",
                                     "-i",str(cl),"-c","copy",str(concat_name)],
                                    capture_output=True, text=True)
                if rc.returncode == 0 and concat_name.exists():
                    log(f"Rough cut saved: {concat_name.name}", "ok")
                else:
                    log("Concat copy failed; trying re-encode...", "warn")
                    rc2 = subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(cl),
                                          "-c:v","libx264","-preset","fast","-crf","18",
                                          "-c:a","aac",str(concat_name)], capture_output=True, text=True)
                    if rc2.returncode == 0: log(f"Rough cut saved (re-encode).", "ok")
                    else: log("Concat failed.", "error")

        stage("Done", 100)
        log("Pipeline complete!", "ok")
        emit("done")

    except Exception as ex:
        import traceback
        q.put({"kind":"log","ts":datetime.now().strftime("%H:%M:%S"),
               "msg":f"FATAL: {ex}","level":"error"})
        q.put({"kind":"log","ts":datetime.now().strftime("%H:%M:%S"),
               "msg":traceback.format_exc(),"level":"error"})
        q.put({"kind":"stage","ts":datetime.now().strftime("%H:%M:%S"),
               "name":"Error","pct":0})
        q.put({"kind":"done","ts":datetime.now().strftime("%H:%M:%S")})
    finally:
        job["running"] = False


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    if job["running"]:
        return jsonify({"error": "A job is already running."}), 409

    cfg = request.get_json(force=True)
    required = ["video_path","output_dir","dotenv_path"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    job["running"] = True
    job["queue"] = queue.Queue()
    t = threading.Thread(target=run_pipeline, args=(cfg, job["queue"]), daemon=True)
    t.start()
    job["thread"] = t
    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    def generate():
        q = job["queue"]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("kind") == "done":
                    break
            except queue.Empty:
                yield "data: {\"kind\":\"ping\"}\n\n"
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.route("/status")
def status():
    return jsonify({"running": job["running"]})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
