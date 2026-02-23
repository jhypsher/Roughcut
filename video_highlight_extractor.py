"""
video_highlight_extractor.py

Extracts highlight clips from a basketball (or other sports) video by:
  1. Extracting frames from the video at a fixed interval
  2. Using Google Gemini Vision to classify each frame (is someone shooting?)
  3. Grouping shooting frames into events with timestamps
  4. Trimming the original video into individual clips + an optional rough-cut

Requirements: see requirements.txt
Usage: configure the CONFIG section below, then run each section in order.
"""

# ============================================================
# CONFIG — edit these paths before running
# ============================================================
import os

VIDEO_PATH      = r"path/to/your_video.mp4"   # path to your video file
OUTPUT_DIR      = r"path/to/output_dir"        # directory for frames and results
DOTENV_PATH     = r"path/to/api.env"           # file containing GOOGLE_API_KEY=...
GEMINI_MODEL    = "gemini-2.5-flash-lite"
FRAME_STEP      = 5          # save every Nth frame  (lower = more frames, slower)
PADDING_SECONDS = 1.0        # seconds of padding added before/after each detected event


# ============================================================
# SECTION 1 — Install / check packages
# ============================================================
# Run once in your environment:
#   pip install google-generativeai python-dotenv pillow opencv-python tqdm moviepy imageio-ffmpeg


# ============================================================
# SECTION 2 — Validate video file
# ============================================================
import subprocess

def validate_video(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    print("Video file found:", file_path)

    size_bytes = os.path.getsize(file_path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            print(f"Size: {size_bytes:.2f} {unit}")
            break
        size_bytes /= 1024.0

    try:
        subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", "-show_format", "-show_streams", file_path],
            check=True,
        )
    except FileNotFoundError:
        print("ffprobe not found — install ffmpeg if you want format info.")
    except subprocess.CalledProcessError as e:
        print("ffprobe returned a non-zero exit code:", e)


# ============================================================
# SECTION 3 — Extract frames
# ============================================================
import cv2

def extract_frames(video_path: str, output_dir: str, frame_step: int = 5) -> int:
    """Save every `frame_step`-th frame to output_dir as frame_NNNNN.jpg.
    Skips extraction if frames already exist."""
    os.makedirs(output_dir, exist_ok=True)

    existing = [f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")]
    if existing:
        print(f"Frames already exist in '{output_dir}'. Skipping frame extraction.")
        return len(existing)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    print("Video opened successfully. Extracting frames...")

    frame_count = saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_step == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")
    return saved_count


# ============================================================
# SECTION 4 — Load Gemini API key
# ============================================================
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

def configure_gemini(dotenv_path: str, model_name: str = GEMINI_MODEL):
    path = Path(dotenv_path)
    if not path.exists():
        raise FileNotFoundError(f"API env file not found: {path}")
    load_dotenv(dotenv_path=str(path))
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"GOOGLE_API_KEY not found in {path}. "
            "Ensure the file contains a line like:\nGOOGLE_API_KEY=YOUR_KEY_HERE"
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    print("Gemini API configured successfully.")
    return model


# ============================================================
# SECTION 5 — Gemini vision processing (batch + parallel)
# ============================================================
import io
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# --- Prompts ---
BATCH_PROMPT_TEMPLATE = """
You will be given multiple images. For each image, return a JSON object (one per image) in a JSON array.
Images in order:
{files_list}

Return ONLY a JSON array (no other text) with one object per image:
[
  {{"file": "frame_00010.jpg", "is_shooting": true, "confidence": 0.87, "shooter": "player #30", "shooter_team": "Warriors", "notes": "Clear release"}},
  {{"file": "frame_00011.jpg", "is_shooting": false, "confidence": 0.15, "shooter": null, "shooter_team": null, "notes": "No shooting motion"}}
]

Rules:
- "is_shooting": true only if clearly releasing/shooting the ball
- "confidence": 0.00-1.00
- "shooter": short description or null
- "shooter_team": team name/color or null
- "notes": 1-2 sentences of reasoning
"""


def call_gemini(model, prompt_parts, max_retries: int = 4, backoff_base: float = 2.0):
    """Call Gemini with exponential-backoff retry."""
    for attempt in range(1, max_retries + 2):
        try:
            return model.generate_content(prompt_parts)
        except Exception as e:
            if attempt > max_retries:
                raise
            wait = backoff_base ** attempt
            print(f"API call failed (attempt {attempt}/{max_retries}) — {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)


def motion_prefilter(
    image_paths: list,
    top_k: int = None,
    min_motion_thresh: float = 5.0,
) -> list:
    """Return frames with high motion (cheap grayscale diff heuristic).
    Thresholds: 5=loose, 10=medium, 15=strict, 20=very strict.
    """
    try:
        import numpy as np
    except ImportError:
        print("numpy not available; skipping motion prefilter.")
        return image_paths

    scores = []
    prev = None
    for p in image_paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            scores.append((p, 0.0))
            prev = None
            continue
        if prev is None:
            scores.append((p, 0.0))
        else:
            diff = cv2.absdiff(im, prev)
            scores.append((p, float(diff.mean())))
        prev = im

    if top_k:
        sorted_by_score = sorted(scores, key=lambda x: x[1], reverse=True)
        filtered = [p for p, s in sorted_by_score[:top_k]]
    else:
        filtered = [p for p, s in scores if s >= min_motion_thresh]

    if not filtered:
        sorted_all = sorted(scores, key=lambda x: x[1], reverse=True)
        filtered = [p for p, _ in sorted_all[:max(1, min(len(sorted_all), top_k or 5))]]

    return filtered


def _parse_gemini_response(resp, filenames: list) -> list:
    """Extract JSON records from a Gemini response, mapping back to filenames."""
    text = getattr(resp, "text", None)
    if text is None:
        try:
            d = resp.to_dict()
            text = d.get("candidates", [{}])[0].get("content") or d.get("output", "") or ""
        except Exception:
            text = str(resp or "")
    text = (text or "").strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except Exception:
                pass
        if parsed is None:
            objs = []
            for ln in text.splitlines():
                try:
                    objs.append(json.loads(ln.strip()))
                except Exception:
                    continue
            if objs:
                parsed = objs

    if parsed is None:
        print(f"Warning: could not parse model response for {filenames}. First 500 chars:\n{text[:500]}")
        return [{"file": f, "is_shooting": False, "confidence": None, "parsed_response_text": text} for f in filenames]

    if isinstance(parsed, list):
        if len(parsed) == len(filenames):
            return parsed
        by_file = {obj.get("file"): obj for obj in parsed if isinstance(obj, dict) and obj.get("file")}
        return [
            by_file.get(f, {"file": f, "is_shooting": False, "confidence": None, "parsed_response_text": text})
            for f in filenames
        ]

    if isinstance(parsed, dict):
        for k in ("results", "items", "images"):
            if k in parsed and isinstance(parsed[k], list):
                arr = parsed[k]
                if len(arr) == len(filenames):
                    return arr
                by_file = {obj.get("file"): obj for obj in arr if isinstance(obj, dict) and obj.get("file")}
                return [by_file.get(f, {"file": f, "is_shooting": False, "confidence": None}) for f in filenames]
        if len(filenames) == 1:
            return [parsed]

    return [{"file": f, "is_shooting": False, "confidence": None, "parsed_response_text": text} for f in filenames]


def batch_process_images(
    model,
    image_paths: list,
    batch_size: int = 8,
    prompt_template: str = BATCH_PROMPT_TEMPLATE,
    save_jsonl: Path = None,
    sleep_between_calls: float = 0.5,
) -> list:
    """Send images to Gemini in batches of `batch_size` and collect results."""
    out_f = open(save_jsonl, "a", encoding="utf-8") if save_jsonl else None
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        filenames = [p.name for p in batch]
        files_list_text = "\n".join(f"{j+1}. {n}" for j, n in enumerate(filenames))
        prompt = prompt_template.format(files_list=files_list_text)

        pil_images = []
        for p in batch:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Warning: could not open {p}: {e}")

        prompt_parts = [prompt] + pil_images

        try:
            resp = call_gemini(model, prompt_parts)
            records = _parse_gemini_response(resp, filenames)
        except Exception as e:
            print(f"Batch call failed for {filenames}: {e}. Creating placeholder records.")
            records = [{"file": f, "is_shooting": False, "confidence": None} for f in filenames]

        results.extend(records)
        if out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        time.sleep(sleep_between_calls)

    if out_f:
        out_f.flush()
        out_f.close()

    return results


def parallel_batch_process_images(
    model,
    image_paths: list,
    batch_size: int = 8,
    num_workers: int = 3,
    prompt_template: str = BATCH_PROMPT_TEMPLATE,
    save_jsonl: Path = None,
    sleep_between_calls: float = 0.3,
) -> list:
    """Process image batches in parallel threads. 3–5 workers recommended."""
    all_batches = [(i, image_paths[i : i + batch_size]) for i in range(0, len(image_paths), batch_size)]
    print(f"Split {len(image_paths)} images into {len(all_batches)} batches ({num_workers} workers)")

    results_by_index: dict = {}
    result_lock = threading.Lock()

    def process_single_batch(batch_index, batch_images):
        try:
            batch_results = batch_process_images(
                model=model,
                image_paths=batch_images,
                batch_size=len(batch_images),
                prompt_template=prompt_template,
                save_jsonl=None,
                sleep_between_calls=sleep_between_calls,
            )
            with result_lock:
                results_by_index[batch_index] = batch_results
            return batch_results
        except Exception as e:
            print(f"Batch {batch_index} error: {e}")
            with result_lock:
                results_by_index[batch_index] = []
            return []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_single_batch, idx, batch): (idx, batch)
            for idx, batch in all_batches
        }
        completed = 0
        for future in as_completed(future_to_batch):
            idx, batch = future_to_batch[future]
            try:
                res = future.result()
                completed += 1
                print(f"✓ Batch {completed}/{len(all_batches)} done ({len(res)} images)")
            except Exception as e:
                print(f"✗ Batch {idx} failed: {e}")

    # Write in sequential order and deduplicate
    all_results = []
    seen_files: set = set()
    writer = open(save_jsonl, "w", encoding="utf-8") if save_jsonl else None

    for idx in sorted(results_by_index.keys()):
        for rec in results_by_index[idx]:
            fname = rec.get("file", "")
            if fname and fname not in seen_files:
                seen_files.add(fname)
                all_results.append(rec)
                if writer:
                    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if writer:
        writer.close()

    print(f"✅ {len(all_results)} unique results written.")
    return all_results


# ============================================================
# SECTION 6 — Estimate timestamps from JSONL results
# ============================================================
import csv
import re
import math
from datetime import timedelta


def read_fps(video_path: str) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stderr=subprocess.STDOUT,
        ).decode().strip()
        if out and "/" in out:
            num, den = out.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(out)
        if fps > 0:
            return fps
    except Exception:
        pass
    raise RuntimeError("Could not determine FPS. Ensure OpenCV or ffprobe is available.")


def format_ts(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total = td.total_seconds()
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total % 60
    return f"{h}:{m:02d}:{s:06.3f}"


def extract_events(
    result_jsonl: Path,
    video_path: str,
    output_dir: Path,
    frame_step: int = 5,
    first_saved_frame_number: int = 5,
    max_gap_seconds: float = 0.5,
    padding_seconds: float = 1.0,
    look_ahead: int = 5,
    required_in_lookahead: int = 2,
    consume_on_detect: bool = True,
) -> list:
    """Read JSONL, group shooting frames into events, write events JSON + CSV."""
    fname_re = re.compile(r"frame_(\d+)\.jpg", re.IGNORECASE)

    fps = read_fps(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = (total_frames - 1) / fps if total_frames > 0 else 0
    cap.release()
    print(f"FPS: {fps:.3f} | Duration: {format_ts(video_duration)} ({total_frames} frames)")

    # --- Read JSONL ---
    records = []
    with open(result_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                try:
                    s = line.find("{")
                    e = line.rfind("}")
                    records.append(json.loads(line[s : e + 1]))
                except Exception:
                    records.append({"_raw_line": line})

    # --- Normalize records ---
    frames = []
    for rec in records:
        fname = None
        if isinstance(rec, dict):
            fname = rec.get("file") or (rec.get("parsed", {}) or {}).get("file")
        if not fname:
            raw = json.dumps(rec, ensure_ascii=False)
            m = fname_re.search(raw)
            fname = m.group(0) if m else None
        if not fname:
            continue

        m = fname_re.search(fname)
        if not m:
            continue
        saved_idx = int(m.group(1))
        orig_frame = first_saved_frame_number + saved_idx * frame_step
        ts_seconds = (orig_frame - 1) / fps

        parsed = None
        if isinstance(rec, dict):
            if isinstance(rec.get("parsed"), dict):
                parsed = rec["parsed"]
            elif "is_shooting" in rec:
                parsed = rec

        is_shooting = confidence = None
        if parsed:
            is_shooting = parsed.get("is_shooting")
            confidence = parsed.get("confidence")
        elif isinstance(rec, dict) and "modal_proportion" in rec:
            mp = rec.get("modal_proportion")
            if mp is not None:
                is_shooting = mp > 0.5
                confidence = float(mp)

        if isinstance(is_shooting, str):
            is_shooting = is_shooting.lower() in ("true", "1", "yes")
        try:
            confidence = float(confidence) if confidence is not None else None
            if confidence is not None:
                confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = None

        frames.append({
            "fname": fname,
            "saved_idx": saved_idx,
            "original_frame": orig_frame,
            "ts_seconds": ts_seconds,
            "is_shooting": bool(is_shooting) if is_shooting is not None else False,
            "confidence": confidence,
        })

    frames.sort(key=lambda x: x["saved_idx"])
    print(f"Normalized {len(frames)} frame entries. Shooting: {sum(1 for f in frames if f['is_shooting'])}")

    # --- Group into events ---
    saved_idx_sorted = sorted(f["saved_idx"] for f in frames)
    saved_idx_to_frame = {f["saved_idx"]: f for f in frames}
    used: set = set()
    events = []

    for pos, idx in enumerate(saved_idx_sorted):
        if idx in used:
            continue
        frame = saved_idx_to_frame[idx]
        if not frame["is_shooting"]:
            continue

        next_pos = range(pos + 1, min(pos + 1 + look_ahead, len(saved_idx_sorted)))
        look_frames = [saved_idx_to_frame[saved_idx_sorted[p]] for p in next_pos]
        num_shooting = sum(1 for f in look_frames if f["is_shooting"])

        if num_shooting >= required_in_lookahead:
            event_frames = [frame] + [f for f in look_frames if f["is_shooting"]]
            if consume_on_detect:
                for f in event_frames:
                    used.add(f["saved_idx"])
            event_frames.sort(key=lambda x: x["saved_idx"])

            start = event_frames[0]
            end = event_frames[-1]
            peak = max(event_frames, key=lambda x: x["confidence"] if x["confidence"] is not None else -1)
            conf_vals = [f["confidence"] for f in event_frames if f["confidence"] is not None]
            mean_conf = sum(conf_vals) / len(conf_vals) if conf_vals else None
            duration = (end["ts_seconds"] - start["ts_seconds"]) + 1.0 / fps

            padded_start = max(0.0, start["ts_seconds"] - padding_seconds)
            padded_end = min(video_duration, end["ts_seconds"] + padding_seconds)

            events.append({
                "detection_start_frame": start["original_frame"],
                "detection_start_time_s": start["ts_seconds"],
                "detection_start_time": format_ts(start["ts_seconds"]),
                "detection_end_frame": end["original_frame"],
                "detection_end_time_s": end["ts_seconds"],
                "detection_end_time": format_ts(end["ts_seconds"]),
                "start_frame": max(1, int(padded_start * fps) + 1),
                "start_time_s": padded_start,
                "start_time": format_ts(padded_start),
                "end_frame": min(total_frames, int(padded_end * fps) + 1),
                "end_time_s": padded_end,
                "end_time": format_ts(padded_end),
                "peak_frame": peak["original_frame"],
                "peak_time_s": peak["ts_seconds"],
                "peak_time": format_ts(peak["ts_seconds"]),
                "num_frames": len(event_frames),
                "mean_confidence": mean_conf,
                "padded_duration_s": padded_end - padded_start,
                "raw_fnames": [f["fname"] for f in event_frames],
            })

    print(f"Detected {len(events)} events.")

    # --- Write outputs ---
    stem = result_jsonl.stem
    out_json = output_dir / (stem + "_events.json")
    out_csv = output_dir / (stem + "_events.csv")

    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(
            {"video_path": str(video_path), "fps": fps,
             "params": {"frame_step": frame_step, "padding_seconds": padding_seconds},
             "events": events},
            jf, ensure_ascii=False, indent=2,
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "start_time", "start_time_s", "end_time", "end_time_s",
            "detection_start_time", "detection_start_time_s",
            "detection_end_time", "detection_end_time_s",
            "peak_time", "peak_time_s",
            "start_frame", "end_frame", "detection_start_frame",
            "detection_end_frame", "peak_frame",
            "num_frames", "mean_confidence", "padded_duration_s", "raw_fnames",
        ])
        for e in events:
            writer.writerow([
                e["start_time"], e["start_time_s"], e["end_time"], e["end_time_s"],
                e["detection_start_time"], e["detection_start_time_s"],
                e["detection_end_time"], e["detection_end_time_s"],
                e["peak_time"], e["peak_time_s"],
                e["start_frame"], e["end_frame"],
                e["detection_start_frame"], e["detection_end_frame"], e["peak_frame"],
                e["num_frames"], e["mean_confidence"], e["padded_duration_s"],
                ";".join(e["raw_fnames"]),
            ])

    print(f"Wrote events to:\n  {out_json}\n  {out_csv}")
    return events


# ============================================================
# SECTION 7 — Trim clips from events JSON
# ============================================================

def ffmpeg_exists() -> bool:
    return subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0


def trim_clips(
    video_path: Path,
    events_json: Path,
    clips_dir: Path,
    pre_pad: float = 0.5,
    post_pad: float = 0.5,
    overwrite: bool = True,
    make_concat: bool = True,
    concat_name: Path = None,
) -> list:
    """Trim the original video into one clip per event and optionally concatenate them."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    if concat_name is None:
        concat_name = clips_dir / "roughcut.mp4"

    def safe_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    with open(events_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise RuntimeError("Events JSON is not a list. Check the file format.")

    clips_made = []
    for i, ev in enumerate(events, start=1):
        start_s = safe_float(ev.get("start_time_s") or ev.get("start") or ev.get("start_seconds"))
        end_s = safe_float(ev.get("end_time_s") or ev.get("end") or ev.get("end_seconds"))
        if start_s is None or end_s is None:
            print(f"Skipping event {i}: missing start/end seconds.")
            continue

        clip_start = max(0.0, start_s - pre_pad)
        clip_end = max(clip_start + 0.001, end_s + post_pad)
        duration = clip_end - clip_start
        out_name = f"clip_{i:03d}_{int(start_s)}s_{int(duration)}s.mp4"
        out_path = clips_dir / out_name

        if out_path.exists() and not overwrite:
            print(f"Skipping existing {out_path.name}")
            clips_made.append(out_path)
            continue

        if ffmpeg_exists():
            cmd_copy = [
                "ffmpeg", "-y",
                "-ss", f"{clip_start:.3f}", "-i", str(video_path),
                "-t", f"{duration:.3f}", "-c", "copy", str(out_path),
            ]
            res = subprocess.run(cmd_copy, capture_output=True, text=True)
            if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
                print(f"[{i}/{len(events)}] Created (copy): {out_path.name}")
                clips_made.append(out_path)
                continue
            # Fallback: re-encode
            print(f"  Copy failed (rc={res.returncode}). Re-encoding...")
            cmd_reencode = [
                "ffmpeg", "-y",
                "-ss", f"{clip_start:.3f}", "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k", str(out_path),
            ]
            res2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
            if res2.returncode == 0 and out_path.exists():
                print(f"[{i}/{len(events)}] Created (re-encode): {out_path.name}")
                clips_made.append(out_path)
            else:
                print(f"Re-encode failed for event {i}:\n" + "\n".join(res2.stderr.splitlines()[-10:]))
        else:
            try:
                from moviepy.editor import VideoFileClip
            except ImportError:
                raise RuntimeError(
                    "ffmpeg not found and moviepy not installed.\n"
                    "Install ffmpeg (recommended) or run: pip install moviepy imageio-ffmpeg"
                )
            print(f"[{i}/{len(events)}] Using moviepy: {out_path.name}")
            clip = VideoFileClip(str(video_path)).subclip(clip_start, clip_end)
            clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac", verbose=False, logger=None)
            clips_made.append(out_path)

    # --- Optional concatenation ---
    if make_concat and clips_made:
        print(f"Concatenating {len(clips_made)} clips → {concat_name.name}")
        if ffmpeg_exists():
            concat_list = clips_dir / "concat_list.txt"
            with open(concat_list, "w", encoding="utf-8") as cf:
                for p in clips_made:
                    cf.write(f"file '{p.as_posix()}'\n")
            cmd_concat = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c", "copy", str(concat_name),
            ]
            resc = subprocess.run(cmd_concat, capture_output=True, text=True)
            if resc.returncode == 0 and concat_name.exists():
                print(f"Rough cut created: {concat_name.name}")
            else:
                print("Concat (copy) failed. Trying re-encode concat...")
                cmd_concat_re = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "128k", str(concat_name),
                ]
                resr = subprocess.run(cmd_concat_re, capture_output=True, text=True)
                if resr.returncode == 0:
                    print(f"Rough cut created (re-encode): {concat_name.name}")
                else:
                    print("Concatenation failed:\n" + "\n".join(resr.stderr.splitlines()[-10:]))
        else:
            try:
                from moviepy.editor import VideoFileClip, concatenate_videoclips
            except ImportError:
                raise RuntimeError("ffmpeg and moviepy both unavailable for concatenation.")
            clips_movie = [VideoFileClip(str(p)) for p in clips_made]
            final = concatenate_videoclips(clips_movie, method="compose")
            final.write_videofile(str(concat_name), codec="libx264", audio_codec="aac")
            for c in clips_movie:
                c.close()
            final.close()

    print(f"Done. Clips saved in: {clips_dir}")
    return clips_made


# ============================================================
# MAIN — Run the full pipeline
# ============================================================
if __name__ == "__main__":
    output_dir_path = Path(OUTPUT_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Validate
    validate_video(VIDEO_PATH)

    # 2. Extract frames
    saved_count = extract_frames(VIDEO_PATH, OUTPUT_DIR, frame_step=FRAME_STEP)

    # 3. Configure Gemini
    model = configure_gemini(DOTENV_PATH)

    # 4. Collect frame paths (optionally motion-filter)
    image_paths = sorted([
        p for p in output_dir_path.iterdir()
        if p.is_file() and p.name.startswith("frame_") and p.suffix.lower() == ".jpg"
    ])
    print(f"Found {len(image_paths)} frames. Running motion prefilter...")
    image_paths = motion_prefilter(image_paths, top_k=500, min_motion_thresh=15.0)
    print(f"After prefilter: {len(image_paths)} candidate frames")

    # 5. Run Gemini in parallel batches
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = output_dir_path / f"gemini_results_{timestamp}.jsonl"

    results = parallel_batch_process_images(
        model=model,
        image_paths=image_paths,
        batch_size=15,
        num_workers=4,
        save_jsonl=result_file,
        sleep_between_calls=0.3,
    )
    print(f"\n✅ Processed {len(results)} images. Results: {result_file}")

    # 6. Extract events / timestamps
    events = extract_events(
        result_jsonl=result_file,
        video_path=VIDEO_PATH,
        output_dir=output_dir_path,
        frame_step=FRAME_STEP,
        first_saved_frame_number=FRAME_STEP,
        padding_seconds=PADDING_SECONDS,
    )

    # 7. Trim clips
    events_json_path = output_dir_path / (result_file.stem + "_events.json")
    trim_clips(
        video_path=Path(VIDEO_PATH),
        events_json=events_json_path,
        clips_dir=output_dir_path / "Clips",
        make_concat=True,
    )
