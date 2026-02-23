# Video Highlight Extractor

Automatically detects and clips highlight moments from a sports video (e.g. basketball shooting events) using **Google Gemini Vision** and **OpenCV**. The pipeline:

1. Extracts frames from the video at a configurable interval
2. Optionally pre-filters frames using a motion-detection heuristic (cheap, no API cost)
3. Sends frame batches to Gemini in parallel threads and asks it to classify each frame
4. Groups consecutive shooting frames into timestamped events
5. Uses `ffmpeg` to trim the original video into individual clips and produces an optional rough-cut

---

## Requirements

- Python 3.9+
- `ffmpeg` on your PATH ([download](https://ffmpeg.org/download.html))
- A Google Gemini API key

Install Python dependencies:

```bash
pip install google-generativeai python-dotenv pillow opencv-python tqdm moviepy imageio-ffmpeg
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/video-highlight-extractor.git
cd video-highlight-extractor
```

### 2. Create your API key file

Create a file called `api.env` (do **not** commit this file) with:

```
GOOGLE_API_KEY=your_key_here
```

> `api.env` is already in `.gitignore`.

### 3. Edit the CONFIG block

Open `video_highlight_extractor.py` and update the paths at the top:

```python
VIDEO_PATH   = r"path/to/your_video.mp4"
OUTPUT_DIR   = r"path/to/output_dir"
DOTENV_PATH  = r"path/to/api.env"
```

You can also tune:

| Variable | Default | Description |
|---|---|---|
| `FRAME_STEP` | `5` | Save every Nth frame. Lower = more frames, more API calls. |
| `PADDING_SECONDS` | `1.0` | Extra seconds to add before/after each detected event clip. |
| `min_motion_thresh` | `15.0` | Motion filter sensitivity. 5=loose, 20=strict. |
| `batch_size` | `15` | Images per Gemini API call. |
| `num_workers` | `4` | Parallel threads. Keep ≤5 to avoid rate limits. |

---

## Usage

### Option A — Web UI (recommended)

```bash
python app.py
```

Opens in your browser at `http://localhost:5000`. Fill in the three paths in the sidebar, tune settings, and click **Run Pipeline**. You'll see:
- A live progress bar across all pipeline stages
- A scrolling log terminal with color-coded messages
- Live metric cards (frames extracted, filtered, shooting detections, events found)
- A table of detected events with timestamps as they're discovered

### Option B — Command line

```bash
python video_highlight_extractor.py
```

The script runs all steps end-to-end. Outputs are written to `OUTPUT_DIR`:

```
output_dir/
├── frame_00000.jpg          # extracted frames
├── frame_00001.jpg
├── ...
├── gemini_results_TIMESTAMP.jsonl      # per-frame Gemini classifications
├── gemini_results_TIMESTAMP_events.json  # grouped shooting events with timestamps
├── gemini_results_TIMESTAMP_events.csv
└── Clips/
    ├── clip_001_42s_4s.mp4
    ├── clip_002_88s_3s.mp4
    ├── ...
    └── roughcut.mp4          # all clips concatenated
```

---

## How it works

### Frame extraction
OpenCV reads the video and saves every `FRAME_STEP`th frame as a JPEG. Re-running skips this step if frames already exist.

### Motion pre-filter
Before calling Gemini (which costs money), a cheap grayscale frame-difference score filters out static/low-action frames. Only the top-K highest-motion frames are sent to the API.

### Gemini classification
Frames are sent in batches of ~15 images per API call. Each call returns a JSON array describing whether a shot is being taken, with a confidence score and shooter description. Calls are parallelized across up to 4 threads with exponential-backoff retry on failures.

### Event grouping
A shooting event is triggered when a frame is classified as `is_shooting=True` **and** at least 2 of the next 5 frames are also shooting. This reduces false positives from single-frame misclassifications.

### Clip trimming
`ffmpeg` is used to trim the original video using stream-copy (fast, lossless), falling back to re-encode if the copy produces a corrupt or empty file. All clips can be concatenated into a single rough-cut.

---

## Security notes

- **Never commit `api.env`** or any file containing your API key.
- The `.gitignore` provided excludes `api.env`, `*.env`, and the output directory.

---

## .gitignore

```
# API keys
api.env
*.env

# Output frames and results
Output/
Clips/
*.jsonl

# Python
__pycache__/
*.pyc
.venv/
```
