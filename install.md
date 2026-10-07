# Driving Video VLM

A small web application for **driving-video understanding with interactive text prompts**.

It does **not** control a vehicle. It performs:

```text
Driving video
     +
current text question
     |
rolling video window
     |
Qwen2.5-VL
     |
textual scene understanding / reasoning
```

The browser shows the uploaded video while the backend periodically analyzes successive
video windows. You can edit the prompt at any time. The next model invocation uses the
new prompt.

## Features

- Upload and display a driving/dashcam video
- Interactive text prompt
- Periodic VLM reasoning
- Configurable reasoning interval
- Configurable rolling video context window
- Configurable model sampling FPS
- Manual "analyze now" button
- Start / pause / restart
- Reasoning history
- Optional textual memory of the previous observation
- No Conda required
- No CARLA required

## Project structure

```text
driving_vlm_project/
├── app.py
├── config.py
├── vlm.py
├── video_utils.py
├── check_gpu.py
├── requirements.txt
├── requirements-basic.txt
├── .gitignore
└── README.md
```

## Recommended machine

For `Qwen/Qwen2.5-VL-7B-Instruct`, use a CUDA-capable NVIDIA GPU.

A GPU with approximately 16–24 GB VRAM is a comfortable starting point for this demo.
Actual memory usage depends strongly on video resolution, context length, sampling FPS,
PyTorch/Transformers versions and dtype.

If memory is limited:

1. Set model video sampling FPS to 0.5 or 1.
2. Reduce context window to 4–6 seconds.
3. Reduce output tokens.
4. Later add quantization.

## 1. Check system Python

Python 3.10 or 3.11 is recommended.

```bash
python3 --version
```

## 2. Create a normal venv

```bash
python3 -m venv driving_vlm_env
source driving_vlm_env/bin/activate
```

Windows PowerShell:

```powershell
python -m venv driving_vlm_env
.\driving_vlm_env\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install PyTorch

The exact PyTorch command depends on your NVIDIA driver/CUDA setup.

First inspect:

```bash
nvidia-smi
```

Then install a CUDA-enabled PyTorch build appropriate for your machine.

After installation:

```bash
python check_gpu.py
```

You want:

```text
CUDA available: True
```

## 4. Install the project dependencies

Linux:

```bash
pip install -r requirements.txt
```

If `decord` cannot be installed:

```bash
pip install -r requirements-basic.txt
```

The basic version lets `qwen-vl-utils` fall back to another supported video reader.

## 5. Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

If running on a remote server, use SSH port forwarding from your laptop:

```bash
ssh -L 7860:localhost:7860 USER@SERVER
```

Then open:

```text
http://127.0.0.1:7860
```

## First model download

The first launch/inference downloads:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

from Hugging Face. The model is cached locally afterwards.

## Suggested first settings

```text
Reasoning interval:   5 seconds
Context window:      10 seconds
Model sample FPS:     1 FPS
Max output tokens:  220
Memory:              enabled
```

## Example prompts

### General environment

```text
Describe the current driving environment from the ego vehicle's perspective.
Mention road layout, vehicles, pedestrians, traffic control and visibility.
```

### Hazards

```text
What safety-critical objects or developing hazards are visible?
Explain why each one matters.
```

### Temporal reasoning

```text
What changed compared with the previous observation?
```

### Traffic participants

```text
Identify the relevant vehicles, pedestrians and cyclists.
Describe their apparent motion relative to the ego vehicle.
```

### Intersection reasoning

```text
Explain the intersection situation.
What traffic lights, signs, vehicles or pedestrians are relevant?
```

## How periodic reasoning works

Suppose:

```text
reasoning_interval = 5 s
context_window      = 10 s
```

The analysis windows evolve roughly like:

```text
cycle 1:  0-10 s
cycle 2:  5-15 s
cycle 3: 10-20 s
cycle 4: 15-25 s
```

The model does **not** receive every source frame.

If:

```text
sample_fps = 1
context_window = 10
```

the model sees approximately 10 temporally sampled frames for each reasoning cycle.

## Interactive prompt behavior

The prompt box is live application state.

Example:

```text
0-15 s:
"Describe the environment."

User edits the prompt.

15-30 s:
"What hazards are developing?"
```

The next inference cycle receives the new question. The model does not need to restart.

## Important prototype limitation

The HTML video player's exact browser playhead is not synchronized to the backend
model cursor in this first version.

The UI displays the same source video, while the backend independently advances its
analysis timeline according to:

```text
reasoning interval
context window
```

This is intentional for a simple, robust first prototype.

For a later version, the frontend can send the browser player's exact `currentTime`
to the Python backend so the model always analyzes the segment immediately preceding
the exact visible playhead.

## Architecture

```text
                         prompt textbox
                              |
                              v
uploaded MP4 ---> Gradio interface
     |                        |
     |                        v
     |                  periodic timer
     |                        |
     v                        v
video display          backend timeline
                              |
                              v
                      extract rolling clip
                              |
                              v
                        Qwen2.5-VL
                              |
                              v
                       text reasoning
                              |
                    +---------+----------+
                    |                    |
                    v                    v
               latest answer      reasoning history
```

## Files

### `app.py`

Gradio UI, application state, timer and periodic analysis loop.

### `vlm.py`

Loads Qwen2.5-VL and performs video + text inference.

### `video_utils.py`

Reads video metadata and extracts rolling MP4 windows.

### `config.py`

Default prompts and inference settings.

## Future upgrades

A useful progression is:

### Version 2
Synchronize analysis to the browser video playhead.

### Version 3
Keep a structured scene memory:

```json
{
  "road": "...",
  "vehicles": [],
  "pedestrians": [],
  "signals": [],
  "hazards": [],
  "changes": []
}
```

### Version 4
Compare multiple models:

```text
Qwen2.5-VL
vs
DriVLMe
vs
DriveLM-compatible model
```

### Version 5
Add structured evaluation:

- object/event recognition accuracy
- hazard identification
- temporal consistency
- hallucination rate
- answer latency

## Notes

This is a research/demo scene-understanding application. Its generated text should
not be used as a safety-critical real-world driving decision system.
