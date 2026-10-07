from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict

import gradio as gr

from config import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_FPS,
    DEFAULT_PROMPT,
    DEFAULT_REASONING_INTERVAL,
    MAX_CONTEXT_WINDOW,
    MAX_REASONING_INTERVAL,
    MIN_CONTEXT_WINDOW,
    MIN_REASONING_INTERVAL,
    MODEL_ID,
)
from video_utils import extract_clip, get_video_info
from vlm import DrivingVLM


vlm = DrivingVLM()


def new_state() -> dict:
    return {
        "video_path": None,
        "duration": 0.0,
        "cursor": 0.0,
        "running": False,
        "last_answer": "",
        "analysis_count": 0,
        "history": [],
    }


def format_status(state: dict) -> str:
    if not state.get("video_path"):
        return "### Status\nUpload a driving video to begin."

    running = state.get("running", False)
    cursor = state.get("cursor", 0.0)
    duration = state.get("duration", 0.0)
    count = state.get("analysis_count", 0)

    return (
        "### Status\n"
        f"- Analysis: **{'running' if running else 'paused'}**\n"
        f"- Analysis timeline: **{cursor:.1f}s / {duration:.1f}s**\n"
        f"- Completed reasoning cycles: **{count}**\n"
        f"- Model: `{MODEL_ID}`"
    )


def format_history(state: dict) -> str:
    history = state.get("history", [])
    if not history:
        return "### Reasoning history\nNo model output yet."

    rows = ["### Reasoning history"]
    for item in reversed(history[-12:]):
        rows.append(
            f"\n**{item['start']:.1f}s–{item['end']:.1f}s**  \n"
            f"**Prompt:** {item['prompt']}  \n\n"
            f"{item['answer']}\n"
        )
    return "\n---\n".join(rows)


def on_video_change(video_path: str | None):
    state = new_state()

    if not video_path:
        return state, format_status(state), "No video loaded.", format_history(state)

    try:
        info = get_video_info(video_path)
    except Exception as exc:
        return state, format_status(state), f"Video error: {exc}", format_history(state)

    state.update(
        {
            "video_path": video_path,
            "duration": float(info["duration"]),
            "cursor": 0.0,
            "running": False,
        }
    )

    info_text = (
        f"**Video:** {os.path.basename(video_path)}  \n"
        f"**Duration:** {info['duration']:.1f}s  \n"
        f"**Source FPS:** {info['fps']:.2f}  \n"
        f"**Resolution:** {info['width']}×{info['height']}"
    )
    return state, format_status(state), info_text, format_history(state)


def start_analysis(state: dict):
    state = dict(state or new_state())
    if not state.get("video_path"):
        return state, format_status(state), gr.Timer(active=False)

    state["running"] = True
    return state, format_status(state), gr.Timer(active=True)


def pause_analysis(state: dict):
    state = dict(state or new_state())
    state["running"] = False
    return state, format_status(state), gr.Timer(active=False)


def restart_analysis(state: dict):
    state = dict(state or new_state())
    state["cursor"] = 0.0
    state["running"] = False
    state["last_answer"] = ""
    state["analysis_count"] = 0
    state["history"] = []
    return (
        state,
        format_status(state),
        "### Latest reasoning\nRestarted. Press **Start analysis**.",
        format_history(state),
        gr.Timer(active=False),
    )


def update_timer_interval(reasoning_interval: float):
    value = max(
        MIN_REASONING_INTERVAL,
        min(MAX_REASONING_INTERVAL, float(reasoning_interval)),
    )
    return gr.Timer(value=value)


def analyze_once(
    state: dict,
    prompt: str,
    context_window: float,
    sample_fps: float,
    max_new_tokens: int,
    use_memory: bool,
):
    state = dict(state or new_state())

    if not state.get("video_path"):
        return (
            state,
            "### Latest reasoning\nUpload a video first.",
            format_status(state),
            format_history(state),
        )

    try:
        duration = float(state["duration"])
        cursor = min(float(state["cursor"]), duration)

        # At the beginning, analyze [0, context_window].
        # Afterwards use a rolling window ending at the analysis cursor.
        if cursor <= 0.0:
            end_sec = min(float(context_window), duration)
            start_sec = 0.0
        else:
            end_sec = min(cursor, duration)
            start_sec = max(0.0, end_sec - float(context_window))

        # If the previous tick reached the end, stop.
        if duration <= 0.0 or (cursor >= duration and state.get("analysis_count", 0) > 0):
            state["running"] = False
            return (
                state,
                "### Latest reasoning\nReached the end of the video.",
                format_status(state),
                format_history(state),
            )

        clip_path = extract_clip(
            state["video_path"],
            start_sec=start_sec,
            end_sec=end_sec,
        )

        previous = state.get("last_answer") if use_memory else None

        started = time.time()
        answer = vlm.analyze_video(
            video_path=clip_path,
            user_prompt=prompt,
            sample_fps=float(sample_fps),
            max_new_tokens=int(max_new_tokens),
            previous_observation=previous,
        )
        latency = time.time() - started

        try:
            os.remove(clip_path)
        except OSError:
            pass

        state["last_answer"] = answer
        state["analysis_count"] = int(state.get("analysis_count", 0)) + 1
        state["history"] = list(state.get("history", []))
        state["history"].append(
            {
                "start": start_sec,
                "end": end_sec,
                "prompt": prompt.strip() or "Describe the driving scene.",
                "answer": answer,
            }
        )

        latest = (
            "### Latest reasoning\n"
            f"**Video window:** {start_sec:.1f}s–{end_sec:.1f}s  \n"
            f"**Inference time:** {latency:.1f}s\n\n"
            f"{answer}"
        )

        return state, latest, format_status(state), format_history(state)

    except Exception as exc:
        state["running"] = False
        return (
            state,
            "### Latest reasoning\n"
            f"**Inference failed:** `{type(exc).__name__}: {exc}`",
            format_status(state),
            format_history(state),
        )


def timer_tick(
    state: dict,
    prompt: str,
    reasoning_interval: float,
    context_window: float,
    sample_fps: float,
    max_new_tokens: int,
    use_memory: bool,
):
    state = dict(state or new_state())

    if not state.get("running"):
        return (
            state,
            gr.skip(),
            format_status(state),
            gr.skip(),
            gr.Timer(active=False),
        )

    # Move the backend analysis timeline forward.
    # The browser video playback is visual; this backend cursor determines
    # which segment is sent to the VLM.
    current = float(state.get("cursor", 0.0))
    if current <= 0.0:
        state["cursor"] = min(float(context_window), float(state["duration"]))
    else:
        state["cursor"] = min(
            current + float(reasoning_interval),
            float(state["duration"]),
        )

    state, latest, status, history = analyze_once(
        state,
        prompt,
        context_window,
        sample_fps,
        max_new_tokens,
        use_memory,
    )

    active = bool(state.get("running")) and float(state["cursor"]) < float(state["duration"])
    if not active:
        state["running"] = False
        status = format_status(state)

    return state, latest, status, history, gr.Timer(active=active)


def manual_analyze(
    state: dict,
    prompt: str,
    context_window: float,
    sample_fps: float,
    max_new_tokens: int,
    use_memory: bool,
):
    state = dict(state or new_state())

    if not state.get("video_path"):
        return (
            state,
            "### Latest reasoning\nUpload a video first.",
            format_status(state),
            format_history(state),
        )

    if float(state.get("cursor", 0.0)) <= 0:
        state["cursor"] = min(float(context_window), float(state["duration"]))

    return analyze_once(
        state,
        prompt,
        context_window,
        sample_fps,
        max_new_tokens,
        use_memory,
    )


CSS = """
#header-note {
    border-left: 4px solid var(--primary-500);
    padding-left: 12px;
}
"""


with gr.Blocks(title="Driving Video VLM", css=CSS) as demo:
    state = gr.State(new_state())

    gr.Markdown(
        """
# Driving Video VLM
Upload a dashcam/driving video, ask a question about the scene, and let the VLM
re-analyze a rolling video window periodically. You can change the prompt at any time;
the next reasoning cycle uses the new prompt.
""",
        elem_id="header-note",
    )

    with gr.Row():
        with gr.Column(scale=3):
            video = gr.Video(
                label="Driving video",
                format="mp4",
            )
            video_info = gr.Markdown("No video loaded.")

            with gr.Row():
                start_btn = gr.Button("▶ Start analysis", variant="primary")
                pause_btn = gr.Button("⏸ Pause")
                restart_btn = gr.Button("↺ Restart analysis")

            status = gr.Markdown("### Status\nUpload a driving video to begin.")

        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Question / reasoning instruction",
                value=DEFAULT_PROMPT,
                lines=7,
                placeholder="Example: What are the main hazards developing in the scene?",
            )

            with gr.Row():
                analyze_btn = gr.Button("Analyze current window now", variant="secondary")

            latest = gr.Markdown(
                "### Latest reasoning\nUpload a video and press **Start analysis**."
            )

    with gr.Accordion("Analysis settings", open=False):
        with gr.Row():
            reasoning_interval = gr.Slider(
                minimum=MIN_REASONING_INTERVAL,
                maximum=MAX_REASONING_INTERVAL,
                value=DEFAULT_REASONING_INTERVAL,
                step=1,
                label="Reasoning interval (seconds)",
                info="How often periodic VLM inference runs.",
            )
            context_window = gr.Slider(
                minimum=MIN_CONTEXT_WINDOW,
                maximum=MAX_CONTEXT_WINDOW,
                value=DEFAULT_CONTEXT_WINDOW,
                step=1,
                label="Video context window (seconds)",
                info="Length of recent video sent to each inference call.",
            )

        with gr.Row():
            sample_fps = gr.Slider(
                minimum=0.5,
                maximum=4.0,
                value=DEFAULT_MODEL_FPS,
                step=0.5,
                label="Model video sampling FPS",
                info="Lower values reduce GPU memory and latency.",
            )
            max_new_tokens = gr.Slider(
                minimum=64,
                maximum=512,
                value=DEFAULT_MAX_NEW_TOKENS,
                step=16,
                label="Max output tokens",
            )

        use_memory = gr.Checkbox(
            value=True,
            label="Give previous observation to the next reasoning cycle",
            info="Enables simple short-term textual memory for questions such as 'What changed?'.",
        )

    history = gr.Markdown("### Reasoning history\nNo model output yet.")

    timer = gr.Timer(value=DEFAULT_REASONING_INTERVAL, active=False)

    video.change(
        fn=on_video_change,
        inputs=[video],
        outputs=[state, status, video_info, history],
        queue=False,
    )

    start_btn.click(
        fn=start_analysis,
        inputs=[state],
        outputs=[state, status, timer],
        queue=False,
    )

    pause_btn.click(
        fn=pause_analysis,
        inputs=[state],
        outputs=[state, status, timer],
        queue=False,
    )

    restart_btn.click(
        fn=restart_analysis,
        inputs=[state],
        outputs=[state, status, latest, history, timer],
        queue=False,
    )

    reasoning_interval.change(
        fn=update_timer_interval,
        inputs=[reasoning_interval],
        outputs=[timer],
        queue=False,
    )

    analyze_btn.click(
        fn=manual_analyze,
        inputs=[
            state,
            prompt,
            context_window,
            sample_fps,
            max_new_tokens,
            use_memory,
        ],
        outputs=[state, latest, status, history],
        concurrency_id="vlm-inference",
        concurrency_limit=1,
    )

    timer.tick(
        fn=timer_tick,
        inputs=[
            state,
            prompt,
            reasoning_interval,
            context_window,
            sample_fps,
            max_new_tokens,
            use_memory,
        ],
        outputs=[state, latest, status, history, timer],
        concurrency_id="vlm-inference",
        concurrency_limit=1,
        show_progress="minimal",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
