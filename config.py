MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

DEFAULT_PROMPT = """Describe the current driving environment from the ego vehicle's perspective.
Identify the road layout, relevant vehicles, pedestrians, cyclists, signs, traffic lights,
obstacles, visibility conditions, and any developing safety-critical situation.
Base the answer only on what is visible in the video segment."""

SYSTEM_PROMPT = """You are a visual driving-scene understanding assistant.

Analyze only the visual evidence in the supplied driving-video segment.
Focus on:
- road layout and lanes
- vehicles and their apparent motion
- pedestrians and cyclists
- traffic lights and road signs
- obstacles
- weather and visibility
- developing hazards

Do not invent objects or events that are not visible.
When uncertain, explicitly say that the evidence is uncertain.
Do not claim to be controlling the vehicle. You only describe and reason about the scene."""

DEFAULT_REASONING_INTERVAL = 5.0   # seconds between model calls
DEFAULT_CONTEXT_WINDOW = 10.0      # seconds of video sent to each model call
DEFAULT_MODEL_FPS = 1.0            # sampled video frames per second for Qwen
DEFAULT_MAX_NEW_TOKENS = 220

# Limits used by the UI.
MIN_REASONING_INTERVAL = 2.0
MAX_REASONING_INTERVAL = 60.0
MIN_CONTEXT_WINDOW = 2.0
MAX_CONTEXT_WINDOW = 30.0
