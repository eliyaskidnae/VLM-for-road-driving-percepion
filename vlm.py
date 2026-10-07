from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from config import MODEL_ID, SYSTEM_PROMPT


@dataclass
class ModelStatus:
    loaded: bool
    model_id: str
    device: str


class DrivingVLM:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> ModelStatus:
        if self.loaded:
            return self.status()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU was not detected. "
                "Check `nvidia-smi` and your PyTorch CUDA installation."
            )

        print(f"Loading model: {self.model_id}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Loading in 4-bit mode for low-VRAM GPU...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Limit visual resolution to reduce VRAM usage.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self.model.eval()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Model loaded successfully.")
        print(
            "CUDA allocated:",
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
        )
        print(
            "CUDA reserved:",
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
        )

        return self.status()

    def status(self) -> ModelStatus:
        device = "not loaded"

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)

        return ModelStatus(
            loaded=self.loaded,
            model_id=self.model_id,
            device=device,
        )

    @torch.inference_mode()
    def analyze_video(
        self,
        video_path: str,
        user_prompt: str,
        sample_fps: float = 0.5,
        max_new_tokens: int = 128,
        previous_observation: str | None = None,
    ) -> str:
        if not self.loaded:
            self.load()

        question = user_prompt.strip()

        if not question:
            question = "Describe the driving scene."

        if previous_observation:
            question = (
                f"{question}\n\n"
                "Previous model observation for temporal context:\n"
                f"{previous_observation}\n\n"
                "Compare with it only when useful. "
                "Prioritize what is visible in the current video segment."
            )

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "path": video_path,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]

        inputs = None
        generated_ids = None
        trimmed_ids = None

        try:
            inputs = self.processor.apply_chat_template(
                conversation,
                fps=float(sample_fps),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = inputs.to(self.model.device)

            print(
                "Running inference | "
                f"FPS={sample_fps} | "
                f"max_new_tokens={max_new_tokens}"
            )

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
            )

            trimmed_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            answer = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            return answer

        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA ran out of memory during video inference. "
                "Reduce the context window, use sample_fps=0.5 or lower, "
                "reduce max_new_tokens, and make sure no other process is "
                "using the GPU."
            ) from exc

        finally:
            if inputs is not None:
                del inputs

            if generated_ids is not None:
                del generated_ids

            if trimmed_ids is not None:
                del trimmed_ids

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()