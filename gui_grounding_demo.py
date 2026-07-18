#!/usr/bin/env python3
"""Interactive GUI-grounding demo for an LLaDA-o understanding checkpoint."""

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw

from demo_pipeline import LLaDAMultimodalDemo


ACTION_RE = re.compile(
    r"(lclick|hover|type_in)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH"),
        help="Base model directory containing configs and tokenizer assets.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("LLADAO_GUI_CHECKPOINT"),
        help="Fine-tuned ema.safetensors file or checkpoint directory.",
    )
    parser.add_argument("--max-mem-per-gpu", default="90GiB")
    parser.add_argument(
        "--offload-dir",
        default=os.environ.get("LLADAO_DEMO_OFFLOAD_DIR", "/tmp/lladao-gui-offload"),
    )
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or MODEL_PATH is required")
    if not args.checkpoint:
        parser.error("--checkpoint or LLADAO_GUI_CHECKPOINT is required")
    return args


def draw_prediction(image: Image.Image, text: str) -> Tuple[Image.Image, str]:
    output = image.convert("RGB").copy()
    match = ACTION_RE.search(text)
    if match is None:
        return output, "No action/bounding box could be parsed from the response."

    action = match.group(1)
    coords = [max(0, min(1000, int(value))) for value in match.groups()[1:]]
    x1, y1, x2, y2 = coords
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    width, height = output.size
    box = (
        round(left * width / 1000),
        round(top * height / 1000),
        round(right * width / 1000),
        round(bottom * height / 1000),
    )
    draw = ImageDraw.Draw(output)
    line_width = max(2, round(min(width, height) / 200))
    draw.rectangle(box, outline=(255, 0, 0), width=line_width)
    label = f"{action} [{x1},{y1},{x2},{y2}]"
    label_box = draw.textbbox((box[0], box[1]), label)
    label_y = max(0, box[1] - (label_box[3] - label_box[1]) - 2 * line_width)
    label_box = draw.textbbox((box[0], label_y), label)
    draw.rectangle(
        (
            label_box[0] - line_width,
            label_box[1] - line_width,
            label_box[2] + line_width,
            label_box[3] + line_width,
        ),
        fill=(255, 0, 0),
    )
    draw.text((box[0], label_y), label, fill=(255, 255, 255))
    return output, f"Parsed {label}; pixel box={box}."


def build_demo(model: LLaDAMultimodalDemo, checkpoint: Path) -> gr.Blocks:
    def predict(
        image: Optional[Image.Image],
        prompt: str,
        block_length: int,
        steps_per_block: int,
        confidence_threshold: float,
    ) -> Tuple[str, Optional[Image.Image], str]:
        if image is None:
            raise gr.Error("Upload a GUI screenshot first.")
        if not prompt.strip():
            raise gr.Error("Enter a GUI instruction first.")

        result = model.understand(
            image,
            prompt.strip(),
            block_length=int(block_length),
            steps_per_block=int(steps_per_block),
            max_blocks=1,
            confidence_threshold=float(confidence_threshold),
        )
        annotated, parsed = draw_prediction(image, result["text"])
        details = (
            f"{parsed}\n"
            f"Latency: {result['elapsed_seconds']:.2f}s; "
            f"convergence steps: {result['convergence_steps']}; "
            f"valid tokens: {result['valid_tokens']}/{result['total_tokens']}"
        )
        return result["text"], annotated, details

    with gr.Blocks(title="LLaDA-o GUI Grounding") as app:
        gr.Markdown(
            "# LLaDA-o GUI Grounding\n"
            f"Checkpoint: `{checkpoint}`\n\n"
            "Upload a screenshot and describe the element or next GUI action. "
            "Coordinates are predicted on a `[0,1000]` scale."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="GUI screenshot")
                prompt_input = gr.Textbox(
                    label="Instruction",
                    placeholder='Locate and click the UI element described as: "Settings".',
                    lines=3,
                )
                with gr.Accordion("Inference settings", open=False):
                    block_length = gr.Slider(16, 128, value=64, step=16, label="Block length")
                    steps_per_block = gr.Slider(
                        16, 128, value=64, step=16, label="Diffusion steps"
                    )
                    confidence_threshold = gr.Slider(
                        0.0, 1.0, value=0.95, step=0.01, label="Confidence threshold"
                    )
                run_button = gr.Button("Ground", variant="primary")
            with gr.Column():
                response_output = gr.Textbox(label="Model response")
                image_output = gr.Image(type="pil", label="Predicted bounding box")
                details_output = gr.Textbox(label="Details")

        run_button.click(
            predict,
            inputs=[
                image_input,
                prompt_input,
                block_length,
                steps_per_block,
                confidence_threshold,
            ],
            outputs=[response_output, image_output, details_output],
        )
    return app


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    print(f"Loading GUI grounding checkpoint: {checkpoint}", flush=True)
    model = LLaDAMultimodalDemo.from_pretrained(
        model_path=args.model_path,
        checkpoint_path=checkpoint,
        enable_visual_generation=False,
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_dir=args.offload_dir,
    )
    app = build_demo(model, checkpoint)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
