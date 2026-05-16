# Evaluation

This directory contains image-generation evaluation scripts for LLaDA-o:

- `gen/geneval/`: GenEval prompts and Mask2Former/OpenCLIP scoring code.
- `gen/dpg_bench/`: DPG-Bench question CSV, grid builder, and mPLUG-based scorer.
- `gen/gen_images_mp_dllm.py`: distributed LLaDA-o text-to-image generation used by both benchmarks.

All commands below assume they are run from the repository root.

## 1. Prepare the LLaDA-o Checkpoint

Download the released checkpoint and point the evaluation scripts to the local directory:

```bash
huggingface-cli download GSAI-ML/LLaDA-o \
  --local-dir /path/to/local/GSAI-ML-LLaDA-o

export LLADAO_MODEL_PATH=/path/to/local/GSAI-ML-LLaDA-o
```

The model directory must contain `llm_config.json`, `vit_config.json`, `ae.safetensors`, tokenizer files, and `ema.safetensors` or `ema.safetensors.index.json`.

## 2. Install GenEval Dependencies

GenEval scoring uses OpenCLIP and MMDetection 2.x. Install the extra packages after the base repository environment is ready:

```bash
pip install open-clip-torch
pip install clip-benchmark
pip install --upgrade setuptools

sudo pip install -U openmim
sudo mim install mmengine mmcv-full==1.7.2

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 2.x
pip install -v -e .
cd -
```

Download the Mask2Former detector weights:

```bash
bash eval/gen/geneval/evaluation/download_models.sh \
  ./eval/gen/geneval/evaluation/models/mask2former

export GENEVAL_DETECTOR_MODEL_PATH=./eval/gen/geneval/evaluation/models/mask2former
```

OpenCLIP defaults to the public `laion2b_s32b_b82k` pretrained tag and will download weights through `open_clip` when needed. For offline runs, download the weight file yourself and set:

```bash
huggingface-cli download laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
  open_clip_pytorch_model.bin \
  --local-dir ./eval/gen/geneval/evaluation/models/open_clip

export GENEVAL_CLIP_PRETRAINED=./eval/gen/geneval/evaluation/models/open_clip/open_clip_pytorch_model.bin
```

## 3. Run GenEval

Generate images and score them:

```bash
bash scripts/eval/run_geneval_dllm.sh "$LLADAO_MODEL_PATH"
```

Default outputs are written under the checkpoint directory:

```text
<LLADAO_MODEL_PATH>/gen_eval_images/
<LLADAO_MODEL_PATH>/geneval_results_long.jsonl
<LLADAO_MODEL_PATH>/geneval_results.txt
```

Useful overrides:

```bash
LLADAO_GENEVAL_GPUS=8
GENEVAL_METADATA_FILE=./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl
GENEVAL_NUM_IMAGES=4
GENEVAL_BATCH_SIZE=1
GENEVAL_RESOLUTION=1024
GENEVAL_MAX_LATENT_SIZE=64
GENEVAL_SKIP_GENERATION=0
GENEVAL_DETECTOR_MODEL_PATH=./eval/gen/geneval/evaluation/models/mask2former
GENEVAL_CLIP_PRETRAINED=laion2b_s32b_b82k
```

If `GENEVAL_SKIP_GENERATION=1`, the script skips image generation and only scores the existing `gen_eval_images/` directory.

## 4. Install DPG-Bench Dependencies

DPG-Bench scoring follows the dependency list from the ELLA DPG-Bench setup:

```bash
pip install -r https://raw.githubusercontent.com/TencentQQGYLab/ELLA/main/requirements-for-dpg_bench.txt
```

The scorer uses the ModelScope mPLUG VQA model. By default the script passes the public model id:

```text
damo/mplug_visual-question-answering_coco_large_en
```

ModelScope downloads it on first use. If you need an offline local copy, pre-download it with ModelScope and set `DPG_SCORE_VQA_MODEL_PATH` to the downloaded folder:

```bash
python -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('damo/mplug_visual-question-answering_coco_large_en', cache_dir='./eval/gen/dpg_bench/models'))"

export DPG_SCORE_VQA_MODEL_PATH=/path/printed/by/snapshot_download
```

## 5. Run DPG-Bench

The generation metadata JSONL must contain one prompt per line. Each line should include:

```json
{"filename": "partiprompts97", "prompt": "a detailed text-to-image prompt"}
```

Run generation, build 2x2 image grids, and score:

```bash
bash scripts/eval/run_dpg_dllm.sh "$LLADAO_MODEL_PATH" /path/to/dpg_prompts.jsonl
```

Default outputs are written under the checkpoint directory:

```text
<LLADAO_MODEL_PATH>/dpg_eval_images/
<LLADAO_MODEL_PATH>/dpg_eval_images_grid/
<LLADAO_MODEL_PATH>/dpg_eval_images_grid/dpg-bench_results.txt
```

If raw samples already exist, skip generation and only build grids plus score:

```bash
DPG_SKIP_GENERATION=1 \
bash scripts/eval/run_dpg_dllm.sh /path/to/dpg_eval_images
```

If 2x2 grid images already exist, skip both generation and grid building:

```bash
DPG_SKIP_GENERATION=1 DPG_SKIP_GRID=1 \
bash scripts/eval/run_dpg_dllm.sh /path/to/dpg_eval_images_grid
```

Useful overrides:

```bash
LLADAO_DPG_GPUS=8
DPG_METADATA_FILE=/path/to/dpg_prompts.jsonl
DPG_OUTPUT_DIR=/path/to/raw_dpg_images
DPG_NUM_IMAGES=4
DPG_BATCH_SIZE=1
DPG_RESOLUTION=1024
DPG_MAX_LATENT_SIZE=64
DPG_CSV_PATH=./eval/gen/dpg_bench/dpg_bench.csv
DPG_GRID_OUTPUT_DIR=/path/to/dpg_eval_images_grid
DPG_GRID_WORKERS=8
DPG_GRID_FORMAT=png
DPG_GRID_PNG_COMPRESS_LEVEL=1
DPG_GRID_OVERWRITE=0
DPG_SCORE_PROCESSES=8
DPG_SCORE_VQA_MODEL_PATH=damo/mplug_visual-question-answering_coco_large_en
DPG_SCORE_RESULT_PATH=/path/to/dpg-bench_results.txt
```

Keep `DPG_GRID_FORMAT=png` for official scoring. JPEG grids are supported only for quick local checks.

## Notes

- Both launchers use `torchrun` for image generation and expect CUDA GPUs.
- DPG scoring uses `accelerate launch`; reduce `DPG_SCORE_PROCESSES` if each VQA process does not fit in memory.
- If the model path matches `variant*_2`, the launchers automatically add `--reg`; in that case set `LLADAO_REPA_MODEL_PATH=/path/to/dinov3`.
- GenEval code is adapted from `djghosh13/geneval`; DPG-Bench scoring is adapted from `TencentQQGYLab/ELLA`.
