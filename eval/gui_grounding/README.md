# GUI-grounding benchmark

This evaluation targets the protocol in [Towards GUI Agents: Vision-Language
Diffusion Models for GUI Grounding](https://arxiv.org/abs/2603.26211). The
paper evaluates single-step action and bounding-box generation on:

1. Mind2Web test;
2. ScreenSpot-Web-Text;
3. ScreenSpot-Web-Icon;
4. VisualWebArena.

The main paper setting is a generation length, block length, and diffusion-step
count of 64. Coordinates are normalized to `[0,1000]`. The scorer reports the
paper's point-in-box SSR (the center of the predicted box must fall in the
ground-truth box), action-type F1, synchronized inference latency, and measured
denoising convergence steps.

## Reproducibility boundary

The paper does not publish evaluation code, prompts, sample IDs, crop seeds,
OCR realignment code, or its static single-step VisualWebArena extraction.
Official VisualWebArena is an online multi-step environment, so it cannot be
silently substituted for that unpublished static set.

The preparation script therefore:

- pins official Multimodal-Mind2Web and ScreenSpot revisions;
- includes all three official Mind2Web test splits because the paper only says
  “test split”;
- makes `mind2web` a target-explicit, single-step grounding benchmark by
  converting the public `target_action_reprs` field into direct instructions
  such as `Click on Track & Field.`; this matches the task shape shown in the
  paper, although the authors' exact prompt wording is unpublished;
- also emits `mind2web_task_history`, which retains the old high-level task and
  action-history prompt as a planning-plus-grounding diagnostic; its score is
  intentionally not treated as paper-comparable;
- applies the same deterministic 1280-pixel target-preserving Mind2Web crop
  used by this repository's fine-tuning pipeline;
- splits the official ScreenSpot web examples into text and icon subsets;
- writes every sample and source decision to a checksummed manifest;
- leaves VisualWebArena unavailable unless an explicit static export is
  supplied, and labels an imported export as not proven identical to the
  paper's subset.

This gives a reproducible paper-aligned benchmark without presenting an
unknown custom subset as an exact reproduction.

## Prepare data on Clariden

From the repository root:

```bash
sbatch scripts/slurm/prepare_gui_grounding_benchmarks.sbatch
```

The default destination is:

```text
$SCRATCH/datasets/lladao_gui_benchmarks/
├── manifest.json
├── validation.json
├── samples/
│   ├── mind2web.jsonl
│   ├── mind2web_task_history.jsonl
│   ├── screenspot_web_text.jsonl
│   └── screenspot_web_icon.jsonl
└── images/
```

Rebuild prepared outputs while retaining downloaded source files:

```bash
FORCE_REBUILD=1 \
sbatch scripts/slurm/prepare_gui_grounding_benchmarks.sbatch
```

An independently obtained static VisualWebArena export can be imported with:

```bash
VISUALWEBARENA_JSONL=/absolute/path/vwa.jsonl \
FORCE_REBUILD=1 \
sbatch scripts/slurm/prepare_gui_grounding_benchmarks.sbatch
```

Each VWA JSONL object must contain `image`, `instruction` or `prompt`, and a
target box. The preferred target field is `target_bbox_1000`; alternatively,
provide `bbox` and `bbox_format` (`xyxy_pixels`, `xywh_pixels`, `xyxy_0_1`, or
`xyxy_1000`). `target_action` defaults to `lclick`.

## Smoke test

Run eight samples from each available benchmark on four GPUs:

```bash
EVAL_LIMIT=8 \
OUTPUT_DIR="$SCRATCH/runs/lladao_gui_benchmark/smoke" \
sbatch --time=01:00:00 scripts/slurm/eval_gui_grounding_benchmarks.sbatch
```

## Full evaluation

```bash
CHECKPOINT="$SCRATCH/runs/lladao_gui_120k/checkpoints/0010000/ema.safetensors" \
OUTPUT_DIR="$SCRATCH/runs/lladao_gui_benchmark/step-0010000/s64-b64-ct095" \
sbatch scripts/slurm/eval_gui_grounding_benchmarks.sbatch
```

The job launches one independent model replica per GPU. Prediction shards are
append-only and resumable. Re-submit with the same `OUTPUT_DIR` after a timeout
to process only missing samples.

Useful overrides:

```bash
BENCHMARKS=mind2web,screenspot_web_text,screenspot_web_icon
BLOCK_LENGTH=64
DIFFUSION_STEPS=64
CONFIDENCE_THRESHOLD=0.95  # use "none" for fixed-step decoding
WARMUP=1
```

To quantify the protocol effect on the exact same screenshots and checkpoint,
run the target-explicit benchmark and the legacy planning prompt together in a
fresh output directory:

```bash
BENCHMARKS=mind2web,mind2web_task_history \
OUTPUT_DIR="$SCRATCH/runs/lladao_gui_benchmark/mind2web-protocol-ab" \
sbatch scripts/slurm/eval_gui_grounding_benchmarks.sbatch
```

Do not reuse predictions produced before changing a prompt protocol: shards
are append-only and resume by sample ID.

Results are written to:

```text
<OUTPUT_DIR>/scores/results.json
<OUTPUT_DIR>/scores/results.csv
```

The JSON includes both point-only SSR and joint step success (correct action
and correct point), as well as three F1 variants. `Action F1 (%)` in the CSV is
macro F1 over action classes present in the ground truth. This is reported
alongside macro F1 over all three fixed labels because ScreenSpot is click-only
and the paper's “macro F1 over three classes” description is otherwise
inconsistent with its near-100 ScreenSpot F1 values.

## Summarize a Table 3 checkpoint sweep

Use the sweep summarizer to audit complete checkpoints under one fixed decoding
configuration. It reports the combined Mind2Web result, all three official test
splits, ScreenSpot diagnostics, the paper gap, and—when supplied—the same
predictions rescored against the original DOM target boxes:

```bash
sbatch scripts/slurm/summarize_gui_grounding_table3.sbatch
```

The Slurm entry point uses the project container instead of Clariden's legacy
login-node Python. From an already active project environment, the equivalent
direct command is:

```bash
python -m eval.gui_grounding.summarize_table3_sweep \
  --results-root "$SCRATCH/runs/lladao_gui_benchmark/table3-m2w-only" \
  --dom-benchmark-root "$SCRATCH/datasets/lladao_gui_benchmarks" \
  --steps-per-epoch 475.1 \
  --require-steps 250,500,750,1000
```

The command writes `table3_sweep.json` and `table3_sweep.csv` beneath the
results root. Once the run finishes, pass `--primary-step 4750`; intermediate
checkpoints are explicitly labeled as training diagnostics, not candidates to
select using test-set performance. The paper reports 83.31% SSR and 99% action
F1 for its highlighted Mind2Web-only, cropped, OCR-target, 10-epoch row.

## Paper reference values

For the paper's LLaDA-V 8B linear-masking model trained on its 120K mixture,
Table 4 reports:

| Benchmark | SSR (%) | Action F1 (%) | Avg latency (s) | Conv. steps |
|---|---:|---:|---:|---:|
| Mind2Web | 82.4 | 98.5 | 3.02 | 16.0 |
| ScreenSpot-Web-Icon | 57.8 | 99.5 | 3.36 | 18.0 |
| ScreenSpot-Web-Text | 73.5 | 99.1 | 3.20 | 17.0 |
| VisualWebArena | 61.4 | 99.4 | 3.05 | 16.5 |

Hardware, LLaDA-o architecture, training data realization, preprocessing, and
the unpublished evaluation details differ, so these are context rather than a
claim of directly comparable reproduction.
