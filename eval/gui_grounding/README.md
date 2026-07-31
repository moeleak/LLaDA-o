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

### Planner suitability diagnostic

`mind2web_task_history` measures single-step, state-conditioned GUI planning:
the model must infer the next action from a high-level task, previous actions,
and the current screenshot. It does not measure open-loop generation of a
complete multi-step plan. Use `mind2web` on the same action UIDs as a grounding
upper bound because that arm names the required next target directly.

The paired analyzer requires exact prediction coverage and verifies that both
arms have identical action UIDs, screenshots, splits, target actions, boxes,
and type values. Its strict success metric requires a valid parse, the correct
action type, a predicted-box center inside the target box, and normalized value
equality for `type_in`:

```bash
python -m eval.gui_grounding.analyze_planner_suitability \
  --benchmark-root /path/to/paired-benchmark \
  --direct-predictions-dir /path/to/direct-predictions \
  --planner-predictions-dir /path/to/planner-predictions \
  --output-dir /path/to/planner-analysis
```

The analyzer writes `planner_suitability.json` and
`planner_suitability.csv`, including Wilson 95% confidence intervals,
per-split and per-action results, paired success transitions, an exact McNemar
test, input hashes, and selected action UIDs.

On 2026-07-31, the Mind2Web step-750 checkpoint was tested on 30 paired
actions: ten from each official test split, with six clicks, three types, and
one hover per split. The native 16K D2F llama.cpp run used 64 generated tokens,
block length 16, temperature 0, Q3_K_M language weights, Q8_0 vision weights,
and the F16 D2F adapter. Both arms used the same images, targets, and decoding;
only the prompt protocol changed.

| Arm | Strict next-action success | Point success | Action accuracy | Parse rate | Type value accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target named directly, Q3 | 27/30 (90.0%) | 90.0% | 100.0% | 100.0% | 9/9 (100.0%) |
| Task + action history, Q3 | 2/30 (6.7%) | 10.0% | 66.7% | 90.0% | 0/9 (0.0%) |
| Task + action history, BF16 | 4/30 (13.3%) | 23.3% | 66.7% | 96.7% | 1/9 (11.1%) |

Of the 27 samples the direct arm solved, the planner arm retained only two
(7.4%). There were 25 direct-only successes, no planner-only successes, and
three failures in both arms. The strict-success drop was 83.3 percentage
points (exact paired McNemar `p = 5.96e-8`). This large paired gap isolates
next-action inference from the checkpoint's demonstrated ability to ground an
explicit target under the Q3 deployment.

As a precision sensitivity check, the same 30 planner prompts were then run
with BF16 language and vision weights plus the F32 D2F adapter. This improved
strict success to 4/30 (13.3%, Wilson 95% CI 5.3–29.7%) and point success to
7/30, but recovered the correct type value only once in nine attempts. The
BF16 arm changes weight precision as well as prompt protocol relative to the
Q3 direct arm, so it is not a causal prompt-only comparison. It does show that
the negative planner result is not explained away by Q3 quantization. Across
the Q3 pair and BF16 sensitivity arm, the diagnostic used 90 model evaluations,
below the repository's 100-sample cap.

The current step-750 checkpoint is therefore useful as a grounded executor
when an upstream component supplies an explicit next target, but is not
reliable enough to be the sole planner. A deployment should keep a separate
planner and use LLaDA-o for target grounding, or planner-tune the checkpoint
on task-history-to-next-action examples before repeating this held-out test.
The result does not establish the capability of the released base checkpoint,
a separately planner-tuned model, or open-loop multi-step planning.

## Full-page 16K–64K long-context diagnostic

The ordinary `mind2web` benchmark above is target-preserving cropped data and
does not test long context. Build the separate full-page set from the pinned
official parquet files:

```bash
python scripts/data/prepare_gui_grounding_benchmarks.py build-fullpage \
  --root /home/ma-user/work/LLaDA-o/data/mind2web-fullpage-16k-64k \
  --raw-root /path/to/pinned/mind2web/raw \
  --tokenizer /home/ma-user/work/LLaDA-o/models/lladao-gui-d2f-vllm-step1377-exact \
  --min-total-tokens 16384 \
  --max-total-tokens 65536

python scripts/data/prepare_gui_grounding_benchmarks.py validate \
  --root /home/ma-user/work/LLaDA-o/data/mind2web-fullpage-16k-64k
```

This diagnostic stores the original screenshot bytes without crop, resize,
OCR realignment, or re-encoding. Exact sequence length includes all padded
14-pixel image patches, two boundary tokens per 980-pixel tile, the tokenized
prompt, and 64 generation tokens. Samples at or below 16,384 or above 65,536
tokens are excluded and counted in the manifest.

LLaDA-o's native multimodal packing gives all visual tokens from one image a
shared LLM RoPE position, so a long visual KV sequence does not by itself test
long RoPE extrapolation. The true-long-position A/B must use
`--full-page-position-mode sequential`; this assigns every visual token an
absolute position and places prompt/generation positions after the dense
visual prefix. It is an explicit extrapolation protocol rather than the
checkpoint's native packing. Disable KV compression for that A/B so the
complete dense prefix remains resident during decoding.

The deployable comparison uses a different 16K baseline: the same original
full-page screenshot is passed through LLaDA-o's checkpoint-native
single-image resize and native position packing, while the YaRN 128K arm keeps
the unresized full-page tiles and sequential positions. Do not report an
unscaled RoPE run above 16K as the original 16K baseline.

The protocol is intentionally marked as not paper-comparable: a prompt tells
the model that the images are row-major pieces of one page and asks for
coordinates normalized to the original full screenshot. Score outputs include
the usual grounding metrics plus 16–32K, 32–48K, and 48–64K subgroups,
throughput, phase latency, active KV, peak CUDA memory, and the observed
maximum prefill/generation RoPE positions.

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

### Recover full-page grounding with prompt-only OCR crops

The ordinary 79–80% Mind2Web result and the low full-page result use different
input distributions. The former uses a target-preserving crop and crop-local
coordinates; the latter gives the checkpoint multiple unresized page tiles and
asks for one global full-page box. The current checkpoint was trained for the
first distribution.

For full-page deployment without retraining, use this hybrid protocol:

1. Match the visible target phrase in the instruction against full-page OCR.
2. Make a 980-pixel crop around the match without using the ground-truth box.
3. Run llama.cpp D2F with its native single-image, 16K checkpoint protocol.
4. Map the crop-local prediction back to full-page `[0,1000]` coordinates.
5. Use the native whole-page resize as the fallback when OCR finds no match.

This is an OCR-plus-model pipeline, not pure long-context model inference. It
does not use YaRN or KV-cache compression. OCR sees the same target text that
is present in the user instruction; target boxes are read only by the final
scorer. Setting `--model-proximity-weight 0` also prevents the full-page model
prediction from influencing which OCR match is selected.

The following is the audited two-GPU, 100-sample launch on `mllm`. The source
prediction directory is required for complete fallback rows, but with
zero proximity weight and `--policy crop` it is not used as a retrieval prior
or as the selected prediction:

```bash
WORK=/home/ma-user/work/LLaDA-o
REPO="$WORK/src/LLaDA-o"
PYTHON="$WORK/env/bin/python"
SOURCE="$WORK/data/mind2web-fullpage-16k-64k"
OCR_TARGETS="$WORK/data/mind2web-fullpage-long100-ocr-targets-v1"
SOURCE_PREDICTIONS="$WORK/results/llamacpp-fullpage-top4-yarn8-n100-25406bf"
RETRIEVAL="$WORK/results/llamacpp-ocr-noprior-n100"
CROPS="$WORK/data/mind2web-fullpage-llamacpp-ocr-crops-n100"
CROP_PREDICTIONS="$WORK/results/llamacpp-ocr-crop-native-n100"
FUSED="$WORK/results/llamacpp-ocr-crop-global-n100"

cd "$REPO"
"$PYTHON" -m eval.gui_grounding.ocr_fullpage_retrieval \
  --benchmark-root "$SOURCE" \
  --predictions-dir "$SOURCE_PREDICTIONS" \
  --output-dir "$RETRIEVAL" \
  --benchmark mind2web_fullpage \
  --limit 100 \
  --model-dir "$WORK/models/easyocr" \
  --detections-jsonl "$WORK/analysis/ocr-detections-long100.jsonl" \
  --model-proximity-weight 0

"$PYTHON" -m eval.gui_grounding.prepare_ocr_retrieval_crops \
  --benchmark-root "$SOURCE" \
  --retrieval-dir "$RETRIEVAL" \
  --output-root "$CROPS" \
  --benchmark mind2web_fullpage \
  --limit 100 \
  --crop-size 980

for SHARD in 0 1; do
  CUDA_VISIBLE_DEVICES="$SHARD" "$PYTHON" \
    -m eval.gui_grounding.run_llamacpp_native_benchmark \
    --repo "$REPO" \
    --benchmark-root "$CROPS" \
    --benchmark mind2web_fullpage \
    --binary "$WORK/llama.cpp/build-cuda/bin/llama-lladao-d2f" \
    --model "$WORK/llama.cpp-models/lladao-language-bf16.gguf" \
    --mmproj "$WORK/llama.cpp-models/lladao-mmproj-bf16.gguf" \
    --lora "$WORK/llama.cpp-models/lladao-d2f-lora-f32.gguf" \
    --output-dir "$CROP_PREDICTIONS" \
    --limit 100 \
    --shard-index "$SHARD" \
    --num-shards 2 \
    --ctx-size 16384 \
    --gpu-layers 999 \
    --threads 16 \
    --timeout 300 \
    --fail-fast &
done
wait

"$PYTHON" -m eval.gui_grounding.fuse_ocr_crop_predictions \
  --benchmark-root "$SOURCE" \
  --ocr-predictions-dir "$RETRIEVAL" \
  --crop-benchmark-root "$CROPS" \
  --crop-predictions-dir "$CROP_PREDICTIONS" \
  --output-dir "$FUSED" \
  --benchmark mind2web_fullpage \
  --limit 100 \
  --policy crop

"$PYTHON" -m eval.gui_grounding.score_benchmark \
  --benchmark-root "$OCR_TARGETS" \
  --predictions-dir "$FUSED" \
  --output-dir "$FUSED/scores-fullpage-ocr-targets-v1" \
  --benchmarks mind2web_fullpage \
  --limit 100
```

#### Tile-screening ablation

The controlled screening ablation changes only
`--tile-retrieval-topk 0` to `--tile-retrieval-topk 4`. Both arms use the same
100 full-page samples, model files, 980-pixel source tiles, overview image,
YaRN factor 8, 65,536-token context, D2F block length 16, prompt, decoding
parameters, target annotations, and prompt-only OCR fusion settings:

| Setting | Source tiles kept | Mean resident image tokens | Target-center Recall@K | OCR-fused SSR | Joint SSR | Action F1 | Parse rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense baseline, screening off | All, mean 10.86 | 33,384 | 100% | 75.00% | 75.00% | 100.00% | 100.00% |
| Top-4 screening | 4 | 15,332 | 91% | 74.00% | 74.00% | 100.00% | 100.00% |

The primary SSR above is the final prompt-only OCR-retrieval prediction scored
against the full-page benchmark target, matching the established
`Final stage = OCR/fused` report. Screening reduces the mean resident
image-token count by 54.1% for a one-point SSR decrease, while Action F1 and
parse rate remain unchanged. Recall@4 is 92% when any overlap with the target
box counts instead of requiring its center.

For diagnosis, raw model-only predictions score 4% in both arms against the
OCR-aligned target boxes. Applying the same OCR fusion and scoring against
those OCR-aligned boxes gives 61% for dense and 60% for Top-4. Keep these
diagnostic target annotations separate from the primary fused benchmark
metric above.

The dense run temporarily shared its GPUs with another benchmark, so its raw
100-sample timing is not a valid performance control. A conservative
same-ID subset excludes every dense result completed during the overlap and
the next in-flight result on each GPU. On those 60 isolated samples:

| Setting | Mean latency | P50 | P95 | Mean retrieval | Mean generation | Mean / P95 sampled card peak |
|---|---:|---:|---:|---:|---:|---:|
| Dense baseline, screening off | 275.22 s | 226.92 s | 617.88 s | 0 s | 262.82 s | 22.56 / 29.15 GiB |
| Top-4 screening | 89.76 s | 84.81 s | 142.42 s | 10.27 s | 71.09 s | 19.28 / 19.60 GiB |

On this isolated subset, Top-4 is 3.07x faster by mean end-to-end latency and
reduces the mean sampled whole-card peak by 14.6% and its P95 by 32.8%. This is
a post-hoc matched tail rather than a randomized interleaved rerun. Each sample
also starts a new binary and reloads the model, so the timing represents this
benchmark runner rather than a persistent serving process.

#### OCR-crop recovery is a separate protocol

The prompt-only OCR-crop pipeline is not a tile-screening ablation because it
changes the model input from a multi-tile full page to a checkpoint-native
single crop. When scored against the separate OCR-aligned target-box manifest,
it raises full-page-coordinate SSR from 4.00% to 74.00%. OCR retrieval accepted
93 samples; their model SSR was 78.49%. The seven whole-page fallbacks scored
14.29%. Do not compare that 74% directly with the DOM-target OCR-fused metric
in the screening table.

All quality rows above use ordered sample-ID SHA-256
`8d54d1912ae7ab966bd341df46488c843e54a0f4c16c6a898d8a5bec7d89bc4f`.
The isolated 60-sample performance subset uses
`eec7da266231427b499586ae58f746e5ebd6f38b5818f54670b7a12f3c82aa9c`.
Latency recorded by fused OCR-crop rows has scope
`crop_model_only_excludes_ocr`; include OCR time for an end-to-end deployment
measurement.

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
