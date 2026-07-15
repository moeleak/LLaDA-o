# GUI grounding Table 1 data

This pipeline prepares the six source/domain buckets in Table 1 of
[Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](https://arxiv.org/abs/2603.26211):

| Domain | Source | Paper Table 1 | Prepared grounding rows |
| --- | --- | ---: | ---: |
| Web | Mind2Web | 20,000 | 7,341 |
| Web | WebLINX | 20,000 | 20,000 |
| Web | OS-Atlas | 20,000 | 20,000 |
| Mobile | OS-Atlas | 20,000 | 20,000 |
| Mobile | RICO Widget Caption | 20,000 | 20,000 |
| Desktop | OS-Atlas | 20,000 | 20,000 |
| | **Total** | **120,000** | **107,341** |

## Mind2Web count

The published Multimodal-Mind2Web train split contains **7,775 raw rows**.
All 7,775 are scanned exactly once and **no duplicate samples or synthetic crop
variants are added**. Of those rows, 7,341 have a valid target bounding box
that intersects the corresponding screenshot and are included in the
fine-tuning Parquet files. The remaining 434 cannot provide valid coordinate
supervision in the published data:

- 413 have no positive candidate bounding box;
- 12 have a zero-width or zero-height target box;
- 9 have a target box entirely outside the supplied screenshot.

These rows are excluded instead of assigning fabricated coordinates. Their IDs
and rejection reasons are recorded in
`parquet/mind2web/rejections.json`. Thus, "7,775 Mind2Web rows" refers to the
complete upstream train split, while 7,341 is the usable coordinate-grounding
count consumed by LLaDA-o.

The paper does not release sample IDs, the random crop seed/parameters, or its
OCR-to-target realignment implementation. The generated corpus is therefore a
deterministic approximation, not a byte-identical reproduction. It uses only
published training data, pins every repository revision, keeps full images for
all sources except target-preserving Mind2Web crops, normalizes boxes to
`[0,1000]`, and records provenance in every row plus `manifest.json`.

WebLINX has 13,515 train actions that can be mapped to a target element box; it
keeps all of them and adds prompt variants with longer action histories until
it reaches the paper's 20K bucket. These variants retain the same image,
target, and action, and are explicitly marked in per-row provenance.

## Prepare

The environment needs `datasets`, `huggingface_hub`, `pyarrow`, Pillow, and the
official `weblinx==0.3.2` parser. They are installed into the persistent
`${SCRATCH}/venvs/lladao-ngc-25.01-v2` environment by
`scripts/bootstrap_lladao_env.sh`; Conda is not used. If the current interactive
shell was started with plain `--pty bash`, source the bootstrap once before
running Python. On aarch64 nodes the bootstrap installs the API-compatible
`decord2` wheel because the original `decord==0.6.0` project does not publish
an aarch64 wheel:

The included `lladao.toml` uses NGC PyTorch 25.01. Do not substitute the 24.12
image: its pre-release PyTorch identifies as 2.6 but is missing an API required
by `transformers==4.56.2`.

```bash
source scripts/bootstrap_lladao_env.sh
python -c 'import datasets, pyarrow, weblinx; print("data environment OK")'

python scripts/data/prepare_gui_grounding.py download \
  --root "${GUI_ROOT}"

python scripts/data/prepare_gui_grounding.py build \
  --root "${GUI_ROOT}"

python scripts/data/prepare_gui_grounding.py validate \
  --root "${GUI_ROOT}" \
  --deep
```

All commands are idempotent except `build`, which refuses to overwrite an
existing source directory unless `--force` is supplied. Downloads resume via
the Hugging Face local-directory cache. WebLINX screenshots use the Git-LFS
batch protocol to avoid one Hub API request per image; completed files are
reused and every new image is checked against its LFS SHA-256.

## Use with LLaDA-o

```bash
export LLADAO_GUI_GROUNDING_DIR=/home/ubuntu/datasets/lladao_gui_120k/parquet
export DATASET_CONFIG_FILE=data/configs/gui_grounding_table1.yaml
```

For GUI-only fine-tuning, launch the training entry point with image generation
disabled and multimodal masked-prediction SFT enabled:

```text
--visual_gen False
--visual_und True
--visual_und_sft True
--merge_vit_text_segments True
--dataset_config_file data/configs/gui_grounding_table1.yaml
```

## Fine-tune with Slurm

The repository includes three launchers:

- `scripts/train_gui_grounding_120k.sh` contains the model and data arguments;
- `scripts/slurm/train_gui_grounding_120k.sbatch` requests Slurm resources and
  starts one distributed launcher per node;
- `scripts/slurm/gui-120k-grounding-finetune.sh` submits the standard one-node
  Clariden production job from a login node.

The defaults target one Clariden GH200 node with four GPUs and perform
full-model BF16 fine-tuning with FSDP `FULL_SHARD`. The training and EMA models
are constructed and sharded sequentially so each rank holds at most one full
FP32 model in host memory during startup. The launch still requests the Clariden
maximum usable node memory (`450G`) and exclusive access to leave room for
checkpoint shards and Python workers. Cluster partition, account, GPU type, and
wall-time policies remain site-specific.

### 1. Check the inputs

The released model directory must contain at least `llm_config.json`,
`vit_config.json`, tokenizer files, and either `ema.safetensors` or
`ema.safetensors.index.json` plus all shards referenced by the index. The data
directory is the `parquet/` directory created by the preparation commands:

```text
${SCRATCH}/models/GSAI-ML-LLaDA-o/
${SCRATCH}/datasets/lladao_gui_120k/parquet/
```

These paths and `${SCRATCH}/runs/lladao_gui_120k` are exported as
`MODEL_PATH`, `LLADAO_GUI_GROUNDING_DIR`, and `RESULTS_DIR` by `lladao.toml`.

Run the deep validation once before allocating GPUs:

```bash
python scripts/data/prepare_gui_grounding.py validate \
  --root "${GUI_ROOT}" \
  --deep
```

### 2. Submit one node

If you normally request an interactive Pyxis shell, allocate it as usual:

```bash
srun \
  -A a0201 \
  -p debug \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=4 \
  --cpus-per-task=32 \
  --exclusive \
  --mem=450G \
  --time=01:30:00 \
  --environment=./lladao.toml \
  --pty bash --rcfile scripts/bootstrap_lladao_env.sh -i
```

Using the bootstrap as Bash's interactive rcfile makes the first allocation
create and populate the persistent virtual environment before displaying the
prompt. Later allocations reuse the version stamp and activate it immediately.

Inside the allocated shell, first run a two-step smoke test. Use a separate
results directory so its optimizer state is not resumed by the full run:

```bash
cd "${LLADAO_REPO_ROOT:-$PWD}"

RESULTS_DIR="${SCRATCH}/runs/lladao_gui_120k_smoke" \
TOTAL_STEPS=2 \
SAVE_EVERY=1 \
LOG_EVERY=1 \
WANDB_NAME=gui-grounding-smoke \
MAX_RESTARTS=0 \
EXPECTED_NUM_TOKENS=8192 \
MAX_NUM_TOKENS=12288 \
bash scripts/train_gui_grounding_120k.sh
```

`LLADAO_GUI_GROUNDING_DIR` is already supplied by the included `lladao.toml`.
The launcher reads `SLURM_GPUS_ON_NODE=4` and starts four local training
processes automatically. For the full run, use a new output directory and a
longer non-debug allocation:

```bash
TOTAL_STEPS=10001 \
SAVE_EVERY=500 \
EXPECTED_NUM_TOKENS=8192 \
MAX_NUM_TOKENS=12288 \
bash scripts/train_gui_grounding_120k.sh
```

From a second login shell, inspect the live Slurm and GPU memory usage with:

```bash
JOB_ID=2765257  # replace with the ID shown by squeue

sstat -j "${JOB_ID}" --allsteps \
  --format=JobID,AveRSS,MaxRSS,TresUsageInMax

srun --overlap --jobid="${JOB_ID}" --nodes=1 --ntasks=1 \
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv
```

After a failed or completed step, `sacct -j "${JOB_ID}"` reports its
state, exit code, requested memory, and peak resident memory:

```bash
sacct -j "${JOB_ID}" \
  --format=JobID,State,ExitCode,ReqMem,MaxRSS,MaxVMSize
```

The training log also reports rank 0 host RSS after model construction,
checkpoint loading, and each FSDP sharding stage.

`TOTAL_STEPS=10001` is a rough four-GPU starting point. Use the logged
`total_samples` to calculate the actual epoch length. On Clariden, keep
`--exclusive --mem=450G`; each rank still needs one full model plus the active
checkpoint shard before FSDP can distribute its parameters. A four-GPU GH200
smoke test reserved nearly all 96 GiB of HBM at the `8192/12288` token settings,
so increase these limits only after measuring memory on the target world size.

For an unattended one-node job on Clariden, run the submission wrapper from a
login node:

```bash
bash scripts/slurm/gui-120k-grounding-finetune.sh
```

It submits to account `a0201` and the `normal` partition for 12 hours, requests
one node with four GPUs and 450 GB of host memory, and prints the job ID and log
path. Settings can be overridden without editing the wrapper:

```bash
TOTAL_STEPS=20001 \
SAVE_EVERY=1000 \
WANDB_NAME=gui-grounding-long \
bash scripts/slurm/gui-120k-grounding-finetune.sh
```

The wrapper passes `lladao.toml` to the batch script's internal `srun`, which
sources the bootstrap inside the container before training. Model, data,
results, and Python environment paths come from the EDF. Set `DRY_RUN=1` to
print the resolved environment and `sbatch` command without submitting.

### 3. Submit multiple nodes

Command-line resource options override the defaults in the `.sbatch` file. For
two Clariden nodes with four GPUs each:

```bash
sbatch \
  --nodes=2 \
  --gres=gpu:4 \
  --export=ALL,GPUS_PER_NODE=4 \
  scripts/slurm/train_gui_grounding_120k.sbatch
```

This requests eight GPUs in total. On Clariden's `debug` partition, the
90-node-minute limit means a two-node job can request at most 45 minutes.

The first allocated host is selected as `MASTER_ADDR`; a job-specific port is
derived from `SLURM_JOB_ID`. One `atorch.distributed.run` launcher runs on each
node, and each launcher starts one process per local GPU.

### 4. Tune the training run

All important settings are environment overrides. For example:

```bash
sbatch \
  --export=ALL,TOTAL_STEPS=10001,SAVE_EVERY=1000,LEARNING_RATE=1e-5,WANDB_OFFLINE=True \
  scripts/slurm/train_gui_grounding_120k.sbatch
```

Useful overrides include:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `TOTAL_STEPS` | `10001` | Optimizer iterations; the final default save is step 10000 |
| `SAVE_EVERY` | `500` | Checkpoint interval |
| `LEARNING_RATE` | `2.5e-5` | Peak learning rate |
| `WARMUP_STEPS` | `300` | Warm-up iterations |
| `EXPECTED_NUM_TOKENS` | `32768` | Soft packed-token target per GPU rank |
| `MAX_NUM_TOKENS` | `36864` | Hard packed-token limit per GPU rank |
| `NUM_WORKERS` | `1` | DataLoader workers per GPU rank |
| `FREEZE_VIT` | `False` | Freeze the vision encoder when `True` |
| `CPU_OFFLOAD` | `False` | Offload FSDP parameters to CPU when `True` |
| `WANDB_OFFLINE` | `True` | Store W&B logs locally |

The paper specifies the 120K data mixture but does not publish a complete set
of optimizer, batch-size, or 120K-run epoch hyperparameters. Consequently, the
defaults above are reproducible engineering starting points based on this
repository's existing training settings, not a claim of exact paper
reproduction. The prepared open-data approximation contains 107,341 usable
rows rather than 120,000.

Because batching is token-based, steps do not map to epochs exactly. Watch the
logged global `total_samples`, average it over several steps, and estimate:

```text
steps_per_epoch = 107341 / average_global_total_samples_per_step
```

If CUDA runs out of memory, lower both `EXPECTED_NUM_TOKENS` and
`MAX_NUM_TOKENS`, for example to `16384` and `18432`. Keep
`MAX_NUM_TOKENS_PER_SAMPLE=16384` unless individual high-resolution examples
are being skipped. Reducing token packing changes the number of samples per
step, so adjust `TOTAL_STEPS` accordingly.

### 5. Monitor and resume

```bash
squeue -j JOB_ID
tail -f slurm-lladao-gui-120k-JOB_ID.out
```

Checkpoints are written below `RESULTS_DIR/checkpoints/0000500`,
`0001000`, and so on. Re-submit with the same `RESULTS_DIR`;
`--auto_resume True` selects the latest checkpoint automatically. Keep the same
world size (nodes times GPUs per node) when resuming because optimizer state is
saved in FSDP shards tied to that world size. To restart from the released model
instead, use a new or empty `RESULTS_DIR`.

Offline W&B logs can be uploaded later with:

```bash
wandb sync /path/to/runs/gui-120k/wandb/wandb/offline-run-*
```

For a command-only preflight without starting workers, run the non-Slurm
launcher on a compute node with `DRY_RUN=1`:

```bash
MODEL_PATH=/path/to/model \
LLADAO_GUI_GROUNDING_DIR=/path/to/data/parquet \
DRY_RUN=1 \
bash scripts/train_gui_grounding_120k.sh
```

WebLINX is CC BY-NC-SA 4.0 and contains third-party web content; its upstream
terms require research/fair-use compliance. The preparation code redacts email
addresses and explicit `password:` values from prompts and type actions.

| Source | Dataset-card license |
| --- | --- |
| Multimodal-Mind2Web | OpenRAIL |
| WebLINX | CC BY-NC-SA 4.0 plus upstream terms of use |
| OS-Atlas | Apache-2.0 |
| RICO Widget Caption repack | CC BY 4.0 |

The OS-Atlas web/desktop and RICO inputs use pinned public Parquet repacks to
avoid downloading the complete 777 GB OS-Atlas archive and to retain embedded
screenshots. Every row identifies both the repack revision and its upstream
source. Check the upstream terms before redistributing generated Parquet files.
