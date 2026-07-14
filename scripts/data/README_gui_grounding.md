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
official `weblinx==0.3.2` parser. In the repository's `llada-o` Conda environment:

```bash
conda run -n llada-o pip install "datasets>=3.6,<4" "weblinx==0.3.2"

conda run -n llada-o python scripts/data/prepare_gui_grounding.py download \
  --root /home/ubuntu/datasets/lladao_gui_120k

conda run -n llada-o python scripts/data/prepare_gui_grounding.py build \
  --root /home/ubuntu/datasets/lladao_gui_120k

conda run -n llada-o python scripts/data/prepare_gui_grounding.py validate \
  --root /home/ubuntu/datasets/lladao_gui_120k \
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
