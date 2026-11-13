# Hateful Meme Classification: Frozen Embeddings and Multimodal FOP

Course project for *Social Media Mining and Analysis* (KV 344053, JKU).

- **Assignment 1:** unimodal baselines (linear probes on frozen image or text embeddings).
- **Assignment 2:** multimodal classifier adapting Fusion and Orthogonal Projection (FOP, Saeed et al., ICASSP 2022): gated fusion of image and text embeddings, trained with cross-entropy plus an orthogonal projection (OP) loss. The pretrained encoders stay frozen and embeddings are cached.

The final model combines EmbeddingGemma 2 (image) and all-mpnet-base-v2 (text), the best encoders from Assignment 1.

## Setup

- Python 3.14, managed with [uv](https://docs.astral.sh/uv/). Requirements are in `pyproject.toml`; exact versions are in `uv.lock`.
- Install with `uv sync`.
- Put the data in `data/`: `train.jsonl`, `dev.jsonl`, `test.jsonl` and the `img/` folder (8,500 train, 500 dev and 1,000 test memes).

## Embeddings

Embeddings are extracted once and cached in `embeddings_cache/` (gitignored, like `*.pt`). The first training run does this automatically, but it is slow without a GPU. To extract them up front (for example on a GPU machine):

```
uv run extract_embeddings.py configs/multimodal_gated_lambda0.json
```

## Training (Assignment 2)

Each command trains seeds 0, 1 and 2 and writes to the config's `OUTPUT_DIR` under `outputs/`:

```
uv run main.py configs/multimodal_gemma.json            # gated fusion + OP (lambda = 0.1), final model
uv run main.py configs/multimodal_gated_lambda0.json    # ablation: gated fusion, lambda = 0
uv run main.py configs/multimodal_linear_lambda0.json   # ablation: linear fusion, lambda = 0
```

Each output folder contains `config.json`, `summary.json` (per-seed and mean/std validation AUROC, trainable parameters), `training_history.csv`, `val_scores.csv` and `best_model.pt`. The last three come from seed 0 only. The reported std is the population standard deviation.

## Hyperparameter search

```
uv run hparam_search.py configs/multimodal_gemma.json
```

Two stages, both on seed 0 and scored by validation AUROC: (1) a grid search at lambda = 0; (2) a sweep over lambda with the stage-1 winner fixed. Results are written to `outputs/multimodal_embeddinggemma_mpnet_search/` (`grid_search.csv`, `lambda_sweep.csv`, `recommended_hparams.json`). The selected values (learning rate 0.001, weight decay 0, batch size 64, common dimension 1024, dropout 0.5) are used by all three configs above, which differ only in `FUSION` and `LAMBDA_OP`.

## Inference

```
uv run predict.py configs/multimodal_gemma.json --split test
```

Loads `best_model.pt` of that run and writes `predictions_test.csv` to its output folder, with columns `meme_id` and `hateful_probability` (one row per meme, ids preserved). Use `--split dev` to score the validation set.

## Analysis scripts

```
uv run make_results_table.py      # LaTeX results table (results_table.tex) from outputs/*/summary.json
uv run error_analysis.py --k 4    # candidate validation examples for the error analysis
```

## Assignment 1

The unimodal baselines (CLIP ViT-B/32 and EmbeddingGemma 2 for images; all-mpnet-base-v2 and EmbeddingGemma 2 for text) were produced with the code at commit `<COMMIT>` (see the git history). Their results are in `outputs/img_*`, `outputs/image_*` and `outputs/text_*`, and are reused as the baselines in Assignment 2.

## Code overview

| File | Purpose |
|---|---|
| `main.py` | Train and evaluate all seeds of one config |
| `hparam_search.py` | Two-stage hyperparameter search |
| `predict.py` | Write test or validation predictions |
| `pipeline.py` | Full multimodal model |
| `projections.py` | Embedding branches (Linear, BatchNorm, ReLU, Dropout, L2 norm) |
| `fusion.py` | Gated and linear fusion |
| `losses.py` | Orthogonal projection loss |
| `dataloaders.py` | Datasets and class-balanced batch sampler |
| `trainer.py` | Training loop with early stopping on validation AUROC |
| `encoders.py` | Frozen image and text encoders (embedding extraction) |
| `configs/` | One JSON file per experiment |

`retrieval_model.py` and `fop_main.py` are the unmodified official FOP code, kept for reference only and not used by the pipeline.

## References

1. Saeed et al., *Fusion and Orthogonal Projection for Improved Face-Voice Association*, ICASSP 2022.
2. Official FOP code: `retrieval_model.py`, `main.py` (included here as `fop_main.py`).