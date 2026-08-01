# VLM-Based Receipt Field Extraction (QLoRA)

Fine-tuned **Qwen2-VL-2B-Instruct** with **QLoRA** to extract structured JSON
(line items and total) from receipt images. Trained on a single Kaggle T4 GPU.

Adapter: [huggingface.co/your-username/qwen2vl-receipt-lora](REPLACE)

## Result (60 held-out receipts, unseen during training)

| Metric | Score |
|---|---|
| Valid JSON output | 100% |
| Total exact-match | 90% |
| Mean char-similarity on near-misses | 0.957 |

Misses are minor: occasional dropped zero-price line items, and currency-symbol
spacing (`Rp29,090` vs `Rp 29,090`). See `samples/` for side-by-side predictions.

## Method

- **Base model:** `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`, loaded in 4-bit.
- **Adaptation:** LoRA on ~0.8% of parameters (language + attention + MLP layers);
  vision encoder frozen. 2 epochs, effective batch size 8, learning rate 2e-4.
- **Data:** [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2),
  400 train / 60 validation. Ground truth flattened to `{items:[{name,price}], total}`.
- **Eval:** JSON validity, exact-match on total, and character-level edit ratio
  on a held-out split the model never saw during training.

## Retrieval experiment (negative result)

Tested a FAISS retrieval layer that injects field-formatting rules into the prompt
at inference. It did not improve exact-match (0.900 base vs 0.883 grounded) on the
already-fine-tuned model, indicating formatting was internalized during training.
Removed from the final pipeline.

## Repo structure

- `train_extract.ipynb` — data prep, QLoRA fine-tune, evaluation
- `eval_metrics.json` — held-out metrics
- `samples/` — qualitative prediction examples (input receipt + prediction vs gold)

## Limitations

- Trained on CORD (café/restaurant receipts); other document types need retraining.
- Field extraction only; **not** a forgery- or tamper-detection system.
- Small held-out set (60); metrics are indicative, not production-grade.

## Reproduce

Open the notebook on Kaggle with a T4 GPU and internet enabled, run all cells.
