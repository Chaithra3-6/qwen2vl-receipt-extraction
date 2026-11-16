# VLM-Based Receipt Field Extraction (QLoRA)

Fine-tuned **Qwen2-VL-2B-Instruct** with **QLoRA** to extract structured JSON
(line items and total) from receipt images. Trained on a single Kaggle T4 GPU.

Adapter: [huggingface.co/chai1me/qwen2vl-receipt-lora](https://huggingface.co/chai1me/qwen2vl-receipt-lora)

## Result (60 held-out receipts)

| Metric | Score |
|---|---|
| Valid JSON output | 100% |
| Total exact-match | 90% |
| Mean character similarity of the total field (all 60) | 0.957 |

How to read these numbers:

- The 60 receipts are the first 60 of CORD-v2's validation split. They were not
  used for training, and training ran a fixed 2 epochs with no evaluation,
  checkpoint selection or early stopping.
- Only the **total** field is scored. Item names and prices are not scored;
  see `samples/` for a qualitative look at them.
- The similarity score is 1 minus normalized Levenshtein distance between the
  predicted and gold total, averaged over all 60 receipts, including the 54
  exact matches. The 6 misses average about 0.57.
- CORD-v2's 100-receipt test split has not been used.

## Method

- **Base model:** `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`, loaded in 4-bit.
- **Adaptation:** LoRA on the language model's attention and MLP layers; vision
  layers frozen. Rank 16, alpha 16, dropout 0, no bias.
  18,464,768 trainable parameters of 2,227,450,368 (0.83%).
- **Training:** 2 epochs (100 steps), batch size 2 with 4 gradient-accumulation
  steps (effective 8), learning rate 2e-4 with linear decay and 5 warm-up steps,
  8-bit AdamW, weight decay 0.01, seed 3407.
- **Data:** [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2),
  first 400 usable training receipts. Images are downscaled so the longer side
  is at most 1024 px. Ground truth is flattened to
  `{"items": [{"name", "price"}], "total"}`.
- **Prompt:** `Extract the receipt into JSON with keys: items (a list of name and price) and total. Output only JSON.`
- **Decoding:** greedy, up to 256 new tokens.

## Sample predictions

Six held-out receipts with model output beside the input image (`samples/`).

Totals were extracted correctly on all six shown. Item extraction is imperfect:
the model sometimes drops zero-price lines (free add-ons), changes currency
spacing (`Rp29,090` vs `Rp 29,090`), and can mis-split payment rows on
cluttered receipts.

## Retrieval experiment (negative result)

Tested a FAISS retrieval layer that adds field-formatting rules to the prompt at
inference. On the same 60 receipts it did not improve total exact-match
(0.900 without, 0.883 with), so it is not part of the final pipeline.

## Repo structure

- `train_extract.ipynb`: data prep, QLoRA fine-tune, evaluation
- `eval_metrics.json`: metrics on the 60 held-out receipts
- `samples/`: prediction examples (input receipt, prediction, gold total)

## Limitations

- Trained on CORD (cafe and restaurant receipts, mostly Indonesian); other
  document types need retraining.
- Small evaluation set (60), one training run, and only the total field is scored.
- Field extraction only; **not** a forgery or tamper detector.

## Reproduce

Open the notebook on Kaggle with a T4 GPU and internet enabled, and run all cells.
