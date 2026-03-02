# Patronising and Condescending Language Detection

**NLP Coursework 2026 — Imperial College London**
Lichang Wan (`jw1123@ic.ac.uk`) | Leaderboard: `JohnnyWan`

Binary classification of patronising and condescending language (PCL) — SemEval 2022 Task 4 (Subtask 1).

---

## Repository Structure

```text
.
├── BestModel/
│   ├── train.ipynb                     # Training notebook (run this to reproduce)
│   ├── best_model.pt                   # Saved model weights (Git LFS)
│   ├── dontpatronizeme_pcl.tsv         # Main dataset
│   ├── dontpatronizeme_categories.tsv  # Multi-label category annotations
│   ├── train_semeval_parids-labels.csv # Official train split IDs
│   ├── dev_semeval_parids-labels.csv   # Official dev split IDs
│   ├── task4_test.tsv                  # Official test set (no labels)
│   ├── training_curves.png             # Loss and F1 training curves
│   ├── threshold_optimisation.png      # Threshold vs F1/P/R plot
│   └── requirements.txt
├── EDA/
│   ├── eda.py
│   ├── eda_technique1_class_distribution.png
│   ├── eda_technique2_lexical_analysis.png
│   └── eda_technique2_keyword_distribution.png
├── dev.txt                             # Dev set predictions (one per line)
├── test.txt                            # Test set predictions (one per line)
└── report.tex                          # Full coursework report
```

---

## Approach

Fine-tuned **RoBERTa-large** with four targeted improvements over the RoBERTa-base baseline (dev F1 = 0.48):

| Component | Purpose |
| --- | --- |
| **Focal loss** (α=0.83, γ=2) | Handles 1:9 class imbalance |
| **Word-swap augmentation** | Doubles PCL training samples (794 → 1,588) without altering vocabulary |
| **Layer-wise LR decay** (λ=0.9) | Preserves pretrained lower-layer representations |
| **Threshold optimisation** | Grid search over t ∈ [0.3, 0.7] on dev set |

**Result: dev F1 = 0.6041** at t* = 0.40 (Precision = 0.6103, Recall = 0.5980).

---

## Reproducing Results

### 1. Install dependencies

```bash
pip install -r BestModel/requirements.txt
```

Requires Python 3.9+ and PyTorch with CUDA recommended.

### 2. Run the training notebook

Open and run `BestModel/train.ipynb` top-to-bottom. The notebook will:

- Load and binarise the dataset (label ≥ 2 → PCL)
- Augment the PCL training set with random word swaps
- Fine-tune RoBERTa-large with focal loss and LLRD
- Optimise the classification threshold on the dev set
- Save `best_model.pt` and write `dev.txt` / `test.txt` to the repo root

All random seeds are fixed (`seed=42`). Expected training time: ~2 hours on a single GPU.

### 3. Key hyperparameters

| Parameter | Value |
| --- | --- |
| Base model | `roberta-large` |
| Learning rate (classifier head) | 1×10⁻⁵ |
| LLRD decay λ | 0.9 |
| Focal loss α / γ | 0.83 / 2.0 |
| Effective batch size | 32 (8 × 4 gradient accumulation) |
| Max sequence length | 128 tokens |
| Early stopping patience | 5 epochs |
| Optimal threshold t* | 0.40 |

---

## Prediction Files

| File | Lines | Format |
| --- | --- | --- |
| `dev.txt` | 2,094 | One prediction per line (`0` = No PCL, `1` = PCL) |
| `test.txt` | 3,832 | One prediction per line (`0` = No PCL, `1` = PCL) |
