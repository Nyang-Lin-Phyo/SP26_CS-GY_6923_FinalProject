# Scaling Laws for Transformer Language Models on SVG Code
**CS-GY 6923 — Optional Project | NYU Tandon School of Engineering | Spring 2026**

## Overview

This project investigates scaling laws for decoder-only transformer language models trained on SVG (Scalable Vector Graphics) code, a structured non-linguistic domain. We train five model sizes under standard parameterization (SP) and five under µP (Maximal Update Parameterization), comparing scaling behavior and learning rate transfer across model widths.

## Repository Structure

```
├── MLprojPart1Final.ipynb       # Data download, cleaning, tokenization
├── MLprojPart2Final.ipynb       # Transformer scaling study (standard parameterization)
├── MLprojPart3Final.ipynb       # µP scaling study and extrapolation
├── MLprojPart4Final.ipynb       # Best model training, sample generation, evaluation
└── README.md
```

## How to Run

All notebooks are designed to run on **Google Colab** with a GPU runtime (A100 recommended). Run them in order.

### Part 1 — Preprocessing
`MLprojPart1Final.ipynb`
- Downloads SVG datasets from HuggingFace (`starvector/svg-icons-simple`, `starvector/svg-emoji-simple`, `starvector/svg-stack-simple`)
- Cleans and filters SVGs
- Trains a Byte-Level BPE tokenizer (vocab size 4,000)
- Saves tokenized train/val/test splits to Google Drive

### Part 2 — Scaling Study (Standard Parameterization)
`MLprojPart2Final.ipynb`
- Loads tokenized data from Google Drive
- Performs learning rate sweep on the Tiny model
- Trains five model sizes (Tiny through XL) for one epoch
- Fits a power law scaling curve
- Saves results and model checkpoints to Google Drive

### Part 3 — µP Scaling Study
`MLprojPart3Final.ipynb`
- Loads tokenized data from Google Drive
- Performs learning rate sweep on the Tiny µP model
- Transfers optimal LR to all larger µP models
- Trains five µP model sizes for one epoch
- Fits power law and extrapolates to 10× XL
- Saves results to Google Drive

### Part 4 — Best Model Training and Generation
`MLprojPart4Final.ipynb`
- Trains the XL µP model for 3 epochs
- Generates unconditional samples at temperatures {0.5, 0.8, 1.0}
- Generates prefix-conditioned completions
- Evaluates XML validity, SVG structural validity, and render rate
- Saves generated SVGs and evaluation metrics to Google Drive

## Dependencies

All dependencies are installed at the top of each notebook via `pip install`. Key libraries used:

- `torch` — model training
- `datasets` — HuggingFace dataset loading
- `tokenizers` — Byte-Level BPE tokenizer
- `mup` — Maximal Update Parameterization
- `lxml` — XML validation
- `cairosvg` — SVG rendering
- `scipy` — power law curve fitting
- `matplotlib` — plotting
- `Pillow` — image handling

## Model Configurations

| Model | Params (SP) | Params (µP) | d_model | n_layers (SP vs µP) | n_heads |
|-------|-------------|-------------|---------|------------------|---------|
| Tiny | 1,854,112 | 2,250,656 | 128 | 4 / 6 | 4 |
| Small | 4,258,720 | 4,258,720 | 192 | 6 / 6 | 6 |
| Medium | 13,821,856 | 13,821,856 | 384 | 6 / 6 | 6 |
| Large | 35,755,936 | 23,146,400 | 512 | 10 / 6 | 8 |
| XL | 91,400,608 | 48,873,376 | 768 | 12 / 6 | 12 |

## Key Results

- Best learning rate (both SP and µP): **5×10⁻³**
- SP scaling exponent: degenerate (α ≈ 900,260) due to LR instability at large model sizes
- µP scaling exponent: **α = 1.73**, loss floor **c = 0.489**
- Best model (XL µP, 3 epochs): test perplexity **1.57**, unconditional XML validity **100%**, render rate **100%**

## References

1. Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361
2. Hoffmann et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556
3. Yang et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. arXiv:2203.09789
4. Rodriguez et al. (2023). StarVector: Generating Scalable Vector Graphics Code from Images and Text. arXiv:2312.11556
5. Karpathy, A. (2022). nanoGPT. https://github.com/karpathy/nanoGPT
