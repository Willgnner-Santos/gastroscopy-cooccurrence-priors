# Clinical Co-occurrence Priors for Multi-label Gastroscopy Classification

Code and results for a study on **relational multi-label learning** applied to upper gastrointestinal (UGI) endoscopy images.  
The core hypothesis is that encoding observed co-occurrences among endoscopic findings — as graph priors or regularization terms — improves classification of images that simultaneously exhibit multiple pathological findings.

---

## Overview

| Item | Detail |
|---|---|
| Task | Multi-label classification of UGI endoscopy images |
| Labels (core) | Enanthema, Polyp, Ulcer, Erosion, Micronodularity |
| Dataset size | 2,046 images after quality filtering |
| Multi-label prevalence | ~52 % of images carry ≥ 2 simultaneous findings |
| Cross-validation | 5-fold iterative stratification (70 / 15 / 15) |
| Backbone | ResNet-50 (primary); EfficientNet-B3, ConvNeXt-Tiny, Swin-T (ablation) |

---

## Repository Structure

```
.
├── Notebooks/                     # Numbered notebooks — full experimental pipeline
│   ├── 01_visual_similarity_groups.ipynb   # Group-aware split to prevent data leakage
│   ├── 02_EDA_dataset.ipynb                # Exploratory analysis; label statistics
│   ├── 03_splits_baseline_colab.ipynb      # K-fold splits + M0 baseline (BCE & ASL)
│   ├── 04_graph_construction.ipynb         # Co-occurrence graph construction
│   ├── 05_relational_models.ipynb          # M1 (Classifier Chains), M2 (reg.), M3 (GCN)
│   ├── 06_clinical_validation.ipynb        # Expert endoscopist validation protocol
│   ├── 07_reviewer_rebuttal.ipynb          # Rebuttal experiments (λ ablation, M3 variants)
│   └── 08_error_analysis_V3.ipynb          # Qualitative error analysis (M0 vs M2)
│
├── graphs/                        # Precomputed adjacency matrices (CSV + NPY)
│   ├── emp_tau{00,01,03}.*        # Empirical co-occurrence graphs
│   ├── pmi_tau{00,01,03}.*        # PMI-normalised graphs
│   ├── clin_tau{00,01,03}.*       # Clinically curated graphs
│   ├── fold{0-4}/                 # Per-fold versions of each graph type
│   └── label_order.json           # Canonical label ordering
│
├── splits/                        # 5-fold CSV splits
│   ├── fold_{0-4}_{train,val,test}.csv
│   └── image_group_mapping.csv    # Visual-similarity group assignments
│
├── results/                       # Saved experiment outputs (JSON / CSV)
│   ├── 01_cooccurrence_extended.csv   # Co-occurrence, PMI, nPMI, φ statistics
│   ├── 02_eda_stats.json              # Dataset statistics
│   ├── 03_baseline_results.json       # M0 (BCE / ASL) 5-fold results
│   ├── 04_graph_stats.json            # Graph construction diagnostics
│   ├── 05_relational_results.json     # M1, M2, M3 aggregated metrics
│   └── 07_rebuttal_results.json       # Rebuttal ablation metrics
│
├── figs/                          # Publication-quality figures (EN and PT variants)
│
├── qualitative_analysis.py        # Standalone script — M0 vs M2 error categorisation
└── requirements.txt               # Python dependencies
```

> **Not included** (large or sensitive): `models/` (trained checkpoints, ~8.5 GB),  
> `clinical_validation/` (expert validation materials), raw image data.

---

## Models

| ID | Description |
|---|---|
| **M0** | ResNet-50 baseline — binary cross-entropy with label-frequency positive weighting |
| **M1** | Classifier Chains — sequential binary classifiers that condition on prior predictions |
| **M2** | M0 + co-occurrence regularisation — auxiliary loss `λ · L_coo` (λ ∈ {0.05 … 1.0}) |
| **M3** | ML-GCN — label embeddings propagated over a GCN with graph-structured adjacency |

Three graph types are studied for M3:

| Graph | Construction |
|---|---|
| G_emp | Raw co-occurrence counts, symmetric |
| G_pmi | PMI-normalised: log P(i,j) / P(i)P(j), clipped to [0, 1] |
| G_clin | Manually curated edge weights ∈ {0, 0.5, 1.0} |

Sparsification threshold τ ∈ {0, 0.1, 0.3, 0.5} is evaluated per graph type.

---

## Key Results (5-fold, ResNet-50)

| Model | F1-macro | F1-macro (multi-label images) | Hamming loss |
|---|---|---|---|
| M0 (BCE) | 0.517 ± 0.045 | 0.442 ± 0.102 | 0.063 ± 0.005 |
| M1 (CC) | 0.519 ± 0.028 | 0.367 ± 0.057 | 0.055 ± 0.003 |
| M2 (λ=0.3) | **0.537 ± —** | — | — |

Full per-label and per-fold metrics are in `results/05_relational_results.json`.

---

## Reproducing the Experiments

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: Python 3.10+, PyTorch ≥ 2.1, torchvision, scikit-learn, scikit-multilearn, pandas, numpy, matplotlib, seaborn.

### Data placement

The notebooks expect the image dataset and label CSV at:

```
/workspace/Dev/Data/Imgs/          # image files
/workspace/Dev/Data/Imgs-anotadas/dataset_labels.csv
```

Adjust the `ROOT` variable at the top of each notebook to match your local path.

### Execution order

Run notebooks **01 → 08** in sequence. Each notebook saves its outputs to `results/`, `graphs/`, `splits/`, or `figs/`, which the next notebook reads.  
Notebooks 03 and 05 support GPU execution; see the `RUNNING_IN_COLAB` flag to switch between local and cloud environments.

---

## Ablations

| Code | Variable |
|---|---|
| A1 | Graph type (emp / pmi / clin) |
| A2 | Include artifact nodes (Saliva, Light reflection) |
| A3 | Regularisation weight λ for M2 |
| A4 | Chain order for M1 (frequency / clinical / random) |
| A5 | Decision threshold (0.5 fixed vs. per-label optimised) |
| A6 | Loss function (BCE vs. Asymmetric Loss) |
| A7 | Sparsification threshold τ |

---

## Hypotheses Under Test

| # | Statement | Criterion |
|---|---|---|
| H1 | Relational models outperform M0 on multi-label images | ΔF1-macro ≥ +3 pp, p < 0.05 (Wilcoxon + Holm-Bonferroni) |
| H2 | No degradation on single-label images | ΔF1-macro ≥ −0.5 pp |
| H3 | Clinical graph outperforms data-driven graph on rare co-occurring pairs | F1 on pairs with < 20 training instances |
| H4 | Improved set-level metrics | ΔHamming ≤ −0.01, ΔSubset-accuracy ≥ +2 pp |

---

## License

Code: [MIT](LICENSE).  
Dataset: not distributed in this repository; subject to the original data use agreement.
