# stGdiff: A Diffusion Model for Spatial Transcriptomics Imputation

**stGdiff** is a diffusion-model-based method for imputing missing gene expression in spatial transcriptomics (ST) data using matched single-cell RNA-seq (scRNA-seq) reference profiles. It builds on the stGDiff framework and introduces refined training and sampling strategies for improved spatial gene imputation and downstream spatial domain detection.

## Key Features

- **Diffusion-based imputation**: Uses a diffusion transformer (DiT) to iteratively denoise masked spatial gene expressions.
- **Cross-validated gene masking**: 5-fold K-fold splits across shared genes for robust evaluation.
- **Multi-scale evaluation**: Reports gene-level metrics (SPCC, SSIM, RMSE, JS divergence) and spatial clustering metrics (ARI, AMI, Homo, NMI).
- **Built-in baselines**: scVI, SpaGE, stPlus, Tangram, and SpatialScope are included under `baseline/`.

## Repository Structure

```
stGdiff/
├── model/                       # Diffusion model definition
│   ├── stGdiff_model.py        # DiT-stGdiff architecture
│   ├── stGdiff_scheduler.py     # Noise scheduler (cosine beta schedule)
│   ├── stGdiff_train.py         # Training loop
│   ├── stGdiff_match.py         # Spatial keypoint matching encoder
│   └── sample.py                # Reverse diffusion sampling
├── process/                     # Data loading & preprocessing
│   ├── data.py
│   ├── data_process.py
│   └── result_analysis.py        # Clustering metrics
├── preprocess/                  # Additional preprocessing utilities
│   ├── data.py
│   └── utils.py
├── baseline/                    # Baseline methods (scVI, SpaGE, stPlus, Tangram, SpatialScope)
├── datasets/                    # Placeholder for input datasets
├── tool/                        # Helper utilities
│   ├── file_tool.py
│   ├── metrics_index.py
│   └── utils.py
├── test-stGdiff.py              # Main entry script
├── main.py                      # Core training & evaluation pipeline
├── test-stDiff.py               # Standalone stGdiff test script
├── test-baseline.py             # Baseline evaluation script
├── test-spscope.py              # SpatialScope wrapper
├── environment.yml              # Conda environment
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/your-username/stGdiff.git
cd stGdiff

conda env create -f environment.yml
conda activate stGdiff
pip install -r requirements.txt
```

## Requirements

- Python 3.8.19
- PyTorch 1.9.0+ (CUDA supported)
- scanpy 1.9.3
- scikit-learn 1.3.2
- numpy 1.24.4, pandas 1.5.3, scipy
- timm, einops, tqdm, ray

## Datasets

The benchmark datasets are derived from the spatial transcriptomics integration benchmark:

> **Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type annotation.** *Nature Methods* (2022).

Place your scRNA-seq and spatial transcriptomics `.h5ad` files under `datasets/`. Each dataset should contain:
- `sc_data`: single-cell h5ad (shared genes with ST)
- `sp_data`: spatial transcriptomics h5ad with `x_coord` and `y_coord` in `.obs`

## Usage

### Running stGdiff

```bash
python test-stGdiff.py \
    --sc_data your_sc_dataset.h5ad \
    --sp_data your_sp_dataset.h5ad \
    --document your_dataset_name \
    --batch_size 2048 \
    --hidden_size 512 \
    --head 16 \
    --step 1500 \
    --epoch 900 \
    --rand 0
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--sc_data` | str | `dataset8_seq_84.h5ad` | Single-cell h5ad filename |
| `--sp_data` | str | `dataset8_spatial_84.h5ad` | Spatial h5ad filename |
| `--document` | str | `dataset8` | Name used for output folders |
| `--batch_size` | int | `2048` | Training batch size |
| `--hidden_size` | int | `512` | Transformer hidden dimension |
| `--noise_std` | float | `10` | Data augmentation noise std |
| `--head` | int | `16` | Number of attention heads |
| `--step` | int | `1500` | Diffusion timesteps |
| `--epoch` | int | `900` | Training epochs per fold |
| `--rand` | int | `0` | Random state for K-fold split |

### Recommended hyperparameters by gene count

| Shared gene count | Batch size | Hidden size |
|---|---|---|
| < 512 | 2048 | 512 |
| 512 – 1024 | 512 | 1024 |
| 1024 – 2048 | 512 | 2048 |

### Running baselines

```bash
python test-baseline.py --sc_data your_sc.h5ad --sp_data your_sp.h5ad --document baseline_result
python test-spscope.py --sc_data your_sc.h5ad --sp_data your_sp.h5ad --document spscope_result
```

## Output

Results are written under `data/<document>/<timestamp>/`:

- `stGdiff_impute<N>.csv` — imputed gene expression matrix
- `stGdiff_gene_Metrics<N>.txt` — per-gene SPCC, SSIM, RMSE, JS
- `stGdiff_cluster_Metrics<N>.txt` — ARI, AMI, Homo, NMI
- `result-<timestamp>.csv` — aggregated run log across seeds

## Citation

If you use stGdiff in your research, please cite:

```bibtex
@inproceedings{stgdiff2026,
  title={stGdiff: A Diffusion Model for Spatial Transcriptomics Imputation},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2026}
}
```

## Acknowledgments

stGdiff builds upon [stGDiff](https://github.com/fdu-wangfeilab/stGDiff). Baseline implementations are adapted from [SpatialBenchmarking](https://github.com/QuKunLab/SpatialBenchmarking), scVI, SpaGE, stPlus, Tangram, and SpatialScope.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
