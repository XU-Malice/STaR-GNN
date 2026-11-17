# STaR-GNN

**STaR-GNN** is a spatiotemporal graph recurrent model for short-term water-demand forecasting across district metered areas (DMAs). This small repository provides the **algorithm implementation only**: the proposed normalization and retrieval modules, the integrated graph recurrent network, a minimal forward-pass example, and model tests.

## Method

- **SAS-Norm (Seasonally Anchored State Normalization):** separates day-level demand states from intraday variation and estimates future daily states from historical data.
- **FA-DPR (Forecast-Aligned Daily Pattern Retrieval):** retrieves historical graph-encoded daily representations aligned to each prediction step, conditioned on decoder state and available future calendar features.
- **Graph recurrent backbone:** models functional inter-DMA dependence with diffusion-based recurrent graph propagation.

## Code

| File | Purpose |
| --- | --- |
| `src/dma_wdf/models/star_components.py` | SAS-Norm and FA-DPR |
| `src/dma_wdf/models/star_gnn.py` | Integrated **STaRGNN** model |
| `src/dma_wdf/models/graph_recurrent.py` | Required graph recurrent backbone |
| `src/dma_wdf/data/graph.py` | Training-only Pearson functional graph utilities |
| `examples/minimal_forward.py` | Synthetic-input model execution |
| `tests/` | Model/module tests |

This repository does not include separate baseline experiments, manuscript figures, reported result tables, raw observations, checkpoints, or the full training and evaluation pipeline.

## Installation and basic check

Use Python 3.11 or newer.

```bash
python -m pip install -e .
python examples/minimal_forward.py
```

To run the unit tests:

```bash
python -m pip install pytest
python -m pytest -q tests/test_star_components.py tests/test_star_gnn.py
```

## Model interface

- Past observations: `x_past` has shape `[batch, 672, N, C]`, with demand in feature channel 0.
- Graph: `random_walk` is an `[N, N]` row-normalized functional graph derived **only from training data**.
- Known future covariates: `x_future_exog` has shape `[batch, horizon, N, future_exog_dim]`; these are known calendar features, **not future observed weather or demand**.
- Prediction output: `[batch, horizon, N]` for horizons of 24 or 168 hours.

The example uses a **synthetic identity graph and randomly initialized model weights** to validate software execution. It is not a trained water-demand forecasting model.

## Software and data availability

- **Software:** STaR-GNN (Python/PyTorch); the algorithm source and basic usage instructions are provided in this repository.
- **Original dataset:** *Battle of Water Demand Forecasting* (BWDF), published as supplementary material to Alvisi et al. (2025), *Journal of Water Resources Planning and Management*, DOI: [10.1061/JWRMD5.WRENG-6887](https://doi.org/10.1061/JWRMD5.WRENG-6887).
- **Original dataset download:** [Alvisi et al. supplementary material](https://ascelibrary.org/doi/suppl/10.1061/JWRMD5.WRENG-6887/suppl_file/supplemental_materials_jwrmd5.wreng-6887_alvisi.zip).
- **Organizer-approved data loading/evaluation package:** [WaterFutures/wf4bwdf](https://github.com/WaterFutures/wf4bwdf) ([archived release DOI](https://doi.org/10.5281/zenodo.17185809)). This package reads the competition data from the original supplementary materials and provides an official-competition-style evaluation interface.

To load the original competition data:

```bash
python -m pip install wf4bwdf
```

```python
import wf4bwdf as bwdf
dataset = bwdf.load_complete_dataset()
dma_inflows = dataset["dma-inflows"]
```

BWDF raw data are **not** redistributed in this repository. The original paper and data maintainers should be cited when using the dataset. The model-only release does not, by itself, reproduce paper metrics; that additionally requires the exact training data splits, preprocessing, experiment setup, training procedure, and checkpoint selection.

## Prior work and license

The graph recurrent backbone employs the established diffusion-convolutional recurrent formulation introduced by Li et al. (2018), *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*. STaR-GNN contributes SAS-Norm and FA-DPR on top of graph recurrent propagation. Renaming internal software classes does **not** change the model mathematics or its scientific provenance.

MIT License. See [LICENSE](LICENSE).
