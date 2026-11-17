# STaR-GNN

**STaR-GNN** is a spatiotemporal graph recurrent model for multi–district metered area (DMA) water-demand forecasting. This repository contains a **model-only research implementation** of the proposed approach, with its essential diffusion-recurrent backbone and basic tests.

## Method

- **SAS-Norm (Seasonally Anchored State Normalization)** separates daily demand level and variability from intraday patterns and estimates future daily states using historical observations.
- **FA-DPR (Forecast-Aligned Daily Pattern Retrieval)** retrieves graph-encoded historical daily representations, conditioned on the decoder state and available future calendar features.
- **Graph-based encoder–decoder:** diffusion graph recurrent units capture functional inter-DMA dependence.

The original implementation retains internal names such as `STaRDCRNN` and `dssn_sasr` for compatibility; these refer to the STaR-GNN model and its SAS-Norm component.

## Repository contents

- `src/dma_wdf/models/star_components.py` — SAS-Norm and FA-DPR modules.
- `src/dma_wdf/models/star_dcrnn.py` — integrated STaR-GNN architecture.
- `src/dma_wdf/models/dcrnn.py` — diffusion-recurrent encoder–decoder **required as the model backbone**, not a separately released baseline experiment.
- `src/dma_wdf/data/graph.py` — training-only Pearson functional graph construction and graph loading.
- `examples/minimal_forward.py` — minimal forward-pass example with synthetic inputs.
- `tests/` — unit tests for the proposed modules and model integration.

This release intentionally omits **baseline experiments, training-result tables, plots, raw BWDF data, checkpoints, and the full paper-specific data/training/evaluation pipeline**. It is intended to make the algorithm source inspectable, **not to claim that published numerical results can be reproduced from this subset alone**.

## Installation

Python 3.11 or later is recommended.

```bash
python -m pip install -e .
python examples/minimal_forward.py
```

To run the included unit tests:

```bash
python -m pip install pytest
python -m pytest -q tests/test_star_components.py tests/test_star_dcrnn.py
```

## Model interface

The implementation expects hourly node features with the demand feature in **channel 0**:

- Historical input `x_past`: `[batch, 672, num_nodes, input_dim]` (28 days).
- Known future calendar features `x_future_exog`: `[batch, horizon, num_nodes, future_exog_dim]`, where `horizon` is 24 or 168 hours.
- Graph `random_walk`: `[num_nodes, num_nodes]`, a row-normalized training-derived positive-Pearson functional graph.
- Output: `[batch, horizon, num_nodes]` predictions.

Do not use future observed demand as an input at inference time. The example uses **randomly initialized weights and a synthetic identity graph**, solely to demonstrate model execution. It makes **no meaningful forecasting claim**.

## Data and reproducibility scope

The associated research uses the public *Battle of Water Demand Forecasting* (BWDF) dataset. Refer to the original competition publications and data providers for the dataset and its terms. Raw observations are **not redistributed** here.

Reproducing the numerical results reported in the research paper additionally requires the exact data splits, training configuration, preprocessing, trained weights or rerun training, and evaluation protocol; these are **outside this minimal algorithm release**.

## Attribution and license

The model builds on the diffusion-convolutional recurrent neural network (DCRNN) formulation (Li et al., 2018) and introduces the STaR-GNN components described above. The included diffusion backbone is required by the integrated model.

Distributed under the MIT License; see [LICENSE](LICENSE).
