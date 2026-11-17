# STaR-GNN

This repository provides the **source code of STaR-GNN**, accompanying the manuscript *“Seasonality-aware hierarchical spatiotemporal modeling for multi-horizon water demand forecasting across district metered areas”*. It is made available specifically for the manuscript's **Software and data availability** section in *Environmental Modelling & Software*.

## Source code

- `src/dma_wdf/models/star_gnn.py` — STaR-GNN model.
- `src/dma_wdf/models/star_components.py` — SAS-Norm and FA-DPR components.
- `src/dma_wdf/models/graph_recurrent.py` — required graph recurrent components.
- `src/dma_wdf/data/graph.py` — functional graph utilities.
- `examples/minimal_forward.py` — basic model execution example.
- `tests/` — model tests.

## Requirements and usage

Python 3.11 or later; PyTorch, NumPy, and Pandas.

```bash
python -m pip install -e .
python examples/minimal_forward.py
```

The example uses synthetic inputs and randomly initialized parameters to demonstrate that the model can run.

## Data availability

The study uses the publicly available **Battle of Water Demand Forecasting (BWDF)** dataset, released as supplementary material to [Alvisi et al. (2025)](https://doi.org/10.1061/JWRMD5.WRENG-6887). The [WaterFutures/wf4bwdf](https://github.com/WaterFutures/wf4bwdf) package provides access to and utilities for the dataset.

The dataset and trained model weights are not included in this repository. This is a **source-code release**, not a complete reproduction package for the manuscript's numerical results.

## License

MIT License. See [LICENSE](LICENSE).
