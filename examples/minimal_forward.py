"""Minimal STaR-GNN forward-pass demonstration using synthetic inputs.

The graph and inputs below are for a shape/smoke test only.
They are NOT real BWDF observations or a trained forecasting model.
"""

import torch

from dma_wdf.models.star_dcrnn import STaRDCRNN


def main() -> None:
    torch.manual_seed(42)
    num_nodes = 2
    horizon = 24
    model = STaRDCRNN(
        random_walk=torch.eye(num_nodes),
        input_dim=3,
        hidden_dim=4,
        horizon=horizon,
        history_hours=672,
        num_nodes=num_nodes,
        num_rnn_layers=1,
        max_diffusion_step=2,
        future_exog_dim=1,
        use_dssn_sasr=True,
        use_fa_dpr=True,
        attention_dim=3,
        condition_on_future_calendar=True,
    )
    x_past = torch.randn(1, 672, num_nodes, 3)
    x_future_calendar = torch.randn(1, horizon, num_nodes, 1)
    model.eval()
    with torch.no_grad():
        prediction = model(x_past, x_future_exog=x_future_calendar)
    assert prediction.shape == (1, horizon, num_nodes)
    assert torch.isfinite(prediction).all()
    print("STaR-GNN forward pass OK:", tuple(prediction.shape))


if __name__ == "__main__":
    main()
