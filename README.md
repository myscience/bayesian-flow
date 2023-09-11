# Bayesian Flow Networks in Easy PyTorch

This repo contains the _unofficial_ implementation for `Bayesian Flow Networks` as introduced in [Graves et al. (2023)](https://arxiv.org/abs/2308.07037).

## Usage

```python
import torch

from src.unet import UNet
from src.bfn import BayesianFlowNetwork

bfn = BayesianFlowNetwork(
    backbone=UNet(
        net_dim=4,
        ctrl_dim=None,
        use_cond=False,
        use_attn=True,
        num_group=4,
        adapter='b c h w -> b (h w) c',
    ),
    loss_kind='continuous',
    data_kind='continuous',
    data_shape=(32, 32),
)

# Get some fake imgs for testing
imgs = torch.randn(16, 3, 32, 32)

# Compute the Bayesian Flow loss
loss = bfn.compute_loss(imgs)

# Compute the model gradients
loss.backward()

...

# Once the model is trained, we can sample from the learnt
# inverse flow by simply doing
samples = bfn(
  num_samples=4,
  num_steps=100,
  sigma_1=1e-3,
)
```

## Citations

```bibtex
@article{graves2023bayesian,
  title={Bayesian Flow Networks},
  author={Graves, Alex and Srivastava, Rupesh Kumar and Atkinson, Timothy and Gomez, Faustino},
  journal={arXiv preprint arXiv:2308.07037},
  year={2023}
}
```