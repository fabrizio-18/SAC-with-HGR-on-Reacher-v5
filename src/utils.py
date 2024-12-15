import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
from torch import distributions as pyd
import math
import torch.nn.functional as F


def plot(vec, plot_type):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    
    clear_output(True)
    plt.figure(figsize=(20, 5))
    
    # Set the title and plot based on the plot type
    plt.subplot(131)
    if plot_type == 'reward':
        plt.title(f'Epoch {len(vec)}. Reward: {vec[-1]:.2f}')
    elif plot_type == 'loss':
        plt.title(f'Epoch {len(vec)}. Loss: {vec[-1]:.2f}')
    
    plt.plot(vec)
    plt.xlabel('Epochs')
    plt.ylabel(plot_type.capitalize())
    plt.grid(True)
    plt.show()


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def _eq_(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use ⁠ cache_size=1 ⁠ instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))
    

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


