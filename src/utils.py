import numpy as np
import gymnasium as gym
import math
from torch import distributions as pyd
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import operator

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action
    


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

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # This forula is numerically stable, see details in the following link
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
    


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
    


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
    



def save_plots(rewards, critic_losses, success_rates, episodes, rewards_eval, success_rates_eval, output_dir='plots', smooth_window=300, save_data=True, eval_interval=30):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if save_data:
        # Save rewards, success rates, and critic losses to numpy files
        np.save(os.path.join(output_dir, "rewards.npy"), rewards)
        np.save(os.path.join(output_dir, "critic_losses.npy"), critic_losses)
        np.save(os.path.join(output_dir, "success_rates.npy"), success_rates)
        np.save(os.path.join(output_dir, "episodes.npy"), episodes)
        np.save(os.path.join(output_dir, "rewards_eval.npy"), rewards_eval)
        np.save(os.path.join(output_dir, "success_rates_eval.npy"), success_rates_eval)

        # Optionally, save as CSV for easier viewing
        np.savetxt(os.path.join(output_dir, "rewards.csv"), rewards, delimiter=',')
        np.savetxt(os.path.join(output_dir, "critic_losses.csv"), critic_losses, delimiter=',')
        np.savetxt(os.path.join(output_dir, "success_rates.csv"), success_rates, delimiter=',')
        np.savetxt(os.path.join(output_dir, "episodes.csv"), episodes, delimiter=',')
        np.savetxt(os.path.join(output_dir, "rewards_eval.csv"), rewards_eval, delimiter=',')
        np.savetxt(os.path.join(output_dir, "success_rates_eval.csv"), success_rates_eval, delimiter=',')

    def smooth(data, window):
        """Calculate moving average and standard deviation."""
        smoothed = np.array([np.mean(data[max(0, i - window + 1):i + 1]) for i in range(len(data))])
        std = np.array([np.std(data[max(0, i - window + 1):i + 1]) for i in range(len(data))])
        return smoothed, std
    
    def smooth_eval(data, eval_window=10):
        """Calculate moving average and standard deviation for evaluation metrics."""
        smoothed = np.array([np.mean(data[max(0, i - eval_window + 1):i + 1]) for i in range(len(data))])
        std = np.array([np.std(data[max(0, i - eval_window + 1):i + 1]) for i in range(len(data))])
        return smoothed, std

    # Smooth rewards
    smoothed_rewards, reward_std = smooth(rewards, smooth_window)

    # Plot episode rewards
    plt.figure()
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.2, color='blue')
    plt.plot(episodes, smoothed_rewards, label="Moving Average Reward", color='blue')
    plt.fill_between(episodes, smoothed_rewards - reward_std, smoothed_rewards + reward_std, color='blue', alpha=0.1)
    plt.xlabel("Training Epochs")
    plt.ylabel("Reward")
    plt.title("Episode Rewards During Training")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "episode_rewards.png"))
    plt.close()

    # Smooth critic loss
    smoothed_losses, loss_std = smooth(critic_losses, smooth_window)

    # Plot critic loss
    plt.figure()
    plt.plot(episodes, critic_losses, label="Critic Loss", alpha=0.2, color='orange')
    plt.plot(episodes, smoothed_losses, label="Moving Average Loss", color='orange')
    plt.fill_between(episodes, smoothed_losses - loss_std, smoothed_losses + loss_std, color='orange', alpha=0.1)
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.title("Critic Loss During Training")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "critic_loss.png"))
    plt.close()

    # Smooth success rates
    smoothed_success, success_std = smooth(success_rates, smooth_window)

    # Plot success rate
    plt.figure()
    plt.plot(episodes, success_rates, label="Success Rate", alpha=0.2, color='green')
    plt.plot(episodes, smoothed_success, label="Moving Average Success Rate", color='green')
    plt.fill_between(episodes, smoothed_success - success_std, smoothed_success + success_std, color='green', alpha=0.1)
    plt.xlabel("Training Epochs")
    plt.ylabel("Success Rate")
    plt.title("Success Rate During Training")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "success_rate.png"))
    plt.close()

    eval_episodes = np.arange(eval_interval, len(rewards_eval) * eval_interval + 1, eval_interval)

    smoothed_rewards_eval, reward_eval_std = smooth_eval(rewards_eval, eval_window=3)
    smoothed_success_eval, success_eval_std = smooth_eval(success_rates_eval, eval_window=3)
    # Plot evaluation rewards
    plt.figure()
    plt.plot(eval_episodes, rewards_eval, label="Evaluation Reward", alpha=0.2, color='purple')
    plt.plot(eval_episodes, smoothed_rewards_eval, label="Moving Average Evaluation Reward", color='purple')
    plt.fill_between(eval_episodes, smoothed_rewards_eval - reward_eval_std, smoothed_rewards_eval + reward_eval_std, color='purple', alpha=0.1)
    plt.xlabel("Training Epochs")
    plt.ylabel("Reward")
    plt.title("Evaluation Rewards")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "evaluation_rewards.png"))
    plt.close()

    # Plot evaluation success rates
    plt.figure()
    plt.plot(eval_episodes, success_rates_eval, label="Evaluation Reward", alpha=0.2, color='red')
    plt.plot(eval_episodes, smoothed_success_eval, label="Moving Average Evaluation Success Rate", color='red')
    plt.fill_between(eval_episodes, smoothed_success_eval - success_eval_std, smoothed_success_eval + success_eval_std, color='red', alpha=0.1)
    plt.xlabel("Training Epochs")
    plt.ylabel("Success Rate")
    plt.title("Evaluation Success Rate")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "evaluation_success_rates.png"))
    plt.close()

    print(f"Plots saved to {output_dir}")


def plot_from_npy(output_dir='simple_buffer_plots', smooth_window=300):
    # Load data from .npy files
    rewards = np.load(os.path.join(output_dir, "rewards.npy"))
    critic_losses = np.load(os.path.join(output_dir, "critic_losses.npy"))
    success_rates = np.load(os.path.join(output_dir, "success_rates.npy"))
    episodes = np.load(os.path.join(output_dir, "episodes.npy"))

    def smooth(data, window):
        """Calculate moving average and standard deviation."""
        smoothed = np.array([np.mean(data[max(0, i - window + 1):i + 1]) for i in range(len(data))])
        std = np.array([np.std(data[max(0, i - window + 1):i + 1]) for i in range(len(data))])
        return smoothed, std

    # Smooth data
    smoothed_rewards, reward_std = smooth(rewards, smooth_window)
    smoothed_losses, loss_std = smooth(critic_losses, smooth_window)
    smoothed_success, success_std = smooth(success_rates, smooth_window)

    # Plot rewards
    plt.figure()
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.2, color='blue')
    plt.plot(episodes, smoothed_rewards, label="Moving Average Reward", color='blue')
    plt.fill_between(episodes, smoothed_rewards - reward_std, smoothed_rewards + reward_std, color='blue', alpha=0.1)
    plt.xlabel("Training Epochs")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "episode_rewards.png"))
    plt.close()

    # Plot critic losses
    plt.figure()
    plt.plot(episodes, critic_losses, label="Critic Loss", alpha=0.2, color='orange')
    plt.plot(episodes, smoothed_losses, label="Moving Average Loss", color='orange')
    plt.fill_between(episodes, smoothed_losses - loss_std, smoothed_losses + loss_std, color='orange', alpha=0.1)
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.title("Critic Loss During Training")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "critic_loss.png"))
    plt.close()

    # Plot success rates
    plt.figure()
    plt.plot(episodes, success_rates, label="Success Rate", alpha=0.2, color='green')
    plt.plot(episodes, smoothed_success, label="Moving Average Success Rate", color='green')
    plt.fill_between(episodes, smoothed_success - success_std, smoothed_success + success_std, color='green', alpha=0.1)
    plt.xlabel("Training Epochs")
    plt.ylabel("Success Rate")
    plt.title("Success Rate During Training")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "success_rate.png"))
    plt.close()




