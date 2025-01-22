import abc
import os
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import cloudpickle
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from torch import utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Credit:

Array = np.ndarray


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, key, batch_size):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.as_generator()` to iterate over the data loader.")

    @abc.abstractmethod
    def as_generator(self):
        pass


class MNISTDataLoader(_AbstractDataLoader):
    def __init__(self, key, batch_size):
        self.key = key
        self.batch_size = batch_size
        # Download MNIST data
        dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
        images = dataset.data / 255.0 * 2 - 1
        classes = dataset.targets
        self.X = einops.rearrange(jnp.array(images), "b h w -> b 1 h w")
        self.y = einops.rearrange(jnp.array(classes), "b -> b 1")
        self.n_batch = self.X.shape[0] // batch_size

    def as_generator(self):
        dataset_size = self.X.shape[0]
        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = self.batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]

                yield (self.X[batch_perm], self.y[batch_perm])
                start = end
                end = start + self.batch_size


def load_model(model, filename):
    model = eqx.tree_deserialise_leaves(filename, model)
    return model


def save_model(model, filename):
    eqx.tree_serialise_leaves(filename, model)


def save_opt_state(opt, opt_state, i, filename="state.obj"):
    """Save an optimiser and its state for a model, to train later"""
    state = {"opt": opt, "opt_state": opt_state, "step": i}
    f = open(filename, "wb")
    cloudpickle.dump(state, f)
    f.close()


def load_opt_state(filename="state.obj"):
    f = open(filename, "rb")
    state = cloudpickle.load(f)
    f.close()
    return state
