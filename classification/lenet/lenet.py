#!/usr/bin/env python3

# ================================================================================
# File       : lenet.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a lenet model
# Date       : 2025-03-31
# ================================================================================

import jax.numpy as jnp
from flax import nnx


class LeNet(nnx.Module):
    def __init__(self, rng, input_shape):
        rngs = nnx.Rngs(params=rng)
        dummy_x = jnp.zeros(input_shape)

        self.conv1 = nnx.Conv(dummy_x.shape[-1], 6, (5, 5), rngs=rngs)
        self.conv2 = nnx.Conv(6, 16, (5, 5), rngs=rngs)

        x = nnx.relu(self.conv1(dummy_x))
        x = nnx.avg_pool(x, (2, 2), (2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.avg_pool(x, (2, 2), (2, 2))

        flat_dim = x.shape[1] * x.shape[2] * x.shape[3]  # Should explicitly be 256

        self.fc1 = nnx.Linear(flat_dim, 120, rngs=rngs)
        self.fc2 = nnx.Linear(120, 84, rngs=rngs)
        self.fc3 = nnx.Linear(84, 10, rngs=rngs)

    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.conv1(x))
        x = nnx.avg_pool(x, (2, 2), (2, 2))

        x = nnx.relu(self.conv2(x))
        x = nnx.avg_pool(x, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))  # batch size retained here

        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        x = self.fc3(x)
        return x
