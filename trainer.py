#!/usr/bin/env python3

# ================================================================================
# File       : trainer.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a training base class and utilities for
#              training
# Date       : 2025-03-31
# ================================================================================

import argparse
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model from config.yaml")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
        required=False,
    )
    return parser.parse_args()


@partial(jax.jit, static_argnames=("loss_fn", "optimizer"))
def train_step(
    params: nnx.State,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    loss_fn: Callable[[nnx.State, Dict[str, jnp.ndarray]], jnp.ndarray],
    optimizer: optax.GradientTransformation,
) -> Tuple[nnx.State, optax.OptState, jnp.ndarray]:  # return loss as array
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss  # no float() here


class BaseTrainer:
    def __init__(
        self,
        model_cls: Callable[..., nnx.Module],
        input_shape: tuple,
        rng: jax.Array,
        lr: float = 1e-3,
    ):
        self.model = model_cls(rng, input_shape)
        self.params = nnx.state(self.model)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

    def loss_fn(self, params: nnx.State, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        raise NotImplementedError

    def train_batch(self, batch: Dict[str, jnp.ndarray]) -> float:
        self.params, self.opt_state, loss = train_step(
            params=self.params,
            opt_state=self.opt_state,
            batch=batch,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
        )
        nnx.update(self.model, self.params)
        return float(loss)  # Convert here, safely outside jit
