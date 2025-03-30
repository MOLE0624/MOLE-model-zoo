#!/usr/bin/env python3

# ================================================================================
# File       : loader.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements dataset loader
# Date       : 2025-03-31
# ================================================================================

import jax.numpy as jnp
import tensorflow_datasets as tfds
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_mnist(batch_size=64):
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()

    def preprocess(ds):
        def _map(x, y):
            x = jnp.expand_dims(jnp.array(x, dtype=jnp.float32) / 255.0, axis=-1)
            return {"image": x, "label": jnp.array(y)}

        return [{"image": x, "label": y} for x, y in tfds.as_numpy(ds)]

    ds_train = preprocess(
        ds_builder.as_dataset(split="train", batch_size=batch_size, as_supervised=True)
    )
    ds_test = preprocess(
        ds_builder.as_dataset(split="test", batch_size=batch_size, as_supervised=True)
    )

    return ds_train, ds_test
