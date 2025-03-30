#!/usr/bin/env python3

# ================================================================================
# File       : train.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements LeNet training
# Date       : 2025-03-31
# ================================================================================

import jax
import optax
from flax import nnx
from jax import numpy as jnp
from jax import random
from lenet import LeNet

from datasets.loader import load_config, load_mnist
from trainer import BaseTrainer, parse_args


class LeNetTrainer(BaseTrainer):
    def loss_fn(self, params: nnx.State, batch: dict[str, jnp.ndarray]):
        # Save the current model parameters temporarily
        original_params = nnx.state(self.model)

        # Temporarily update the model parameters for forward pass
        nnx.update(self.model, params)
        logits = self.model.apply(batch["image"])

        # Restore the original parameters immediately after forward pass
        nnx.update(self.model, original_params)

        labels = batch["label"]
        onehot = jax.nn.one_hot(labels, num_classes=10)
        loss = optax.softmax_cross_entropy(logits, onehot).mean()
        return loss


def main():
    args = parse_args()
    config = load_config(args.config)
    print(config)

    rng = random.PRNGKey(0)
    input_shape = tuple(config["input_shape"])
    lr = float(config["learning_rate"])
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    # print(input_shape)
    # print(lr)
    # print(epochs)
    # print(batch_size)

    trainer = LeNetTrainer(LeNet, input_shape, rng, lr=lr)

    # batch = next(iter(load_mnist(batch_size=64)[0]))
    # print(batch["image"].shape)  # Must print (64, 28, 28, 1)

    if config["dataset"] == "mnist":
        train_ds, _ = load_mnist(batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    for epoch in range(epochs):
        epoch_loss = []
        train_ds, _ = load_mnist(batch_size=batch_size)  # reload each epoch
        for batch in train_ds:
            # print(f"image shape: {batch['image'].shape}, label shape: {batch['label'].shape}")
            loss = trainer.train_batch(batch)
            epoch_loss.append(loss)

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}: Avg Loss = {avg_loss:.4f}")


if __name__ == "__main__":
    main()
