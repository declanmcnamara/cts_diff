import os
from functools import partial

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import lightning as L
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import torch
import tqdm
from hydra import compose, initialize
from omegaconf import DictConfig

from network import UNet
from samplers import ForwardSDE, ReverseSDE
from utils import (
    MNISTDataLoader,
    load_model,
    load_opt_state,
    save_model,
    save_opt_state,
)


@eqx.filter_jit
def score(xt, t, x0, forward_sde: ForwardSDE):
    wrapped_grad_fn = eqx.filter_grad(forward_sde.marginal_log_prob)
    return wrapped_grad_fn(xt, t, x0)


@eqx.filter_jit
def sample_time(key, t0: float, t1: float, n_sample: int):
    t = jr.uniform(key, (n_sample,), minval=t0, maxval=t1 / n_sample)
    t = t + (t1 / n_sample) * jnp.arange(n_sample)
    return t


@eqx.filter_jit
def my_single_loss_fn(score_model, t, x0, forward_sde: ForwardSDE, key):
    mu, scale, eps = forward_sde.forward_sample_rparam(t, x0, key)
    xt_draw = mu + scale * eps
    pred_score = score_model(t, xt_draw, key=key)
    actual_score = score(xt_draw, t, x0, forward_sde)
    weight = lambda t: 1 - jnp.exp(-forward_sde.beta_module.beta_int(t))
    return weight(t) * jnp.square(pred_score - actual_score).sum()


@eqx.filter_jit
def my_batched_loss_fn(score_model, batch_x0, forward_sde, key):
    time_key, sde_key = jr.split(key)
    batch_size = batch_x0.shape[0]
    t = sample_time(time_key, 0.0, 10.0, batch_size)
    part_x0_t = partial(
        my_single_loss_fn, score_model=score_model, forward_sde=forward_sde, key=sde_key
    )
    return jax.vmap(part_x0_t)(x0=batch_x0, t=t).mean()


@eqx.filter_jit
def make_step(score_model, batch_x0, forward_sde, optimizer, opt_state, key):
    time_key, loss_key = jr.split(key)
    score_model = eqx.nn.inference_mode(score_model, False)

    loss_value, grads = eqx.filter_value_and_grad(my_batched_loss_fn)(
        score_model, batch_x0, forward_sde, loss_key
    )
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(score_model, eqx.is_array)
    )
    score_model = eqx.apply_updates(score_model, updates)
    key = jr.split(time_key, 1)[0]  # new key
    return score_model, opt_state, loss_value, key


@hydra.main(version_base=None, config_path="conf", config_name="mnist")
def main(cfg: DictConfig):
    # with initialize(version_base=None, config_path="conf"):
    #     cfg = compose(config_name="mnist")
    seed = cfg.seed
    device = cfg.training.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)  # comment this out if on CPU

    key = jr.key(seed)
    model_key, loader_key, train_key, sample_key = jr.split(key, 4)
    train_loader = MNISTDataLoader(key=loader_key, batch_size=cfg.training.batch_size)

    score_model = UNet(
        data_shape=(1, 28, 28),
        is_biggan=cfg.model.is_biggan,
        dim_mults=cfg.model.dim_mults,
        hidden_size=cfg.model.hidden_size,
        heads=cfg.model.heads,
        dim_head=cfg.model.dim_head,
        dropout_rate=cfg.model.dropout_rate,
        num_res_blocks=cfg.model.num_res_blocks,
        attn_resolutions=cfg.model.attn_resolutions,
        final_activation=cfg.model.final_activation,
        q_dim=None,
        a_dim=None,
        key=model_key,
    )

    optimizer = hydra.utils.instantiate(cfg.optimizer)
    opt_state = optimizer.init(eqx.filter(score_model, eqx.is_array))

    beta_int_fcn = lambda t: t
    forward_sde = ForwardSDE(beta_int_fcn)

    counter = 0
    n_per_epoch = train_loader.n_batch
    epoch_losses = []
    epoch_idx = 0
    for batch_idx, (X_batch, y_batch) in enumerate(
        tqdm.tqdm(train_loader.as_generator(), total=cfg.training.n_steps)
    ):
        if batch_idx >= cfg.training.n_steps:
            break

        score_model, opt_state, loss, train_key = make_step(
            score_model, X_batch, forward_sde, optimizer, opt_state, train_key
        )
        epoch_losses.append(loss)
        counter += 1

        # Logging
        if counter % n_per_epoch == 0:
            # Logging
            avg_epoch_loss = jnp.array(epoch_losses).mean()
            print(f"Epoch {epoch_idx}: Avg. Loss {avg_epoch_loss}")
            epoch_losses = []

            # Log example image
            if epoch_idx % 10 == 0:
                score_model = eqx.nn.inference_mode(score_model, True)
                reverse_sde = ReverseSDE(1e-1, score_model, forward_sde)
                reverse_sde.plot_grid(5, 5, sample_key)

                save_model(score_model, "current_model.eqx")

                # Save optimiser state
                save_opt_state(
                    optimizer,
                    opt_state,
                    i=epoch_idx * n_per_epoch,
                    filename="current_opt_state",
                )

            epoch_idx += 1


if __name__ == "__main__":
    main()
