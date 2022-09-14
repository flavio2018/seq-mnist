"""This script trains a DNTM on the PMNIST task."""
import logging
import os
import time

import omegaconf
import torch
import wandb
from torchmetrics.classification import Accuracy

import hydra
from data.perm_seq_mnist import get_dataset
from model.builders import build_model
from utils.pytorchtools import EarlyStopping
from utils.run_utils import configure_reproducibility
from utils.train_utils import get_optimizer
from utils.wandb_utils import log_config, log_weights_gradient


@hydra.main(config_path="../../conf/local", config_name="train_smnist")
def click_wrapper(cfg):
    train_and_test_dntm_smnist(cfg)


def train_and_test_dntm_smnist(cfg):
    device = torch.device(cfg.run.device, 0)
    configure_reproducibility(cfg.run.seed)

    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project="dntm_mnist",
        entity="flapetr",
        mode=cfg.run.wandb_mode,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb.run.name = cfg.run.codename
    wandb.save(os.path.join(os.getcwd(), f"{cfg.run.codename}.log"))
    log_config(cfg_dict)
    train_ds, test_ds = get_dataset(cfg)
    train_tds = torch.utils.data.TensorDataset(
        train_ds.data[:900].view(900, -1, 1).type(torch.float32).to(cfg.run.device),
        train_ds.targets[:900],
    )
    test_tds = torch.utils.data.TensorDataset(
        test_ds.data[:100].view(100, -1, 1).type(torch.float32).to(cfg.run.device),
        train_ds.targets[:100],
    )
    model = build_model(cfg, device)

    loss_fn = torch.nn.NLLLoss()
    opt = get_optimizer(model, cfg)
    early_stopping = EarlyStopping(
        verbose=True,
        path=os.path.join(os.getcwd(), f"{cfg.run.codename}.pth"),
        trace_func=logging.info,
        patience=cfg.train.patience,
    )

    for epoch in range(cfg.train.epochs):
        logging.info(f"Epoch {epoch}")

        start_time = time.time()
        train_loss, train_accuracy = training_step(
            device, model, loss_fn, opt, train_tds, epoch, cfg
        )
        epoch_duration = time.time() - start_time
        training_throughput = 784 * 900 / epoch_duration
        wandb.log({"TT": training_throughput})
        valid_loss, valid_accuracy = valid_step(
            device, model, loss_fn, test_tds, epoch, cfg
        )

        wandb.log({"loss_training_set": train_loss, "loss_validation_set": valid_loss})
        print(
            f"Epoch {epoch} --- train loss: {train_loss} - valid loss: {valid_loss} -",
            f"train acc: {train_accuracy} - valid acc: {valid_accuracy}",
        )
        wandb.log(
            {"acc_training_set": train_accuracy, "acc_validation_set": valid_accuracy}
        )
        log_weights_gradient(model)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break


@torch.no_grad()
def valid_step(device, model, loss_fn, valid_ds, epoch, cfg):
    logging.info("Starting validation step")
    valid_accuracy = Accuracy().to(device)
    valid_epoch_loss = 0
    model.eval()
    for batch_i in range(len(valid_ds) // cfg.train.batch_size):
        mnist_images, targets = valid_ds[
            batch_i * cfg.train.batch_size : (batch_i + 1) * cfg.train.batch_size
        ]
        logging.debug(f"Memory allocated: {str(torch.cuda.memory_allocated(device))} B")
        logging.debug(f"Memory reserved: {str(torch.cuda.memory_allocated(device))} B")

        mnist_images, targets = mnist_images.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        model.prepare_for_batch(mnist_images, device)

        _, output = model(mnist_images)

        loss_value = loss_fn(output.T, targets)
        valid_epoch_loss += loss_value.item() * mnist_images.size(0)

        valid_accuracy(output.T, targets)
    valid_accuracy_at_epoch = valid_accuracy.compute()
    valid_epoch_loss /= len(valid_ds)
    valid_accuracy.reset()
    return valid_epoch_loss, valid_accuracy_at_epoch


def training_step(device, model, loss_fn, opt, train_ds, epoch, cfg):
    logging.info("Starting training step")
    train_accuracy = Accuracy().to(device)

    epoch_loss = 0
    model.train()
    for batch_i in range(len(train_ds) // cfg.train.batch_size):
        mnist_images, targets = train_ds[
            batch_i * cfg.train.batch_size : (batch_i + 1) * cfg.train.batch_size
        ]
        # mnist_images.shape is (BS, 784)
        model.zero_grad()

        mnist_images, targets = (
            mnist_images.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
        )

        model.prepare_for_batch(mnist_images, device)

        _, output = model(mnist_images)
        loss_value = loss_fn(output.T, targets)

        epoch_loss += loss_value.item() * mnist_images.size(0)

        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.train.max_grad_norm,
            norm_type=2.0,
            error_if_nonfinite=cfg.train.error_clip_grad_if_nonfinite,
        )
        opt.step()

        train_accuracy(output.T, targets)

    accuracy_over_batches = train_accuracy.compute()
    epoch_loss /= len(train_ds)
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches


if __name__ == "__main__":
    click_wrapper()
