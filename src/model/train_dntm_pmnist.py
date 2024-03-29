"""This script trains a DNTM on the PMNIST task."""
import logging
import os
import time

import omegaconf
import torch
import wandb
from torchmetrics.classification import Accuracy

import hydra
from data.perm_seq_mnist import get_dataloaders
from model.builders import build_model
from model.dntm.MemoryReadingsStats import MemoryReadingsStats
from utils.pytorchtools import EarlyStopping
from utils.run_utils import configure_reproducibility
from utils.train_utils import get_optimizer
from utils.wandb_utils import log_config, log_weights_gradient, log_mem_stats, log_params_norm, log_intermediate_values_norm


@hydra.main(config_path="../../conf/local", config_name="train_smnist")
def click_wrapper(cfg):
    train_and_test_dntm_smnist(cfg)


def train_and_test_dntm_smnist(cfg):
    device = torch.device(cfg.run.device, 0)
    rng = configure_reproducibility(cfg)

    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    print(omegaconf.OmegaConf.to_yaml(cfg))
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
    log_config(cfg_dict)
    train_dataloader, valid_dataloader = get_dataloaders(cfg, rng)
    model = build_model(cfg, device)
    os.mkdir(path=os.getcwd() + '/output')
    memory_reading_stats = MemoryReadingsStats(path=os.getcwd() + '/output')

    loss_fn = torch.nn.NLLLoss()
    opt = get_optimizer(model, cfg)
    early_stopping = EarlyStopping(
        verbose=True,
        path=os.path.join(os.getcwd(), 'output', f"{cfg.run.codename}.pth"),
        trace_func=logging.info,
        patience=cfg.train.patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.run.use_amp)

    # training
    torch.autograd.set_detect_anomaly(cfg.run.detect_anomaly)
    for epoch in range(cfg.train.epochs):
        logging.info(f"Epoch {epoch}")

        start_time = time.time()
        train_loss, train_accuracy = training_step(
            device, model, loss_fn, opt, train_dataloader, epoch, cfg, scaler
        )
        epoch_duration = time.time() - start_time
        training_throughput = 784 * len(train_dataloader.sampler) / epoch_duration
        wandb.log({"TT": training_throughput})

        valid_loss, valid_accuracy = valid_step(
            device, model, loss_fn, valid_dataloader, epoch, memory_reading_stats
        )

        print(
            f"Epoch {epoch} --- train loss: {train_loss} - valid loss: {valid_loss} -",
            f"train acc: {train_accuracy} - valid acc: {valid_accuracy}",
        )
        wandb.log(
            {"acc_training_set": train_accuracy, "acc_validation_set": valid_accuracy}
        )

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break


@torch.no_grad()
def valid_step(device, model, loss_fn, valid_data_loader, epoch, memory_reading_stats):
    logging.info("Starting validation step")
    valid_accuracy = Accuracy().to(device)
    valid_epoch_loss = 0
    all_labels = torch.tensor([])
    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(valid_data_loader):
        logging.debug(f"Memory allocated: {str(torch.cuda.memory_allocated(device))} B")
        logging.debug(f"Memory reserved: {str(torch.cuda.memory_allocated(device))} B")
        all_labels = torch.cat([all_labels, targets])

        mnist_images, targets = (
            mnist_images.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
        )
        model.prepare_for_batch(mnist_images, device)

        _, output = model(mnist_images)
        memory_reading_stats.update_memory_readings(model.memory_reading, epoch=epoch)

        loss_value = loss_fn(output.T, targets)
        valid_epoch_loss += loss_value.item() * mnist_images.size(0)
        wandb.log({"loss_validation_set": loss_value})

        valid_accuracy(output.T, targets)
    torch.save(all_labels, memory_reading_stats.path + "labels" + f"_epoch{epoch}.pt")
    valid_accuracy_at_epoch = valid_accuracy.compute()
    valid_epoch_loss /= len(valid_data_loader.sampler)
    valid_accuracy.reset()
    return valid_epoch_loss, valid_accuracy_at_epoch


def training_step(device, model, loss_fn, opt, train_data_loader, epoch, cfg, scaler):
    logging.info("Starting training step")
    train_accuracy = Accuracy().to(device)

    epoch_loss = 0
    model.train()
    for batch_i, (mnist_images, targets) in enumerate(train_data_loader):
        # mnist_images.shape is (BS, 784)
        step = compute_step(epoch, batch_i, mnist_images.size(0))
        model.zero_grad()

        mnist_images, targets = (
            mnist_images.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
        )

        model.prepare_for_batch(mnist_images, device)

        with torch.cuda.amp.autocast(enabled=cfg.run.use_amp):
            _, output = model(mnist_images)
            loss_value = loss_fn(output.T, targets)

        epoch_loss += loss_value.item() * mnist_images.size(0)
        wandb.log({
            "loss_training_set": loss_value,
            "step": step,
            })

        scaler.scale(loss_value).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.train.max_grad_norm,
            norm_type=2.0,
            error_if_nonfinite=cfg.train.error_clip_grad_if_nonfinite,
        )
        scaler.step(opt)
        scaler.update()

        train_accuracy(output.T, targets)
        log_weights_gradient(model, step)
        log_mem_stats(model, step)
        log_params_norm(model, step)
        log_intermediate_values_norm(model, step)


    accuracy_over_batches = train_accuracy.compute()
    epoch_loss /= len(train_data_loader.sampler)
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches


def compute_step(epoch, batch_i, batch_size):
    num_steps_per_epoch = 60000 * 0.9 * batch_size
    return epoch*num_steps_per_epoch + batch_i + 1


if __name__ == "__main__":
    click_wrapper()
