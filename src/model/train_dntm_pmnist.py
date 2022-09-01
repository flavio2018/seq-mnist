"""This script trains a DNTM on the PMNIST task."""
import logging
import os

import omegaconf
import torch
import wandb
from torchmetrics.classification import Accuracy
from torchvision.utils import make_grid

import hydra
from data.perm_seq_mnist import get_dataloaders
from model.builders import build_model
from model.dntm.MemoryReadingsStats import MemoryReadingsStats
from utils.pytorchtools import EarlyStopping
from utils.run_utils import configure_reproducibility
from utils.train_utils import get_optimizer
from utils.wandb_utils import log_config, log_preds_and_targets, log_weights_gradient


@hydra.main(config_path="../../conf/local", config_name="train_smnist")
def click_wrapper(cfg):
    train_and_test_dntm_smnist(cfg)


def train_and_test_dntm_smnist(cfg):
    device = torch.device(cfg.run.device, 0)
    rng = configure_reproducibility(cfg.run.seed)

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
    train_dataloader, valid_dataloader = get_dataloaders(cfg, rng)
    model = build_model(cfg, device)
    memory_reading_stats = MemoryReadingsStats(path=os.getcwd())

    loss_fn = torch.nn.NLLLoss()
    opt = get_optimizer(model, cfg)
    early_stopping = EarlyStopping(
        verbose=True,
        path=os.path.join(os.getcwd(), f"{cfg.run.codename}.pth"),
        trace_func=logging.info,
        patience=cfg.train.patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.run.use_amp)

    # training
    torch.autograd.set_detect_anomaly(cfg.run.detect_anomaly)
    for epoch in range(cfg.train.epochs):
        logging.info(f"Epoch {epoch}")

        train_loss, train_accuracy = training_step(
            device, model, loss_fn, opt, train_dataloader, epoch, cfg, scaler
        )
        torch.cuda.empty_cache()
        valid_loss, valid_accuracy = valid_step(
            device, model, loss_fn, valid_dataloader, epoch, memory_reading_stats
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
def valid_step(device, model, loss_fn, valid_data_loader, epoch, memory_reading_stats):
    logging.info("Starting validation step")
    valid_accuracy = Accuracy().to(device)
    valid_epoch_loss = 0
    all_labels = torch.tensor([])
    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(valid_data_loader):
        logging.info(f"Batch {batch_i}")
        logging.debug(f"Memory allocated: {str(torch.cuda.memory_allocated(device))} B")
        logging.debug(f"Memory reserved: {str(torch.cuda.memory_allocated(device))} B")
        all_labels = torch.cat([all_labels, targets])
        model.prepare_for_batch(mnist_images, device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        _, output = model(mnist_images)
        memory_reading_stats.update_memory_readings(model.memory_reading, epoch=epoch)

        loss_value = loss_fn(output.T, targets)
        valid_epoch_loss += loss_value.item() * mnist_images.size(0)

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
        batch_size = len(mnist_images)  # mnist_images.shape is (BS, 784)
        logging.info(f"Batch {batch_i}")
        model.zero_grad()

        if (epoch == 0) and (batch_i == 0):
            mnist_batch_img = wandb.Image(
                make_grid(mnist_images.reshape(batch_size, 1, 28, -1))
            )
            wandb.log(
                {f"Training data batch {batch_i}, epoch {epoch}": mnist_batch_img}
            )

        model.prepare_for_batch(mnist_images, device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=cfg.run.use_amp):
            logging.info("Start processing batch")
            _, output = model(mnist_images)
            logging.info("End processing batch")
            loss_value = loss_fn(output.T, targets)

        log_preds_and_targets(batch_i, output, targets)

        epoch_loss += loss_value.item() * mnist_images.size(0)

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

    accuracy_over_batches = train_accuracy.compute()
    epoch_loss /= len(train_data_loader.sampler)
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches


if __name__ == "__main__":
    click_wrapper()
