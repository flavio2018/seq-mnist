"""This script trains a DNTM on the PMNIST task."""
import torch.nn
import logging
import os
import hydra
import omegaconf
import wandb

from data.perm_seq_mnist import get_dataloaders
from model.builders import build_model
from utils.run_utils import configure_reproducibility
from utils.train_utils import get_optimizer
from utils.wandb_utils import log_weights_gradient, log_preds_and_targets
from utils.pytorchtools import EarlyStopping

from torchmetrics.classification import Accuracy
from torchvision.utils import make_grid


@hydra.main(config_path="../../conf/local", config_name="train_smnist")
def click_wrapper(cfg):
    train_and_test_dntm_smnist(cfg)


def train_and_test_dntm_smnist(cfg):
    device = torch.device("cuda", 0)
    rng = configure_reproducibility(cfg.run.seed)

    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="dntm_mnist", entity="flapetr", mode=cfg.run.wandb_mode)
    wandb.run.name = cfg.run.codename
    for subconfig_name, subconfig_values in cfg_dict.items():
        if isinstance(subconfig_values, dict):
            wandb.config.update(subconfig_values)
        else:
            logging.warning(f"{subconfig_name} is not being logged.")

    train_dataloader, valid_dataloader = get_dataloaders(cfg, rng)
    model = build_model(cfg.model, device)
    memory_reading_stats = MemoryReadingsStats(path=os.getcwd())
    memory_reading_stats.init_random_matrix(model.memory.overall_memory_size)

    loss_fn = torch.nn.NLLLoss()
    opt = get_optimizer(model, cfg)
    early_stopping = EarlyStopping(verbose=True,
                                   path=os.path.join(os.getcwd(),
                                                     f"{cfg.run.codename}.pth"),
                                   trace_func=logging.info,
                                   patience=cfg.train.patience)

    # training
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(cfg.train.epochs):
        logging.info(f"Epoch {epoch}")

        train_loss, train_accuracy = training_step(device, model, loss_fn, opt, train_dataloader, epoch, cfg)
        valid_loss, valid_accuracy = valid_step(device, model, loss_fn, valid_dataloader, epoch, memory_reading_stats)
        memory_readings_stats.load_memory_readings(epoch)
        memory_reading_stats.compute_stats()

        wandb.log({'loss_training_set': train_loss,
                   'loss_validation_set': valid_loss})
        print(f"Epoch {epoch} --- train loss: {train_loss} - valid loss: {valid_loss} - train acc: {train_accuracy} - valid acc: {valid_accuracy}")
        wandb.log({'acc_training_set': train_accuracy,
                   'acc_validation_set': valid_accuracy})
        log_weights_gradient(model)

        wandb.log({'memory_reading_variance': memory_reading_stats.readings_variance,
                   'memory_reading_kl_div': memory_reading_stats.kl_divergence})
        memory_reading_stats.plot_random_projections()
        wandb.log({f"memory_readings_random_projections": wandb.Image(
            memory_reading_stats.path+"memory_readings_projections_epoch{0:03}.png".format(epoch))})
        memory_readings_stats.reset()

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break


def valid_step(device, model, loss_fn, valid_data_loader, epoch, memory_reading_stats):
    valid_accuracy = Accuracy().to(device)
    valid_epoch_loss = 0
    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(valid_data_loader):
        model.prepare_for_batch(mnist_images, device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        _, outputs = model(mnist_images)
        output = outputs[-1, :, :]
        memory_reading_stats.update_memory_readings(model.memory_reading, epoch=epoch)
        
        loss_value = loss_fn(output.T, targets)
        valid_epoch_loss += loss_value.item() * mnist_images.size(0)

        batch_accuracy = valid_accuracy(output.T, targets)
    valid_accuracy_at_epoch = valid_accuracy.compute()
    valid_epoch_loss /= len(valid_data_loader.sampler)
    valid_accuracy.reset()
    return valid_epoch_loss, valid_accuracy_at_epoch


def training_step(device, model, loss_fn, opt, train_data_loader, epoch, cfg):
    train_accuracy = Accuracy().to(device)

    epoch_loss = 0
    model.train()
    for batch_i, (mnist_images, targets) in enumerate(train_data_loader):
        batch_size = len(mnist_images)   # mnist_images.shape is (BS, 784)
        logging.info(f"MNIST batch {batch_i}")
        model.zero_grad()

        if (epoch == 0) and (batch_i == 0):
            mnist_batch_img = wandb.Image(make_grid(mnist_images.reshape(batch_size, 1, 28, -1)))
            wandb.log({f"Training data batch {batch_i}, epoch {epoch}": mnist_batch_img})

        model.prepare_for_batch(mnist_images, device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        _, outputs = model(mnist_images)
        output = outputs[-1, :, :]
        log_preds_and_targets(batch_i, output, targets)

        loss_value = loss_fn(output.T, targets)
        epoch_loss += loss_value.item() * mnist_images.size(0)

        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm, norm_type=2.0, error_if_nonfinite=True)
        opt.step()

        batch_accuracy = train_accuracy(output.T, targets)

    accuracy_over_batches = train_accuracy.compute()
    epoch_loss /= len(train_data_loader.sampler)
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches


if __name__ == "__main__":
    click_wrapper()
