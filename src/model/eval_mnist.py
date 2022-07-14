import logging
import os

import omegaconf
import torch
import wandb
from torchmetrics.classification import Accuracy

import hydra
from data.perm_seq_mnist import get_dataloaders
from model.builders import build_model
from model.dntm.MemoryReadingsStats import MemoryReadingsStats
from utils.run_utils import configure_reproducibility
from utils.wandb_utils import log_config


@hydra.main(config_path="../../conf/local", config_name="test_model_mnist")
def test_mnist(cfg):
    device = torch.device(cfg.run.device, 0)
    rng = configure_reproducibility(cfg.run.seed)
    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project="dntm_mnist", entity="flapetr", mode=cfg.run.wandb_mode)
    wandb.run.name = cfg.run.codename
    log_config(cfg_dict)

    _, valid_dataloader = get_dataloaders(cfg, rng)
    model = build_model(cfg, device)
    memory_reading_stats = MemoryReadingsStats(path=os.getcwd())
    loss_fn = torch.nn.NLLLoss()

    logging.info("Starting testing phase")
    valid_loss, valid_accuracy = test_step(
        device, model, loss_fn, valid_dataloader, memory_reading_stats
    )
    print(f"Accuracy on validation set: {valid_accuracy}")
    print(f"Loss on validation set: {valid_loss}")
    logging.info(f"Accuracy on validation set: {valid_accuracy}")
    logging.info(f"Loss on validation set: {valid_loss}")


@torch.no_grad()
def test_step(device, model, loss_fn, test_data_loader, memory_reading_stats):
    test_accuracy = Accuracy().to(device)
    test_loss = 0
    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(test_data_loader):
        logging.info(f"MNIST batch {batch_i}")

        model.prepare_for_batch(mnist_images, device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        _, output = model(mnist_images)
        logging.debug(output.T.argmax(dim=1))
        memory_reading_stats.update_memory_readings(model.memory_reading)

        loss_value = loss_fn(output.T, targets)
        test_loss += loss_value.item() * mnist_images.size(0)

        test_accuracy(output.T, targets)
    test_loss /= len(test_data_loader.sampler)
    return test_loss, test_accuracy.compute()


if __name__ == "__main__":
    test_mnist()
