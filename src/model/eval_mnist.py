import hydra
import logging
import wandb

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from data.perm_seq_mnist import get_dataloaders
from model.builders import build_model
from utils.run_utils import configure_reproducibility
from model.dntm.MemoryReadingsStats import MemoryReadingsStats


@hydra.main(config_path="../../conf/local", config_name="test_model_mnist")
def test_mnist(cfg):
    device = torch.device("cuda", 0)
    rng = configure_reproducibility(cfg.run.seed)

    _, valid_dataloader = get_dataloaders(cfg, rng)
    model = build_model(cfg.model, device)
    memory_reading_stats = MemoryReadingsStats()
    memory_reading_stats.init_random_matrix()

    logging.info("Starting testing phase")
    valid_accuracy = test_step(device, model, valid_dataloader, memory_reading_stats)
    memory_reading_stats.compute_stats()
    print(f"Accuracy on validation set: {valid_accuracy}")
    print(memory_reading_stats)
    logging.info(f"Accuracy on validation set: {valid_accuracy}")
    logging.info(memory_reading_stats.get_stats())


def test_step(device, model, test_data_loader, memory_reading_stats):
    test_accuracy = Accuracy().to(device)

    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(test_data_loader):
        logging.info(f"MNIST batch {batch_i}")

        model.prepare_for_batch(mnist_images, device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        _, output = model(mnist_images)
        output = output[-1, :, :]
        memory_reading_stats.update_memory_readings(model.memory_reading)

        batch_accuracy = test_accuracy(output.T, targets)
    return test_accuracy.compute()


if __name__ == '__main__':
    test_mnist()