import torch
from .DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from .DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wandb


def read_write_consistency_regularizer(sequence_read_weights, sequence_write_weights, lambda_):
    """This method implements the first regularization term described in the paper.
    The regularization term uses all the read and write weights computed by the model
    processing a sequence, which should therefore be collected during training.

    The method assumes that read and write weights are column vectors, i.e. they have shape (1, N)
    and the batches of weights have shapes (k, N), where N is the number of memory locations and k
    is the sequence length."""
    term = torch.zeros(sequence_read_weights.shape[0])
    for t in range(sequence_read_weights.shape[0]):
        normalized_sum_of_write_weights_up_to_t = sequence_write_weights[:t+1, :].sum(axis=0).view(1, -1) / (t+1)
        scaled_product_with_read_weights = 1 - normalized_sum_of_write_weights_up_to_t.T @ sequence_read_weights[t, :].view(1, -1)
        norm = torch.linalg.matrix_norm(scaled_product_with_read_weights)
        term[t] = norm**2
    return lambda_ * term.sum()


def get_digit_string_repr(digit):
    repr = ''
    digit = (digit.view(28, 28) != 0).to(torch.int32)
    for row in digit:
        for item in row:
            str_item = item.item()
            repr += ' ' + ' ' if str_item == 0 else '0'
        repr += '\n'
    return repr


def save_address_sequence_heatmap(weights_full, read: bool, labels):
    read_or_write = 'read' if read else 'write'
    
    print(f"Saving {read_or_write} heatmap")
    labels = labels.replace('<pad>', ' ')
    fig = plt.figure(figsize=(9, 3))
    sns.heatmap(weights_full, vmin=weights_full.min(), vmax=weights_full.max(),
                cbar=True, xticklabels=labels, yticklabels=False).set_title(read_or_write)
    plt.xticks(rotation=0)
    path = os.path.join(os.getcwd(), f"memory_access_{read_or_write}_full.png")
    plt.savefig(path)
    wandb.log({f"memory_access_{read_or_write}_full": wandb.Image(path)})