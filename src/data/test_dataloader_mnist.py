import hydra
import omegaconf

from src.data.perm_seq_mnist import get_dataloaders
from src.utils import configure_reproducibility
from src.models.train_dntm_utils import get_digit_string_repr


@hydra.main("../../conf/local", "test_data_mnist")
def main(cfg):
    rng = configure_reproducibility(cfg.run.seed)
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    train_dataloader, valid_dataloader = get_dataloaders(cfg, rng)
    visualize_first_element(train_dataloader)
    visualize_first_element(valid_dataloader)


def visualize_first_element(data_loader):
    for batch, targets in data_loader:
        first_element = batch[0]
        first_target = targets[0]
        print(get_digit_string_repr(first_element), first_target.item())
        return


if __name__ == "__main__":
    main()
