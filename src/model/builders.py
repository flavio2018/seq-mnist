from model.dntm.DynamicNeuralTuringMachine import build_dntm
from model.dntm_var.DynamicNeuralTuringMachine import build_dntm as build_dntm_var


def build_model(cfg, device):
    if cfg.model.name == "dntm_var":
        return build_dntm_var(cfg, device)
    elif cfg.model.name == "dntm":
        return build_dntm(cfg, device)
    return build_dntm(cfg, device)
