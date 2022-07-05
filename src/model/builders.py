from model.dntm.DynamicNeuralTuringMachine import build_dntm

def build_model(cfg, device):
    return build_dntm(cfg, device)
