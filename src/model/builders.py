from model.dntm.DynamicNeuralTuringMachine import build_dntm

def build_model(model_conf, device):
    return build_dntm(model_conf, device)
