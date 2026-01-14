import torch

from policy.maact.optical_flow.run import get_model


def predict(first, second, device="cuda"):
    with torch.no_grad():
        netNetwork = get_model(device)
        intWidth = first.shape[-1]
        intHeight = first.shape[-2]
        flow = torch.nn.functional.interpolate(input=netNetwork(first, second), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
        return flow.detach()