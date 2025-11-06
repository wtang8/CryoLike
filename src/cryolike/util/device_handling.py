import torch

def get_device(device: str | torch.device | None = None):
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available() and device.type == 'cuda':
        print("Warning: CUDA is requested but not available, switching to CPU.")
        return torch.device('cpu')
    return device