import torch

def to_device(batch, device, key='samples'):
    if isinstance(batch[key], torch.Tensor):
        samples = batch[key].to(device)
    else:
        # For dictionary samples (like PubMed dataset), move tensors to device
        samples = {}
        for k, v in batch[key].items():
            if isinstance(v, torch.Tensor):
                samples[k] = v.to(device)
            else:
                samples[k] = v
    return samples