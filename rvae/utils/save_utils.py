import torch
from torchvision import datasets, transforms
from rvae.models.vae import RVAE, VAE

def save_model(model, optimizer, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)


def load_model(load_path, model, optimizer, device):
    checkpoint = torch.load(load_path, map_location=device)
    if isinstance(model, RVAE):
        checkpoint['model_state_dict']['pr_means'] = checkpoint['model_state_dict']['pr_means'][0]
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
