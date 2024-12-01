import torch
import torch.nn as nn

latent_space = 1500

encoder = nn.Sequential(
    nn.Conv2d(3, 64, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(64, 128, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(128, 256, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(256, 256, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(256, 256, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Flatten(),
    nn.Linear(256*3*3, latent_space*2),
    nn.ReLU()
)

decoder = nn.Sequential(
    nn.Linear(latent_space, 256*3*3),
    nn.ReLU(),
    nn.Unflatten(1, (256, 3, 3)),
    nn.Upsample(scale_factor=2),
    nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 3, 3, stride=1, padding=1)
)

class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparametrization(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :latent_space]
        log_var = mu_logvar[:, latent_space:]
        return mu, log_var
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    

if __name__ == "__main__":
    model = VAEModel()
    print(model(torch.rand(1, 3, 96, 96))[0].shape)
    torch.save(model, "anigirl_vae_model.pth")