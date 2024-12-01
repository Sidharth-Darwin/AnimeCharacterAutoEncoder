import torch
import torch.nn as nn


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
    nn.Linear(256*3*3, 1500),
    nn.ReLU()
)

decoder = nn.Sequential(
    nn.Linear(1500, 256*3*3),
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
    nn.Conv2d(64, 3, 3, stride=1, padding=1),
    nn.Sigmoid()
)

class AutoEncoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

if __name__ == "__main__":
    model = AutoEncoderModel(encoder, decoder)
    print(model(torch.rand(1, 3, 96, 96)).shape)
    checkpoint = {
        "encoder": encoder,
        "decoder": decoder
    }
    torch.save(checkpoint, "anigirl2_full_model.pth")