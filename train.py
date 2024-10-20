from time import perf_counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, RandomAdjustSharpness, RandomAutocontrast, RandomHorizontalFlip, RandomRotation, RandomPosterize, InterpolationMode
from dataset import AniWhoImageDataset
from model import AutoEncoderModel, encoder, decoder
from utils import train_step, show_image

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# anime_characters = ['hatsune_miku', 'albedo', 'aqua_(konosuba)', 'asuna_(sao)', 'c.c', 'chitanda_eru', 'chloe_von_einzbern', 'emilia_rezero', 'fubuki_(one-punch_man)', 'fujiwara_chika', 'hyuuga_hinata', 'illyasviel_von_einzbern', 'ishtar_(fate_grand_order)', 'kamado_nezuko', 'makise_kurisu', 'megumin', 'nico_robin', 'senjougahara_hitagi', 'zero_two_(darling_in_the_franxx)'] # Some female character

anime_characters = None

img_transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomPosterize(bits=6, p=0.5),
    RandomRotation((-10, 10), interpolation=InterpolationMode.BILINEAR),
    RandomAutocontrast(p=0.5),
    RandomAdjustSharpness(sharpness_factor=1.5, p=0.5)
])
dataset = AniWhoImageDataset("./data/dataset", img_transforms=img_transforms, anime_characters=anime_characters)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True)
print("Length of data loader: ", len(train_loader))

model = AutoEncoderModel(encoder, decoder).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=20, min_lr=0.0001)

start = perf_counter()
current = start
for epoch in range(1000):
    loss = train_step(model, train_loader, criterion, optimizer, device)
    # scheduler.step(loss)
    if epoch % 50 == 0:
        last = current
        current = perf_counter()
        print(f"Epoch: {epoch} Time taken: {current - last} seconds")
        print(f"Loss: {loss.item()}")
        # print(f"Loss: {loss.item()} Learning rate: {scheduler.get_last_lr()[0]}")
        print("-" * 80)
print(f"Epoch: {epoch}")
print(f"Loss: {loss}")
print("-" * 80)
print(f"Total Time taken: {current - start} seconds")

# if anime_characters:
#     torch.save(model.state_dict(), f"./{'-'.join([ch[:5] for ch in anime_characters])}_model.pth")
# else:
#     torch.save(model.state_dict(), "anifemale_model.pth")
torch.save(model.state_dict(), "ani_model.pth")

model.eval()
data = dataset[np.random.randint(len(dataset))].unsqueeze(0)
data = data.to(device)
output = model(data)
show_image(output.squeeze(), data.squeeze(), "Output")
