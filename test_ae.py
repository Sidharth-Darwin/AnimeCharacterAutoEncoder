import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from dataset import AniWhoImageDataset
from ae_model import AutoEncoderModel
from utils import interpolate, show_interpolated_image, adjust_interpolation_using_graph, merge_features

model_path = "anigirl2_full_model.pth"
model_weight_path = "ani_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# anime_characters = ['hatsune_miku', 'albedo', 'aqua_(konosuba)', 'asuna_(sao)', 'c.c', 'chitanda_eru', 'chloe_von_einzbern', 'emilia_rezero', 'fubuki_(one-punch_man)', 'fujiwara_chika', 'hyuuga_hinata', 'illyasviel_von_einzbern', 'ishtar_(fate_grand_order)', 'kamado_nezuko', 'makise_kurisu', 'megumin', 'nico_robin', 'senjougahara_hitagi', 'zero_two_(darling_in_the_franxx)']

anime_characters = ['hatsune_miku']

dataset = AniWhoImageDataset("./data/dataset", anime_characters=anime_characters)
checkpoint = torch.load(model_path)
encoder = checkpoint["encoder"]
decoder = checkpoint["decoder"]
model = AutoEncoderModel(encoder, decoder).to(device)
model.load_state_dict(torch.load(model_weight_path))
model.eval()


# if not os.path.exists("gmm_enc_model.pkl"):
#     encoded_images = []
#     for data in dataset:
#         data = data.to(device)
#         output = model.encoder(data.unsqueeze(0)).squeeze()
#         encoded_images.append(output.detach().cpu())

#     mx = GaussianMixture(n_components=1, covariance_type="tied", random_state=42)
#     mx.fit(encoded_images)

#     with open("gmm_enc_model.pkl", "wb") as f:
#         pickle.dump(mx, f)
# else:
#     with open("gmm_enc_model.pkl", "rb") as f:
#         mx = pickle.load(f)


# n_images = 10
# generated_enc, _ = mx.sample(n_samples=n_images)

# for i in range(n_images):
#     enc = torch.tensor(generated_enc[i], dtype=torch.float32)
#     enc = enc.to(device)
#     output = model.decoder(enc.unsqueeze(0)).squeeze()
#     plt.imshow(output.detach().cpu().permute(1, 2, 0))
#     plt.show()
    


encoded_images = []
for data in dataset:
    data = data.to(device)
    output = model.encoder(data.unsqueeze(0)).squeeze()
    encoded_images.append(output.detach().cpu())
while True:
    img_idx1 = int(input("Enter index of image 1: "))
    img_idx2 = int(input("Enter index of image 2: "))
    if img_idx1 >= len(encoded_images) or img_idx2 >= len(encoded_images):
        print("Invalid index")
        break
    img_1 = dataset[img_idx1]
    img_2 = dataset[img_idx2]
    adjust_interpolation_using_graph(img_1, img_2, model.encoder, model.decoder, merge_features)
    do_continue = input("Do you want to continue? (y/n): ")
    if do_continue.lower() == "n":
        break
