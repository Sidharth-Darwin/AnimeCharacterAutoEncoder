import torch
from dataset import AniWhoImageDataset
from vae_model import VAEModel, latent_space
from utils import vae_adjust_interpolation_using_graph, merge_features

model_weight_path = "ani_vae_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# anime_characters = ['hatsune_miku', 'albedo', 'aqua_(konosuba)', 'asuna_(sao)', 'c.c', 'chitanda_eru', 'chloe_von_einzbern', 'emilia_rezero', 'fubuki_(one-punch_man)', 'fujiwara_chika', 'hyuuga_hinata', 'illyasviel_von_einzbern', 'ishtar_(fate_grand_order)', 'kamado_nezuko', 'makise_kurisu', 'megumin', 'nico_robin', 'senjougahara_hitagi', 'zero_two_(darling_in_the_franxx)']

anime_characters = ['hatsune_miku']

dataset = AniWhoImageDataset("./data/dataset", anime_characters=anime_characters)
model = VAEModel().to(device)
model.load_state_dict(torch.load(model_weight_path))
model.eval()

encoded_images = []
for data in dataset:
    data = data.to(device)
    output = model.encoder(data.unsqueeze(0)).squeeze()[:latent_space]
    encoded_images.append(output.detach().cpu())
while True:
    img_idx1 = int(input("Enter index of image 1: "))
    img_idx2 = int(input("Enter index of image 2: "))
    if img_idx1 >= len(encoded_images) or img_idx2 >= len(encoded_images):
        print("Invalid index")
        break
    img_1 = dataset[img_idx1]
    img_2 = dataset[img_idx2]
    vae_adjust_interpolation_using_graph(img_1, img_2, model.encoder, model.decoder, latent_space, merge_features)
    do_continue = input("Do you want to continue? (y/n): ")
    if do_continue.lower() == "n":
        break
