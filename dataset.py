import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

class AniWhoImageDataset(Dataset):
    def __init__(self, img_data_path, img_transforms=None, anime_characters=None, accepted_img_formats=['jpg', 'jpeg', 'png']):
        super(AniWhoImageDataset, self).__init__()
        self.img_data_path = img_data_path
        self.img_paths = []
        self.img_transforms = img_transforms
        self.anime_characters = anime_characters
        if anime_characters:
            for character in anime_characters:
                for format in accepted_img_formats:
                    self.img_paths.extend(glob.glob(os.path.join(self.img_data_path, f"{character}/*.{format}")))
        else:
            for format in accepted_img_formats:
                self.img_paths.extend(glob.glob(os.path.join(self.img_data_path, f"*.{format}")))
                self.img_paths.extend(glob.glob(os.path.join(self.img_data_path, f"*/*.{format}")))
        self.len = len(self.img_paths)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        if self.img_transforms is not None:
            image = self.img_transforms(image)
        image = image/255.0
        return image
    

if __name__ == "__main__":
    dataset = AniWhoImageDataset("./data/dataset", anime_characters=None)
    print(dataset[1].shape)
    print(len(dataset))
    for d in dataset:
        if d.shape != torch.Size([3, 96, 96]):
            print("Unexpected shape", d.shape)
            break
    print("Done")