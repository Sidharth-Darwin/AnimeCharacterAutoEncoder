# Data was downloaded from https://github.com/mgradyn/AniWho/blob/main/Dataset/dataset.zip

import os
import zipfile
import urllib.request


def download_data(data_path="./data"):
    print("Saving data at ", os.path.abspath(data_path))
    url = "https://github.com/mgradyn/AniWho/blob/main/Dataset/dataset.zip?raw=true"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    urllib.request.urlretrieve(url, os.path.join(data_path, "dataset.zip"))
    with zipfile.ZipFile(os.path.join(data_path, "dataset.zip"), "r") as zip_ref:
        zip_ref.extractall(data_path)
    os.remove(os.path.join(data_path, "dataset.zip"))
    print("Data saved!")


if __name__ == "__main__":
    download_data()