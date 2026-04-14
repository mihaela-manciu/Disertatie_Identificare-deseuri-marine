import rasterio
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional


class MaridaPatchDataset(Dataset):
    """
    Dataset PyTorch  pentru patch-uri MARIDA (pentru train)    """

    def __init__(self, patches_dir: str, split_file: Optional[str] = None):
        self.patches_dir = Path(patches_dir)
        self.patch_paths = []

        # extrage numele(adauga prefixul "S2_" dacă lipseste in train_X.txt)
        valid_prefixes = set()
        if split_file and Path(split_file).exists():
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        nume = Path(line).stem

                        if not nume.startswith("S2_"):
                            nume = f"S2_{nume}"
                        valid_prefixes.add(nume)
            print(f"Se cauta {len(valid_prefixes)} foldere ")

        #cauta toate fișierele tif de pe disk
        toate_fisierele = list(self.patches_dir.rglob("*.tif"))

        #  potrivire nume imagini
        for patch_file in toate_fisierele:
            # extrage numele folderului in care este patch_0.tif (ex: S2_1-12-19_48MYU_0)
            parent_folder = patch_file.parent.name

            if not split_file or parent_folder in valid_prefixes:
                self.patch_paths.append(patch_file)

        print(f"S-au gasit si incarcat {len(self.patch_paths)} patch-uri finale gata de antrenare!")

        # constante specifice pentru Sentinel-2
        self.max_reflectance = 3000.0
        # medii si deviatii standard specifice apei/coastei in Sentinel-2(nu ImageNet)
        self.s2_mean = np.array([0.05, 0.08, 0.07]).reshape(3, 1, 1)
        self.s2_std = np.array([0.04, 0.04, 0.05]).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.patch_paths[idx]

        with rasterio.open(img_path) as src:
            data = src.read([1, 2, 3])

        #trece la float32
        data = data.astype(np.float32)

        #normalizare in [0, 1]
        data = np.clip(data, 0, self.max_reflectance) / self.max_reflectance

        # standardizare specifica satelitara
        data = (data - self.s2_mean) / self.s2_std

        #forteaza PyTorch sa pastreze datele pe 32-bit (FloatTensor)
        tensor_data = torch.from_numpy(data).float()

        return tensor_data, str(img_path)



class SatelliteResNetExtractor:
    """
    ResNet18 modificat pentru a nu pierde informatia spatiala a obiectelor sub-pixel
    """

    def __init__(self, batch_size: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializare extractor pe: {self.device}")

        #incarcare model ( weights standard pentru a avea filtre de detectare a formelor/muchiilor)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modificare pt date satelitare (patch-uri 64x64)
        # 1. inlocuire convolutie  (7x7, pas 2) cu  (3x3, pas 1)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # copiaza "cunostintele" din vechea convolutie printr-o suma
        with torch.no_grad():
            resnet.conv1.weight.copy_(
                torch.sum(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).conv1.weight, dim=(2, 3),
                          keepdim=True).expand_as(resnet.conv1.weight) / 9.0)

        # anulare MaxPool-ul care distruge detaliile mici (plasticul plutitor)
        resnet.maxpool = nn.Identity()

        #eliminare ultimul strat (Fully Connected) pentru a extrage vectorul latent
        self.model = nn.Sequential(*(list(resnet.children())[:-1]))

        self.model = self.model.to(self.device)
        self.model.eval()  # opreste straturile de Dropout și BatchNormalization

    def extract(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[str]]:
        all_features = []
        all_paths = []


        with torch.no_grad():
            for batch_imgs, batch_paths in tqdm(dataloader, desc="Extragere Trasaturi Latente"):
                batch_imgs = batch_imgs.to(self.device)

                # output are forma (Batch, 512, H, W), deoarece patch-urile sunt 64x64,
                # la final ajung un tensor mic, aplica o mediere spatiala globala
                features = self.model(batch_imgs)
                features = torch.flatten(features, start_dim=1)

                all_features.append(features.cpu().numpy())
                all_paths.extend(batch_paths)

        return np.vstack(all_features), all_paths



if __name__ == "__main__":

    PATCHES_OUTPUT_DIR = r"D:\TAID\Disertatie\MARIDA\Patches_Output"

    TRAIN_SPLIT_FILE = r"D:\TAID\Disertatie\MARIDA\splits\train_X.txt"

    try:

        print("\n 1. Incarcare dataset")
        dataset = MaridaPatchDataset(patches_dir=PATCHES_OUTPUT_DIR, split_file=TRAIN_SPLIT_FILE)

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        print(f"Total patch-uri de procesat pentru antrenare: {len(dataset)}")


        print("\n 2: Extragere trasaturi")
        extractor = SatelliteResNetExtractor(batch_size=64)
        features_matrix, file_paths = extractor.extract(dataloader)

        print(f"\n  Dimensiunea matricei: {features_matrix.shape}")

        print("\n 3. Salvare date ")
        np.save("marida_train_features.npy", features_matrix)

        with open("marida_train_paths.txt", "w") as f:
            for p in file_paths:
                f.write(f"{p}\n")

        print("Au fost create fisierele: 'marida_train_features.npy' si 'marida_train_paths.txt'")

    except Exception as e:
        print(f"Eroare: {e}")