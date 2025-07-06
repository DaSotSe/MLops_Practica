# datasets.py
# dataset.py

import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

class POIDataset(Dataset):
    """
    Dataset multimodal con inputs separados: etiquetas, metadatos y rutas de imagen.
    """
    def __init__(self, target, image_paths, features, transform=None):
        assert len(target) == len(features), "Debe haber el mismo número de targets que de features"
        assert len(target) == len(image_paths), "Debe haber el mismo número de targets que de imágenes"

        self.target = torch.tensor(target, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        target = self.target[idx]
        features = self.features[idx]

        img_path = self.image_paths[idx]
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)

        if self.transform is not None:
            im = self.transform(im)

        return features, im, target 

''' import os
import cv2
from PIL import Image

class POIDataset(Dataset):
    # … tu __init__, __len__ …

    def __getitem__(self, idx):
        # … carga de features, label, etc. …

        img_path = self.image_paths[idx]
        # 1) Comprobar que existe el fichero
        if not os.path.isfile(img_path):
            # Aquí puedes:
            #  - lanzar un error claro
            #  - o usar una imagen placeholder en su lugar
            raise FileNotFoundError(f"No existe la imagen: {img_path}")

        # 2) Intentar leerlo
        im = cv2.imread(img_path)
        if im is None:
            raise IOError(f"OpenCV no pudo leer la imagen (posible ruta corrupta o formato no soportado): {img_path}")

        # 3) Convertir color y resto del pipeline:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        if self.transform:
            im = self.transform(im)

        # 4) Devolver
        return features, im, target''' 

''' import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

class POIDataset(Dataset):
    """
    Dataset multimodal con inputs separados: etiquetas, metadatos y rutas de imagen.
    """
    def __init__(self, target, image_paths, features, transform=None):
        assert len(target) == len(features), "Debe haber el mismo número de targets que de features"
        assert len(target) == len(image_paths), "Debe haber el mismo número de targets que de imágenes"

        self.target = torch.tensor(target, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Cargar etiqueta y características numéricas
        target = self.target[idx]
        features = self.features[idx]

        # Leer y validar la imagen
        img_path = self.image_paths[idx]
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"No existe la imagen: {img_path}")

        im = cv2.imread(img_path)
        if im is None:
            raise IOError(f"OpenCV no pudo leer la imagen: {img_path}")

        # Convertir BGR->RGB y a PIL
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)

        # Aplicar transformaciones si existen
        if self.transform is not None:
            im = self.transform(im)

        return features, im, target'''

