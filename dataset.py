import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import xml.etree.ElementTree as ET

class VOCDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = [f for f in os.listdir(root) if f.endswith('.jpg') or f.endswith('.png')]
        self.annotations = [f for f in os.listdir(root) if f.endswith('.xml')]
        self.label_map = {"bottle": 1, "can": 2}  # Example label map, update as needed

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images[index])
        ann_path = os.path.join(self.root, self.annotations[index])
        
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.label_map:
                continue
            label_id = self.label_map[label]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return img, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))