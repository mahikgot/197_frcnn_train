import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='./input', type=str, dest='path')
args = parser.parse_args()

class Input(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __getitem__(self, index):
        return transform(self.data[index]).cuda()
    def __len__(self):
        return len(self.data)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((480, 640)), torchvision.transforms.ToTensor()])
data = [Image.open(f) for f in Path(args.path).glob('*.jpg')]
name = [f for f in Path(args.path).glob('*.jpg')]
loader = DataLoader(dataset=Input(data, transform), batch_size=2)


checkpoint = torch.load('checkpoint.pth')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=4, weights_backbone='ResNet50_Weights.IMAGENET1K_V1')
model.load_state_dict(checkpoint['model'])
model.cuda()
model.eval()


def infer(model, dataloader, path_list, data_list):
    Path('./output').mkdir(parents=True, exist_ok=True)
    img_ctr = 0
    for batch in dataloader:
        preds = model(batch)
        labels = [[pred["labels"].cpu().detach().numpy(), pred["scores"].cpu().detach().numpy()] for pred in preds]
        boxes = [pred["boxes"].cpu().detach().numpy() for pred in preds]

        print(preds)
        for img_idx, img in enumerate(boxes):
            for cat_idx, cat in enumerate(labels[img_idx][0]):
                if labels[img_idx][1][cat_idx] >= 0.95:
                    bbox = img[cat_idx]
                    if cat == 1:
                        color = 'blue'
                    elif cat == 2:
                        color = 'red'
                    else:
                        color = 'yellow'
                    draw = ImageDraw.Draw(data_list[img_ctr]).rectangle(bbox, outline=color)
            data_list[img_ctr].save('./output/' + path_list[img_ctr].name[:-4] + '_out.jpg')
            img_ctr += 1

infer(model, loader, name, data)
