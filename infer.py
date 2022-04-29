import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageOps
import argparse
from pathlib import Path
import cv2
from collections import deque
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='./input', type=str, dest='path')
args = parser.parse_args()

class Input(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __getitem__(self, index):
        return transform(self.data[index]).to(torch.device('cuda:0'))
    def __len__(self):
        return len(self.data)

def convert_to_img_data(vid):
    success, frame = vid.read()
    fps = vid.get(cv2.CAP_PROP_FPS)
    images=[]
    while success:
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(converted)
        images.append(image)
        success, frame = vid.read()

    return images, fps

def list_to_mp4(frames, fps, name):
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('./output/' + name[:-4] + '_out.mp4', fourcc, fps, videodims)
    for i in frames:
        video.write(cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR))
    video.release()


def infer(model, dataloader, path_list, data_list, vid=False, fps=0):
    img_ctr = 0
    frames = deque([])
    if data_list:
        width, height = data_list[img_ctr].size
        rx = width/640
        ry = height/480
        ray = [rx,ry,rx,ry]
    whole = time.time()
    for batch in dataloader:
        start = time.time()
        preds = model(batch)
        end = time.time()
        print('model:', end-start)
        labels = [[pred["labels"].cpu().detach().numpy(), pred["scores"].cpu().detach().numpy()] for pred in preds]
        boxes = [pred["boxes"].cpu().detach().numpy() for pred in preds]

        start = time.time()
        for img_idx, img in enumerate(boxes):
            for cat_idx, cat in enumerate(labels[img_idx][0]):
                if labels[img_idx][1][cat_idx] >= 0.98:
                    bbox = np.multiply(img[cat_idx], ray)
                    if cat == 1:
                        color = 'blue'
                        text = 'summit'
                    elif cat == 2:
                        color = 'red'
                        text  = 'coke'
                    else:
                        color = 'yellow'
                        text = 'juice'
                    ImageDraw.Draw(data_list[img_ctr]).rectangle(bbox.tolist(), outline=color, width=2)
                    ImageDraw.Draw(data_list[img_ctr]).text((bbox[0], bbox[1]), text=text, fill='black')

            if vid:
                frames.append(data_list[img_ctr])
            else:
                data_list[img_ctr].save('./output/' + path_list[img_ctr].name[:-4] + '_out.jpg')
            img_ctr += 1
        end = time.time()
        print('draw:', end-start)
    whole_end = time.time()
    print('whole:', whole_end-whole)

    huli = time.time()
    if vid:
        list_to_mp4(frames, fps, path_list)
        huli_last = time.time()
        print('output', huli_last-huli)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((480, 640)), torchvision.transforms.ToTensor()])

img_data = [Image.open(f) for f in Path(args.path).glob('*.jpg')]
img_name = [f for f in Path(args.path).glob('*.jpg')]
img_loader = DataLoader(dataset=Input(img_data, transform), batch_size=2)

vids_data = [cv2.VideoCapture(args.path+ '/' + f.name) for f in Path(args.path).glob('*.mp4')]
vids_name = [f for f in Path(args.path).glob('*.mp4')]
frames_list = [convert_to_img_data(vid) for vid in vids_data]
vids_loader = [[DataLoader(dataset=Input(frames[0], transform), batch_size=2), frames[1]] for frames in frames_list]

checkpoint = torch.load('checkpoint.pth')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=4, weights_backbone='ResNet50_Weights.IMAGENET1K_V1')
model.load_state_dict(checkpoint['model'])
model.to(torch.device('cuda:0'))
model.eval()

print('tang')
Path('./output').mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    infer(model, img_loader, img_name, img_data)
    for idx, loader in enumerate(vids_loader):
        infer(model, loader[0], vids_name[idx].name,  frames_list[idx][0], fps=loader[1], vid=True)
