import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image, ImageDraw, ImageOps
import argparse
from pathlib import Path
import cv2
from collections import deque
import time
import shutil
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='./input', type=str, dest='path')
parser.add_argument('--new-fps', default='24', type=int, dest='new_fps')
parser.add_argument('--clean', default=False, type=bool, dest='clean')
parser.add_argument('--batch-size', default=2, type=int, dest='batch_size')

args = parser.parse_args()

class Vid_Input(Dataset):
    def __init__(self, vid, transform):
        super(Vid_Input).__init__()
        self.vid = vid
        self.transform = transform

    def __getitem__(self, idx):
        success, frame = self.vid.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            return self.transform(image).to(torch.device('cuda:0'))

    def __len__(self):
        return int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

class Img_Input(Dataset):
    def __init__(self, image, transform):
        self.image = image
        self.transform = transform
    def __getitem__(self, index):
        return self.transform(self.image[index]).to(torch.device('cuda:0'))

    def __len__(self):
        return len(self.image)

def convert_to_img_data(vid, new_fps):
    success, frame = vid.read()
    old_fps = vid.get(cv2.CAP_PROP_FPS)
    fps = old_fps/new_fps
    images=[]
    frame_number = 0
    while success:
        image = Image.fromarray(frame)
        images.append(image)
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = vid.read()
        frame_number += fps

    return images, new_fps

def list_to_mp4(frames, fps, name):
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('./output/' + name[:-4] + '_out.mp4', fourcc, fps, videodims)
    for i in frames:
        video.write(np.array(i))
    video.release()


def infer(model, dataloader, path_list, vid=False, fps=None):
    frames = deque([])
    img_ctr = 0
    for batch in dataloader:
        preds = model(batch)
        labels = [[pred["labels"].cpu().detach().numpy(), pred["scores"].cpu().detach().numpy()] for pred in preds]
        boxes = [pred["boxes"].cpu().detach().numpy() for pred in preds]

        transform = torchvision.transforms.ToPILImage()
        for img_idx, img in enumerate(boxes):
            image = transform(batch[img_idx])
            for cat_idx, cat in enumerate(labels[img_idx][0]):
                if labels[img_idx][1][cat_idx] >= 0.98:
                    bbox = img[cat_idx]
                    if cat == 1:
                        color = 'blue'
                        text = 'summit'

                    elif cat == 2:
                        color = 'red'
                        text  = 'coke'

                    else:
                        color = 'yellow'
                        text = 'juice'

                    ImageDraw.Draw(image).rectangle(bbox, outline=color, width=2)
                    ImageDraw.Draw(image).text((bbox[0], bbox[1]), text=text, fill='black')

            if vid:
                frames.append(image)
            else:
                image.save('./output/' + path_list[img_ctr][:-4] + '_out.jpg')
            img_ctr += 1

    if vid:
        list_to_mp4(frames, fps, path_list)

if __name__=='__main__':
    begin = time.time()
    img_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(480), torchvision.transforms.ToTensor()])
    img_path = [f for f in Path(args.path).glob('*.jpg')]
    img_name = [f.name for f in img_path]
    img_data = [Image.open(f) for f in img_path]
    img_loader = DataLoader(dataset=Img_Input(img_data, img_transform), batch_size=args.batch_size)

    vid_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(480), torchvision.transforms.ToTensor()])
    vids_name = [f for f in Path(args.path).glob('*.mp4')]
    vids_data = [cv2.VideoCapture(args.path+ '/' + f.name) for f in vids_name]
    vids_loader = [[DataLoader(dataset=Vid_Input(vid, vid_transform), batch_size=args.batch_size)] for vid in vids_data]

    checkpoint = torch.load('checkpoint.pth')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=4, weights_backbone='ResNet50_Weights.IMAGENET1K_V1')
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device('cuda:0'))
    model.eval()


    if args.clean:
        shutil.rmtree('./output', ignore_errors=True)
    Path('./output').mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        infer(model, img_loader, img_name)
        for idx, loader in enumerate(vids_loader):
            infer(model, loader[0], vids_name[idx].name, vid=True, fps=args.new_fps)
    end = time.time()
    print(end-begin)
