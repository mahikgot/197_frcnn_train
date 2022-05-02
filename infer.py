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
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='./input', type=str, dest='path')
parser.add_argument('--clean', default=False, type=bool, dest='clean')
parser.add_argument('--batch-size', default=4, type=int, dest='batch_size')
parser.add_argument('--num-workers', default=0, type=int, dest='num_workers')

args = parser.parse_args()

class Vid_Input(Dataset):
    def __init__(self, vid, transform):
        super(Vid_Input).__init__()
        self.vid = vid
        self.transform = transform
        self.get_ar()


    def __getitem__(self, idx):
        success, frame = self.vid.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.copyMakeBorder(frame, self.border_v, self.border_v, self.border_h, self.border_h, cv2.BORDER_CONSTANT, 0)
            frame = cv2.resize(frame, (640, 480))
            image = Image.fromarray(frame)
            return self.transform(image).to(torch.device('cuda:0'))

    def __len__(self):
        return int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_ar(self):
        _, frame = self.vid.read()
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, -1)
        # https://stackoverflow.com/questions/48437179/opencv-resize-by-filling
        self.border_v = 0
        self.border_h = 0
        height = frame.shape[0]
        width = frame.shape[1]
        if (0.75) >= (height/width):
            self.border_v = int((((0.75)*width)-height)/2)
        else:
            self.border_h = int((((4/3)*height)-width)/2)
class Img_Input(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        return self.transform(image).to(torch.device('cuda:0'))

    def __len__(self):
        return len(self.images)

def list_to_mp4(frames, fps, name):
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('./output/' + name[:-4] + '_out.mp4', fourcc, fps, videodims)
    for i in frames:
        video.write(cv2.cvtColor(np.array(i), cv2.COLOR_BGR2RGB))
    video.release()


def infer(model, dataloader, path_list, vid=False, fps=None):
    frames = deque([])
    img_ctr = 0
    for batch in dataloader:
        preds = model(batch)

        transform = torchvision.transforms.ToPILImage()
        for im in range(len(batch)):
            image = transform(batch[im])
            for idx, score in enumerate(preds[im]['scores']):
                if score >= 0.98:
                    bbox = [preds[im]['boxes'][idx][i].item() for i in range(4)]
                    cat = preds[im]['labels'][idx].item()
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
    img_loader = DataLoader(dataset=Img_Input(img_path, img_transform), batch_size=args.batch_size, num_workers=args.num_workers)

    vid_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
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
            fps = vids_data[idx].get(cv2.CAP_PROP_FPS)
            infer(model, loader[0], vids_name[idx].name, vid=True, fps=fps)
    end = time.time()
    print('time:', end-begin)
