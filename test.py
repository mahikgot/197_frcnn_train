import subprocess
from requests import get

def download():
    url = 'https://github.com/mahikgot/197_frcnn_train/releases/download/model/checkpoint.pth'
    output  = 'checkpoint.pth'
    print('Downloading model')
    file = get(url)
    with open(output, 'wb') as f:
        f.write(file.content)
    print('Done downloading')

# subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
# subprocess.run(['pip', 'install', '--pre', 'torch', 'torchvision', '--extra-index-url', 'https://download.pytorch.org/whl/nightly/cu113', '-U'])
# download()
# subprocess.run(['python', 'convert.py'])
subprocess.run(['python', 'detection/train.py', '--resume', 'checkpoint.pth','--test-only', '--data-path','./dataset', '--model', 'fasterrcnn_resnet50_fpn', '--workers', '1', '--lr', '0.0025', '--weights-backbone', 'ResNet50_Weights.IMAGENET1K_V1'])


