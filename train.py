import subprocess

subprocess.run(['python', 'detection/train.py', '--resume', 'checkpoint.pth','--data-path', './dataset', '--model', 'fasterrcnn_resnet50_fpn', '--workers', '1', '--lr', '0.0025', '--weights-backbone', 'ResNet50_Weights.IMAGENET1K_V1', '--epochs', '27','--device', 'cuda:0'])

