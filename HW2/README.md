# 197_frcnn_train
Mark Lewis R. Guiang

train Faster R-CNN with given drinks dataset 
https://arxiv.org/abs/1506.01497
## Execution
Requires cuda 11.3
### Evaluation of Pre-Trained Model
- Installs prerequisites
- Downloads Pre-Trained Weights
- Downloads and Converts drinks dataset to COCO Format
```sh
python test.py
```
### Training of Pre-Trained Model
Trains pre-trained model from epoch 26 to 27
```sh
python train.py
```
## Inference
Put .jpg and .mp4 files inside 'input' folder in root directory of project
```sh
python infer.py
```
Output files located in 'output' folder
Example Output from video input:
https://drive.google.com/file/d/1LNPfI7K9tvTqfzYjfVFKu_iO_d30IqRT/view?usp=sharing
