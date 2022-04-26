import numpy as np
import csv
import json
import copy
from pycocotools import mask
import gdown
import os
import tarfile
from pathlib import Path

def get_images(data_dict):
    object_keys = data_dict.keys()
    output = { 'images': [] }
    for file in object_keys:
        if len(data_dict[file]['regions']):
            output['images'].append({'file_name': data_dict[file]['filename'], 'id': int(data_dict[file]['filename'][:7]), 'width': 640, 'height': 480})
    return output['images']

def get_annotations(data_dict, labels_list):
    output = { 'annotations': [] }
    inner = {
            'segmentation': None,
            'area': None,
            'iscrowd': 0,
            'image_id': None,
            'bbox': [],
            'category_id': None,
            'id': None
            }
    object_keys = data_dict.keys()
    counter = 1
    bound_counter = 0
    for file in object_keys:
        for region in data_dict[file]['regions']:
            category = copy.deepcopy(inner)
            category['segmentation'] = [[]]
            points = [region['shape_attributes']['all_points_x'],  region['shape_attributes']['all_points_y']]
            for point in zip(points[0], points[1]):
                category['segmentation'][0].append(point[0])
                category['segmentation'][0].append(point[1])
            category['area'] = int(mask.area(mask.frPyObjects(category['segmentation'], 480, 640))[0])
            category['id'] = counter
            category['image_id'] = int(data_dict[file]['filename'][:7])
            category['category_id'] = int(region['region_attributes']['Name'])
            bound_im = [region for region in labels_list if region['frame'] == data_dict[file]['filename']]
            if bound_counter >= len(bound_im):
                bound_counter = 0
            if len(bound_im):
                category['bbox'].extend([int(bound_im[bound_counter]['xmin']), int(bound_im[bound_counter]['ymin']), int(bound_im[bound_counter]['xmax']) - int(bound_im[bound_counter]['xmin']), int(bound_im[bound_counter]['ymax']) - int(bound_im[bound_counter]['ymin'])])
            output['annotations'].append(category)
            counter += 1
            bound_counter += 1
    return output['annotations']
def get_categories():
    cat = [
            {
                'supercategory': 'drink',
                'id': 1,
                'name': 'summit'
                },
            {
                'supercategory': 'drink',
                'id': 2,
                'name': 'coke'
                },
            {
                'supercategory': 'drink',
                'id': 3,
                'name': 'juice'
                }
        ]
    return cat
def convert(json_fname, label_fname):
    with open(json_fname, 'r') as f:
        seg_train = json.load(f)
    with open(label_fname, 'r') as f:
        labels_iter = csv.DictReader(f)
        labels_list = []
        for row in labels_iter:
            labels_list.append(row)
    output = {}
    output['images'] = get_images(seg_train['_via_img_metadata'])
    output['annotations'] = get_annotations(seg_train['_via_img_metadata'], labels_list)
    output['categories'] = get_categories()
    output['info'] = {}
    output['licenses']  = {}

    return output

def download():
    url = 'https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA'
    output = 'drinks.tar.gz'
    if not Path('./' + output).exists:
        gdown.download(url, output, quiet=False)


    targz = tarfile.open('drinks.tar.gz', 'r:gz')
    targz.extractall()
    targz.close

def filter():
    Path('./dataset/images').mkdir(parents=True, exist_ok=True)
    Path('./dataset/annotations').mkdir(parents=True, exist_ok=True)
    Path('./dataset/old_annotations').mkdir(parents=True, exist_ok=True)

    [f.unlink() for f in Path('./drinks').glob('.*')]
    [f.replace('./dataset/images/' + f.name) for f in Path('./drinks').glob('*.jpg')]
    [f.replace('./dataset/old_annotations/' + f.name) for f in Path('./drinks').glob('*')]
    Path('./drinks').rmdir()

def ayos(train_out, val_out):
    Path('./dataset/train').mkdir(parents=True, exist_ok=True)
    Path('./dataset/val').mkdir(parents=True, exist_ok=True)

    [Path('./dataset/images/' + f['file_name']).replace('./dataset/train/' + f['file_name']) for f in train_out['images']]
    [Path('./dataset/images/' + f['file_name']).replace('./dataset/val/' + f['file_name']) for f in val_out['images']]

download()
filter()
train_out = convert('./dataset/old_annotations/segmentation_train.json', './dataset/old_annotations/labels_train.csv')
val_out = convert('./dataset/old_annotations/segmentation_test.json', './dataset/old_annotations/labels_test.csv')
ayos(train_out, val_out)

with open('./dataset/annotations/instances_train.json', 'w') as outfile:
    json.dump(train_out, outfile)
with open('./dataset/annotations/instances_val.json', 'w') as outfile:
    json.dump(val_out, outfile)

