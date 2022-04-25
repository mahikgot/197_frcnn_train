import numpy as np
import csv
import json
import copy

with open('segmentation_train.json', 'r') as f:
    seg_train = json.load(f)
with open('labels_train.csv', 'r') as f:
    labels_iter = csv.DictReader(f)
    labels_list = []
    for row in labels_iter:
        labels_list.append(row)

def get_images(data_dict):
    object_keys = data_dict.keys()
    output = { 'images': [] }
    for file in object_keys:
        output['images'].append({'filename': data_dict[file]['filename'], 'id': int(data_dict[file]['filename'][:7])})
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
    counter = 0
    for file in object_keys:
        for region in data_dict[file]['regions']:
            category = copy.deepcopy(inner)
            category['segmentation'] = [[]]
            for x in region['shape_attributes']['all_points_x']:
                category['segmentation'][0].append(int(x))
            for y in region['shape_attributes']['all_points_y']:
                category['segmentation'][0].append(int(y))
            category['id'] = counter
            category['image_id'] = int(data_dict[file]['filename'][:7])
            category['category_id'] = int(region['region_attributes']['Name'])
            bound_im = [region for region in labels_list if region['frame'] == data_dict[file]['filename']]
            if len(bound_im):
                category['bbox'].extend([int(bound_im[0]['xmin']), int(bound_im[0]['ymin']), int(bound_im[0]['xmax']) - int(bound_im[0]['xmin']), int(bound_im[0]['ymax']) - int(bound_im[0]['ymin'])])
            output['annotations'].append(category)
            counter += 1
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
    output['image'] = get_images(seg_train['_via_img_metadata'])
    output['annotation'] = get_annotations(seg_train['_via_img_metadata'], labels_list)
    output['categories'] = get_categories()

    return output
with open('labels.json', 'w') as outfile:
    json.dump(convert('segmentation_train.json', 'labels_train.csv'), outfile)

