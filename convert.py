import numpy as np
import csv
import json
import copy

def get_images(data_dict):
    object_keys = data_dict.keys()
    output = { 'images': [] }
    for file in object_keys:
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
            for x in region['shape_attributes']['all_points_x']:
                category['segmentation'][0].append(int(x))
            for y in region['shape_attributes']['all_points_y']:
                category['segmentation'][0].append(int(y))
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
with open('./dataset/annotations/instances_train.json', 'w') as outfile:
    json.dump(convert('./dataset/segmentation_train.json', './dataset/labels_train.csv'), outfile)
with open('./dataset/annotations/instances_val.json', 'w') as outfile:
    json.dump(convert('./dataset/segmentation_test.json', './dataset/labels_test.csv'), outfile)


