"""
@author: <nktoan163@gmail.com>
"""
import cv2
import json
from pprint import pprint

file_name = 'b1_mp4-3330_jpg.rf.3c2d00486b065ca82641cf8e6ed668b8.jpg'
image = cv2.imread('dataset_coco/court/train/{}'.format(file_name))

annotation_file_path = 'dataset_coco/court/train/_annotations.coco.json'
with open(annotation_file_path, "r") as json_file:
    json_data = json.load(json_file)
    annotations = json_data['annotations']
    category = json_data['categories']

# pprint(annotations)
for annotation in annotations:
    image_id = annotation['image_id']
    # print("image_id:", image_id)

image_id = None
for img in json_data['images']:
    if img['file_name'] == file_name:
        image_id = img['id']
        break

if image_id is not None:
    for ann in annotations:
        if ann['image_id'] == image_id:
            bbox = ann['bbox']
            pprint(bbox)
            xmin, ymin, width, height = map(int, bbox)
            cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)

cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)

