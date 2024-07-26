"""
@author: <nktoan163@gmail.com>
"""
import os
import cv2
import json
import glob
import shutil

def prepare_dataset(root_dir=None, output_dir=None, train=True, rm_files=True):
    root = os.path.join(root_dir, 'train' if train else 'valid')
    global_width = 640
    global_height = 640
    if rm_files:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        os.makedirs(os.path.join(output_dir, 'train', 'images'))
        os.makedirs(os.path.join(output_dir, 'train', 'labels'))
        os.makedirs(os.path.join(output_dir, 'valid', 'images'))
        os.makedirs(os.path.join(output_dir, 'valid', 'labels'))

    json_file = list(glob.iglob("{}/*.json".format(root)))
    print(json_file)
    images_name = []
    for img_name in os.listdir(root):
        if img_name.endswith('jpg'):
            images_name.append(img_name)

    with open(json_file[0], 'r') as json_file:
        data = json.load(json_file)
        images_json = data['images']
        annotations = data['annotations']

    count = 1
    for image in images_name:
        image_path = os.path.join(root, image)
        read_image = cv2.imread(image_path)
        cv2.imwrite(os.path.join(output_dir, 'train' if train else 'valid', 'images', '{}.jpg'.format(count)),
                    read_image)
        current_bboxes = []
        for obj in images_json:
            if image == obj['file_name']:
                image_id = obj['id']
                for annotation in annotations:
                    if annotation['image_id'] == image_id:
                        bbox = annotation['bbox']
                        category_id = annotation['category_id']
                        current_bboxes.append([bbox, category_id])

        filename = os.path.join(output_dir, 'train' if train else 'valid', 'labels', '{}.txt'.format(count))
        with open(filename, 'w') as text_file:
            for box in current_bboxes:
                (xmin, ymin, width, height), cat = box
                xcent = (xmin + width / 2) / global_width
                ycent = (ymin + height / 2) / global_height
                width /= global_width
                height /= global_height
                text_file.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cat - 1, xcent, ycent, width, height))
        count += 1
        print(count)


if __name__ == '__main__':
    prepare_dataset(root_dir='data_coco/player', output_dir='data_yolo/player', train=True, rm_files=True)
    prepare_dataset(root_dir='data_coco/player', output_dir='data_yolo/player', train=False, rm_files=False)
    prepare_dataset(root_dir='data_coco/shuttle', output_dir='data_yolo/shuttle', train=True, rm_files=True)
    prepare_dataset(root_dir='data_coco/shuttle', output_dir='data_yolo/shuttle', train=False, rm_files=False)
    # prepare_dataset(root_dir='dataset_coco/court', output_dir='dataset_yolo/court', train=True, rm_files=True)
    # prepare_dataset(root_dir='dataset_coco/court', output_dir='dataset_yolo/court', train=False, rm_files=False)

