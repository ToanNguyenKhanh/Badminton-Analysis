"""
@author: <nktoan163@gmail.com>
"""

import os
import torch
from ultralytics import YOLO


def train(dataset_path=None, output_dir=None, epoch=1, batch_size=8, imgsz=640):
    training_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", training_device)

    output_dir = output_dir
    folder_name = 'models'
    starting_model = 'yolov8n.pt'
    batch_size = batch_size
    epoch = epoch
    imgsz = imgsz

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_path = dataset_path
    model = YOLO(starting_model)
    model.train(task='detect',
                data=dataset_path,
                epochs=epoch,
                batch=batch_size,
                imgsz=imgsz,
                device=training_device,
                project=output_dir,
                name=folder_name)

if __name__ == '__main__':
    # train(dataset_path='/home/toan/PycharmProjects/Badminton-Player-Tracking-and-Analysis/dataset_yolo/court.yaml',
    #       output_dir='court_output',epoch=50, batch_size=16, imgsz=416)

    train(dataset_path='/home/toan/PycharmProjects/Badminton-Player-Tracking-and-Analysis/data_yolo/shuttle.yaml',
          output_dir='shuttle_output', epoch=100, batch_size=8, imgsz=640)

    train(dataset_path='/home/toan/PycharmProjects/Badminton-Player-Tracking-and-Analysis/data_yolo/player.yaml',
          output_dir='player_output', epoch=100, batch_size=8, imgsz=640)
