"""

MIT License

Copyright 2024 Maximilian Petersson and Nahom Solomon

"""



import os
from ultralytics import YOLO

dataset_dir_general = '/content/drive/MyDrive/II142X/datasets/general_dataset.yaml'

model_general = YOLO('yolov8n.yaml')

model_general.train(data=dataset_dir_general, epochs=100, patience=20)

model_general.val(data=dataset_dir_general)

success_specific = model_general.export(format='onnx')