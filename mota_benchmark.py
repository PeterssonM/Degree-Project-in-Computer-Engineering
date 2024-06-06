"""

MIT License

Copyright 2024 Maximilian Petersson and Nahom Solomon

"""



import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from scipy.optimize import linear_sum_assignment

base_path = ""

images_dir = ''
labels_dir = ''

#TRACKER = 'botsort.yaml'
TRACKER = 'bytetrack.yaml'
SHOW = False

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(np.array(img))
    return images

def load_txt_files_from_folder(folder):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r') as file:
                lines = file.readlines()
                boxes = []
                for line in lines:
                    # Split each line by whitespace and convert strings to floats
                    box_coordinates = [float(coord) for coord in line.strip().split()[1:]]
                    boxes.append(box_coordinates)
                texts.append(boxes)
    return texts

def generate_numbered_filepath():
    i = 1
    while True:
        numbered_path = os.path.join(base_path, f"results{i}")
        if not os.path.exists(numbered_path):
            os.makedirs(numbered_path)
            return numbered_path
        i += 1

def save_img(image, result_object, save_path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for box in result_object[0].boxes:
        box_id = int(box.id.item())
        box_xywh = [item for sublist in box.xywh.tolist() for item in sublist]
        x_center, y_center, width, height = box_xywh 
        
        # Top-left corner coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        
        # Bottom-right corner coordinates
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(image_bgr, str(box_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(save_path, image_bgr)

def write_metrics_to_file(metrics, file_path):
    with open(file_path, 'w') as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

def get_tracking_results(model, images, path_to_store_predicted_results):
    total_detections = []
    total_detections_ids = []

    for i, image in enumerate(images):
        result_object = model.track(image, tracker=TRACKER, conf=0.257, persist=True, save=False, show=SHOW)
        
        save_img(image, result_object, os.path.join(path_to_store_predicted_results, f"result_{i}.jpg"))

        temp_box_xywhn = []
        temp_box_id = []
        for box in result_object[0].boxes:
            temp_box_xywhn.append([item for sublist in box.xywhn.tolist() for item in sublist])
            temp_box_id.append(box.id.item())
        
        total_detections.append(temp_box_xywhn)
        total_detections_ids.append(temp_box_id)

    return total_detections, total_detections_ids

def calculate_iou(box1, box2):
    # Convert [x, y, w, h] format to [x1, y1, x2, y2] format for easier calculation
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection coordinates
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate area of each box
    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]

    # Calculate Union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def count_identity_switches(prev_detections, prev_detections_ids, curr_detections, curr_detections_ids):
    # Compute cost matrix
    cost_matrix = np.zeros((len(prev_detections), len(curr_detections)))
    for i, prev_det in enumerate(prev_detections):
        for j, curr_det in enumerate(curr_detections):
            cost_matrix[i, j] = 1 - calculate_iou(prev_det, curr_det)  # 1 - IoU as cost

    # Solve assignment problem using the Hungarian Algorithm
    prev_indices, curr_indices = linear_sum_assignment(cost_matrix)

    # Count unmatched detections (identity switches)
    num_identity_switches = 0
    for prev_index, curr_index in zip(prev_indices, curr_indices):
        if(prev_detections_ids[prev_index] != curr_detections_ids[curr_index]):
            num_identity_switches += 1

    return num_identity_switches

def count_total_detections(total_detections):
    counter = 0
    for detections_in_frame in total_detections:
        for detection in detections_in_frame:
            counter += 1
    
    return counter

def calculate_MOTA(model, images, gt_boxes):
    path_to_store_predicted_results = generate_numbered_filepath()
    
    total_detections, total_detections_ids = get_tracking_results(model, images, path_to_store_predicted_results)

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    IDS = 0  # Identity Switches
    GT = 0  # Total number of Ground Truths
    
    for i in range(len(images)):
        curr_detections = total_detections[i]
        curr_detections_ids = total_detections_ids[i]
        prev_detections = total_detections[i-1] if i > 0 else []
        prev_detections_ids = total_detections_ids[i-1] if i > 0 else []
        GT += len(gt_boxes[i])
        gt_boxes_for_frame = gt_boxes[i]

        # Count TP
        for detection in curr_detections:
            iou_max = 0
            for gt_box in gt_boxes_for_frame:
                iou = calculate_iou(detection, gt_box)
                if(iou_max < iou):
                    iou_max = iou
            if(iou_max > 0.5):
                TP += 1

        IDS += count_identity_switches(prev_detections, prev_detections_ids, curr_detections, curr_detections_ids)
    
    FP = count_total_detections(total_detections) - TP
    FN = GT - TP

    MOTA = 1 - ((FN + FP + IDS) / GT)
    
    metrics = {
        'Tracker': TRACKER,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'IDS': IDS,
        'GT': GT,
        'MOTA': MOTA
    }
    print(metrics)
    write_metrics_to_file(metrics, os.path.join(path_to_store_predicted_results, f"metrics.txt"))

    print("Results saved to: ", path_to_store_predicted_results)

    return MOTA

#Main
images = load_images_from_folder(images_dir)
ground_trouth_box_coordinates = load_txt_files_from_folder(labels_dir)

model = YOLO('last.pt')

calculate_MOTA(model, images, ground_trouth_box_coordinates)

