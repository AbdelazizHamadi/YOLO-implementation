"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import json


class PancakeDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_path, S=7, B=2, C=1, transform=None, ):
        self.annotation_path = annotation_path

        # Opening JSON file
        f = open(self.annotation_path)
        # returns JSON object as a dictionary
        self.dataset = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        self.imgFiles = self.get_images_paths()
        self.boxes = self.get_boxes()

    def get_images_paths(self):
        self.imgFiles = []

        for i in range(len(self.dataset['images'])):
            self.imgFiles.append(os.path.join(self.img_dir, self.dataset['images'][i]['file_name']))

        return self.imgFiles

    def convertbox2yolo(self, dims, box):
        dw = 1. / dims[0]
        dh = 1. / dims[1]
        # box x, y , w, h
        x = ((box[0] + box[2]) + box[0]) / 2.0
        y = ((box[1] + box[3]) + box[1]) / 2.0
        w = box[2]
        h = box[3]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh

        return x, y, w, h

    def yolobbox2bbox(self, box, w=1920, h=1280):
        x, y = box[1], box[2]
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        return torch.tensor([x1, y1, x2, y2])


    def get_boxes(self):

        self.boxes = []

        for i in range(len(self.dataset['images'])):
            image_id = self.dataset['images'][i]['id']
            file_name = self.dataset['images'][i]['file_name']
            bbox = []
            for j in range(len(self.dataset['annotations'])):
                if self.dataset['annotations'][j]['image_id'] == image_id:
                    bbox.append([file_name, int(self.dataset['annotations'][j]['category_id']),
                                 self.dataset['annotations'][j]['bbox'],
                                 self.dataset['annotations'][j]['width'], self.dataset['annotations'][j]['height']])

            self.boxes.append(bbox)

        return self.boxes

    def get_image_boxes(self, image_boxes_list, box_format="yolo"):

        yolo_boxes = []
        normal_boxes = []

        for info in image_boxes_list:

            # info : [file_name, class, box[...], width, height]
            if box_format == "yolo":
                x, y, w, h = self.convertbox2yolo(dims=(info[3], info[4]), box=info[2])
                yolo_boxes.append([info[1], x, y, w, h])

            else:
                xmin, ymin, w, h = info[2][0], info[2][1], info[2][2], info[2][3]
                normal_boxes.append([info[1], xmin, ymin, w, h])

                pass

        return torch.tensor(yolo_boxes), torch.tensor(normal_boxes)

    def __getitem__(self, index):

        image = Image.open(self.imgFiles[index]).convert("RGB")
        image_boxes, _ = self.get_image_boxes(self.boxes[index], box_format="yolo")

        if self.transform:
            # image = self.transform(image)
            image, image_boxes = self.transform(image, image_boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in image_boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
                Calculating the width and height of cell of bounding box,
                relative to the cell is done by the following, with
                width as the example:

                width_pixels = (width*self.image_width)
                cell_pixels = (self.image_width)

                Then to find the width relative to the cell is simply:
                width_pixels/cell_pixels, simplification leads to the
                formulas below.
                """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 1] == 0:
                # Set that there exists an object
                label_matrix[i, j, 1] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 2:6] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

    pass
