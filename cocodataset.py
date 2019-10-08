import argparse
import better_exceptions
from pathlib import Path
from collections import defaultdict
from itertools import chain
import json
import pandas as pd
from tqdm import tqdm
import cv2

from utils import get_hierarchy, find_contour

def get_args():
    parser = argparse.ArgumentParser(description="This script creates coco format dataset for maskrcnn-benchmark",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--layer", "-l", type=int,default=0,
                                help="target layer; 0 or 1")

    parser.add_argument("--mode", "-m", type=str,default="train",
                        help="target dataset; train or validation")

    parser.add_argument("--img_num", type=int, default=1500,
                        help="max image num for each class")

    args = parser.parse_args()
    return args

class Rect:
    def __init__(self,x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cx = (x1 + x2) / 2
        self.cy = (y1 + y2) / 2
        self.area = (self.x2 - self.x1) * (self.y2 - self.y1)

    def is_inside(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def calc_overlap_rate(rect1, rect2):
    x_left = max(rect1.x1, rect2.x1)
    x_right = min(rect1.x2, rect1.x2)
    y_top = max(rect1.y1, rect2.y1)
    y_bottom = min(rect1.y1, rect2.y1)

    intersection = max(0,x_right - x_left)*max(0, y_bottom-y_top)
    iou = intersection/rect1.area

    return iou

def main():
    args = get_args()
    layer = args.layer
    mode = args.mode
    images = []
    annotations = []

    layer0_class_strings, layer1_class_strings, class_string_to_parent = get_hierarchy()
    target_class_strings = get_hierarchy()[layer]

    target_class_string_to_class_id = {class_string:i+1 for i, class_string in
                                        enumerate(sorted(target_class_strings))}

    parent_class_string = list(set(class_string_to_parent.values()))
    layer0_independent_class_strings = [class_string for class_string in layer0_class_strings if
                                                        class_string not in parent_class_string]

    data_dir = Path("__file__").parent,joinpath('datasets')
    img_dir = data_dir.joinpath(f"{mode}")
    mask_dir = data_dir.joinpath(f"{mode}_mask")
    mask_csv_path = data_dir.joinpath(f"challenge-2019-{mode}-segmentation-masks.csv")
    df = pd.read_csv(str(mask_csv_path))

    output_dir = data_dir.joinpath("coco")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_annotation_dir = output_dir.joinpath("annotations")
    output_annotation_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir = output_dir.joinpath("train2017")
    output_annotation_dir.mkdir(parents=True, exist_ok=True)

    class_string_to_img_ids = defaultdict(list)
    img_id_to_meta = defaultdict(list)

    print("=> parsing {}".format(mask_csv_path.name))

    for i, row in tqdm(df.iterrows(),total = len(df)):
        mask_path, img_id, label_name, _, xp1, xp2, yp1, yp2, _, _ = row.values
        class_string_to_img_ids[label_name].append(im_id)
        img_id_to_meta[img_id].append({"mask_path": mask_path, "class_string": label_name, "bbox": [xp1,xp2,yp1,yp2]})

    # use only args.img_num of images of each class
    target_img_ids = list(
        set(chain.from_iterable(
            [class_string_to_img_ids[class_string][:args.img_num] for class_string in target_class_strings])))
    print("=> use {} images for training".format(len(target_img_ids)))
    bbox_id = 0

    for i, img_id in enumerate(tqdm(target_img_ids)):
        added = False
        img_path = img_dir.joinpath(img_id + ".jpg")
        img = cv2.imread(str(img_path),1)
        h, w, _ = img.shape
        target_rects = []

        #Collect target boxes
        for m in img_id_to_meta[img_id]:
            class_string = m["class_string"]

            #non-target
            if class_string not in target_class_strings:
                continue

            xp1, xp2, yp1, yp2 = m["bbox"]
            target_rects.append(Rect(xp1,yp1,xp2,yp2))

        for m in img_id_to_meta[img_id]:
            class_string = m["class_string"]
            xp1, xp2, yp1, yp2 = m["bbox"]
            x1, x2, y1, y2 = xp1*w , xp2*w, yp1*w, yp2*w

            # For layer1: remove layer0 with no child class
            if layer == 1 and class_string in layer0_independent_class_strings:
                continue

            # for both layer0 and layer1: non-target object is removed if it occludes target bbox over 25%
            if class_string not in target_class_strings:
                curr_rect = Rect(xp1, yp1, xp2, yp2)
                overlap_rate = max([calc_overlap_rate(r,curr_rect) for r in target_rects])
