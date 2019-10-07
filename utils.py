import argparse
import better_exceptions
from pathlib import Path
import json
from collections import defaultdict
import cv2
import base64
import pandas as pd
import numpy as np
from pycocotools import _mask as coco_mask
import zlib
import typing as t

def get_class_mapping():
    csv_path = Path(__file__).parent.joinpath("datasets", "classes-segmentation.txt")
    df = pd.read_csv(str(csv_path), header=None, names=["class_string"])
    class_string_to_id = dict(zip(df.class_string, df.index))
    class_id_to_string = dict(zip(df.index, df.class_string))
    return class_string_to_id, class_id_to_string

def get_string_to_name():
    csv_path = Path(__file__).parent.joinpath("datasets", "challenge-2019-classes-description-segmentable.csv")
    df = pd.read_csv(str(csv_path), header=None, names=["class_string", "class_name"])
    class_string_to_name = dict(zip(df.class_string,df.class_name))
    return class_string_to_name

def get_layer_names():
    
