import datetime
import glob
import json
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.draw

import model as modellib
import utils
import visualize
from config import Config

# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("Segmentation"):
    ROOT_DIR = "Mask_RCNN"

sys.path.append(ROOT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "result")

# print(os.getcwd())

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class_names = []
class_df = pd.read_csv(
    "challenge-2019-classes-description-segmentable.csv", header=None)
for i in range(300):
    class_names.append(class_df[1][i])
	
'''
******************************************************************************
                                 Configuration
******************************************************************************
'''


class OpenImageConfig(Config):

    # Give the configuration a recognizable name
    NAME = "openimage"
    
    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 4

    # Using GPU with 12GB memory, it can fit two images
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 300  # Background + Total Class

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 25

    # Skip detections with < 90% confidence
    #DETECTION_MIN_CONFIDENCE = 0.9


'''
******************************************************************************
                                    Dataset
******************************************************************************
'''


class OpenImageDataset(utils.Dataset):

    def load_openimage(self, dataset_dir, subset):

        # Add classes
        cls_df = pd.read_csv(
            "challenge-2019-classes-description-segmentable.csv", header=None)
        # print(cls_df[1][0])

        for i in range(300):
            self.add_class("openimage", i + 1, cls_df[1][i])
            # print(cls_df[1][i])

        # Train or validation dataset?
        assert subset in ["train", "validation"]
        dataset_dir = os.path.join(dataset_dir, subset)

        df = pd.read_csv("challenge-2019-" + subset +
                         "-segmentation-masks.csv")

        # Getting Image Ids
        img_ids = []
#         for i,filepath in enumerate(sorted(glob.glob('{}/*'.format(dataset_dir)))):
#             idx = os.path.basename(filepath).split(".")[0]
#             img_ids.append(idx)

        for i, temp_img_id in enumerate(df["ImageID"]):
            img_path = temp_img_id + ".jpg"
            image_path = os.path.join(dataset_dir, img_path)
            if os.path.exists(image_path):
                img_ids.append(temp_img_id)


#         for i, filepath in enumerate(sorted(glob.glob('{}/*'.format(dataset_dir)))):
#             idx = os.path.basename(filepath).split(".")[0]
#             for j, ids in enumerate(df["ImageID"]):
#                 if idx == ids:
#                     img_ids.append(ids)

        # Adding of Image
        for i, img_name in enumerate(img_ids):
            file_path = img_name + ".jpg"
            image_path = os.path.join(dataset_dir, file_path)

            self.add_image(
                "openimage",
                image_id=img_name,
                path=image_path
            )

    def load_mask(self, image_id):

        info = self.image_info[image_id]

        subset = os.path.split(os.path.split(info['path'])[0])
        mask_dir = os.path.join(subset[0], (subset[1] + "_mask"))

        cls_df = pd.read_csv(
            "challenge-2019-classes-description-segmentable.csv", header=None)

        class_dict = {}
        for i in range(300):
            class_dict[cls_df[0][i]] = i + 1

        real_id = info["id"]
        df1 = pd.read_csv("challenge-2019-" +
                          subset[1] + "-segmentation-masks.csv")
        df1.set_index("ImageID", inplace=True)

        id_df = df1.loc[real_id]

        mask = []
        temp_class_id = []

        if id_df.size == 9:
            temp_class_id.append(id_df.LabelName)
            m = skimage.io.imread(os.path.join(
                mask_dir, id_df.MaskPath)).astype(np.bool)
            mask.append(m)

        else:
            for j, label in enumerate(id_df.LabelName):
                temp_class_id.append(label)

            for i, m_path in enumerate(id_df.MaskPath):
                # m = skimage.io.imread(os.path.join(mask_dir,m_path))
                # print(m_path)
                m = skimage.io.imread(os.path.join(
                    mask_dir, m_path)).astype(np.bool)
                mask.append(m)

        mask = np.stack(mask, axis=-1)
        class_id = np.empty([mask.shape[-1]], dtype=np.int32)

        for i in range(len(temp_class_id)):
            class_id[i] = class_dict[temp_class_id[i]]

        # print(class_id,mask.shape)

        return mask, class_id

    def image_reference(self, image_id):
        """Return the path of the image"""

        info = self.image_info[image_id]
        if info["source"] == "openimage":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


# print(os.path.join(os.getcwd(), "Train"))
# OpenImageDataset().load_openimage(os.getcwd(), "Valid")
# OpenImageDataset().load_mask("0af4326e3db0c11a", "validation")

"""
******************************************************************************
                                    Training
******************************************************************************
"""


def train(model, dataset_dir, subset):
    """Train the model"""
    # Training dataset
    dataset_train = OpenImageDataset()
    dataset_train.load_openimage(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = OpenImageDataset()
    dataset_val.load_openimage(dataset_dir, "validation")
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    # augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training Stage - 1
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')
    # augmentation = augmentation)

    # Training Stage - 2
    # Finetuning layers from ResNet stage 4 and up
    model.train(dataset_train, datset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers="4+")
    # augmentation = augmentation)

    # Training Stage - 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all')

    # augmentation = augmentation)


'''
********************************************************************************
RLE Encoding
********************************************************************************
'''


def rle_encode(mask):
    '''Encodes a mask in Run Length Encoding (RLE).
    Return as string of space-seperated values.
    '''

    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indices of transition point(where gradient !=0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to length
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    '''Decodes an RLE encoded list of space seperated numbers and
    Return binary mask
    '''
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(
            shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instances mask to submission format."
    assert mask.ndim == 3, 'Mask must be [H,W,count]'
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    print(order.shape, mask.shape)
    #mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    # lines = []
    # for o in order:
    #     m = np.where(mask == o, 1, 0)
    #     # Skip if empty
    #     if m.sum() == 0.0:
    #         continue
    #     rle = rle_encode(m)
    #     lines.append("{}, {}".format(image_id, rle))
    # return "\n".join(lines)


def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))
    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    #submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, "submit")
    #os.makedirs(submit_dir)

    # Read dataset
    img_ids = []
    dataset_dir = os.path.join(dataset_dir, subset)
    image_file = os.listdir(dataset_dir)
    #submission = []
    for img in image_file:
        if not img.startswith('.'):
            img_file = os.path.join(dataset_dir, img)
            image = skimage.io.imread(img_file)
            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # Detect object
			
            r = model.detect([image])[0]
            # Encode image to RLE. Returns a string of multiple lines
            source_id = img.split(".")[0]
            #rle = mask_to_rle(source_id, r["masks"], r["scores"])
            #submission.append(rle)
            # Save image with masks
            visualize.display_instances(
                image, r['rois'], r['masks'], r['class_ids'],
                class_names, r['scores'],
                #show_bbox=False, show_mask=False,
                title="Predictions")
            plt.savefig("{}/{}.png".format(submit_dir, source_id))


		
    # Save to csv file
#     submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
#     file_path = os.path.join(submit_dir, "submit.csv")
#     with open(file_path, "w") as f:
#         f.write(submission)
#     print("Saved to ", submit_dir)


'''
**********************************************************************************
Command Line
**********************************************************************************
'''
if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Mask R-CNN for open image segmentation")

    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")

    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')

    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")

    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = OpenImageConfig()
    else:
        class InferenceConfig(OpenImageConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

        # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
