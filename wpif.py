"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np


from config import Config
import utils
import model as modellib
from labelFile import LabelFile

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
WPIF_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_wpif.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

# WPIF for workpiece in factory
class WPIFConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "wpif"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class WPIFDataset(utils.Dataset):
    class_names = ['bg', 'pad']

    def load_wpif(self, dataset_dir):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the dataset.
        """

        # get all labeled json file under dataset
        examples_list = []
        for example in os.listdir(dataset_dir):
            if example[-5:] == '.json':
                examples_list.append(os.path.join(dataset_dir, example))

        # Add classes
        self.add_class("WPIF", self.class_names.index('pad'), 'pad')

        for example_name in examples_list:
            label_file = LabelFile()
            label_file.load(example_name, dataset_dir)
            # Add images
            image_path = label_file.imagePath
            image_name = os.path.split(image_path)[-1]
            image_id = int(image_name.split('.')[0][6:12])
            width = label_file.width
            height = label_file.height
            annotations = []
            for label, points, line_color, fill_color in label_file.shapes:
                annotations.append([label, points])
            self.add_image("WPIF", image_id=image_id,
                           path=image_path,
                           width=width, height=height,
                           annotations=annotations)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a WPIF image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "WPIF":
            return super(self.__class__).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = image_info["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.class_names.index(annotation[0])
            if class_id:
                m = utils.extract_mask_from_polygon(image_info["height"],
                                                    image_info["width"],
                                                    annotation[1])
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "WPIF":
            return "Workpiece in factories database"
        else:
            super(self.__class__).image_reference(self, image_id)

############################################################
#  WPIF Evaluation
############################################################
def evaluate_wpif(model, config, dataset, eval_type="bbox", limit=0):
    """
    Evaluation on WPIF dataset, using VOC-Style mAP # IoU=0.5 for bbox
    @TODO: add segment evaluation
    :param model:
    :param config:
    :param dataset:
    :param eval_type:
    :param limit:
    :return:
    """
    image_ids = dataset.image_ids
    if limit:
        image_ids = np.random.choice(dataset.image_ids, limit)

    t_prediction = 0
    t_start = time.time()
    APs = []

    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)

        # Run object detection
        t = time.time()
        results = model.detect([image], verbose=0)
        r = results[0]
        t_prediction += (time.time() - t)

        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id,
                             r["rois"], r["class_ids"], r["scores"])
        APs.append(AP)

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    print("mAP: ", np.mean(APs))


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on my WPIF.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on my WPIF")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/wpif/",
                        help='Directory of the my WPIF dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'wpif'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (defaults=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = WPIFConfig()
    else:
        class InferenceConfig(WPIFConfig):
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
    if args.model.lower() == "wpif":
        model_path = WPIF_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = WPIFDataset()
        dataset_train.load_wpif(args.dataset)
        dataset_train.prepare()

        # This training schedule is an example. Update to fit your needs.

        # Training - Stage 1
        # Adjust epochs and layers as needed
        print("Training network heads")
        model.train(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Training Resnet layer 4+")
        model.train(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100,
                    layers='4+')

        # Training - Stage 3
        # Finetune layers from ResNet stage 3 and up
        print("Training Resnet layer 3+")
        model.train(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=200,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = WPIFDataset()
        dataset_val.load_wpif(args.dataset)
        dataset_val.prepare()
        print("Running WPIF evaluation on {} images.".format(args.limit))
        evaluate_wpif(model, config, dataset_val, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
