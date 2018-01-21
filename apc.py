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
import scipy.io
from PIL import Image
import skimage.io


from config import Config
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
WPIF_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_apc.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

# WPIF for workpiece in factory
class APCConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "apc"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 4

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    IMAGE_TYPE = 'RGBHHA'

    # Number of classes (including background)
    NUM_CLASSES = 39 + 1  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class APCDataset(utils.Dataset):
    class_names = ['bg']

    def load_apc(self, dataset_dir):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the dataset.
        """
        # get all scenes path
        scenes = scipy.io.loadmat(os.path.join(dataset_dir, 'scenes.mat'))['scenes']
        scenes_list = []
        for i in range(scenes.shape[1]):
            scenes_list.append(os.path.join(dataset_dir, scenes[0, i][0]))

        # get all class names
        objects = scipy.io.loadmat(os.path.join(dataset_dir, 'objects.mat'))['objects']
        for i in range(objects.shape[0]):
            self.class_names.append(objects[i][0][0][0][0][0])
            self.add_class("APC", i+1, self.class_names[i+1])

        image_id = 0
        for scene in scenes_list:
            for example in os.listdir(scene):
                if example[-4:] == '.mat':
                    label_file = os.path.join(scene, example)
                    label = scipy.io.loadmat(label_file)['label']
                    example_name = example[:12]
                    image_id = image_id + 1
                    image_path = os.path.join(scene, example_name + '.color.png')
                    hha_path = os.path.join(scene, example_name + '.hha.png')
                    im = Image.open(image_path)
                    width, height = im.size
                    annotations = []
                    for i in range(label.shape[1]):
                        annotation = {}
                        annotation['class'] = label[0, i][0, 0][0][0]
                        annotation['pose'] = label[0, i][0, 0][1]
                        annotation['mask'] = label[0, i][0, 0][2]
                        # bbox = [y1, x1, y2, x2] and 1 <= y <= height
                        annotation['bbox'] = label[0, i][0, 0][3][0] - 1
                        annotations.append(annotation)
                    self.add_image('APC', image_id=image_id,
                                   path=image_path,
                                   width=width, height=height,
                                   annotations=annotations,
                                   hha_path=hha_path)

    def load_hha(self, image_id):
        """
        Load the specified HHA image
        :param image_id:
        :return: a [H, W, 3] Numpy array
        """
        # Load HHA image
        hha = skimage.io.imread(self.image_info[image_id]['hha_path'])
        assert hha.ndim == 3
        return hha

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
        if image_info["source"] != "APC":
            return super(self.__class__).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = image_info["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.class_names.index(annotation['class'])
            if class_id:
                mask = annotation['mask']
                bbox = annotation['bbox']
                y1, x1, y2, x2 = bbox
                if y1 == 0:
                    y2 = y2 + 1
                else:
                    y1 = y1 -1
                if x1 == 0:
                    x2 = x2 + 1
                else:
                    x1 = x1 - 1
                full_mask = np.zeros([image_info['height'], image_info['width']], dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = mask
                instance_masks.append(full_mask)
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
        if info["source"] == "APC":
            return "Amazon Picking Challenge Database"
        else:
            super(self.__class__).image_reference(self, image_id)

############################################################
#  WPIF Evaluation
############################################################
def evaluate_apc(model, config, dataset, eval_type="bbox", limit=0):
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
        description='Train Mask R-CNN on APC dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on APC dataset")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/wpif/",
                        help='Directory of the APC dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'apc'")
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
        config = APCConfig()
    else:
        class InferenceConfig(APCConfig):
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
    if args.model.lower() == "apc":
        model_path = WPIF_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    elif args.model.lower() == "none":
        model_path = ""
    else:
        model_path = args.model

    # Load weights
    if len(model_path) > 1:
        print("Loading weights ", model_path)
        # exclude some weight if we have only two class(1bg + 1) and load weights from coco
        #model.load_weights(model_path, by_name=True,
        #                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = APCDataset()
        dataset_train.load_apc(args.dataset)
        dataset_train.prepare()

        # This training schedule is an example. Update to fit your needs.

        # Training - Stage 1
        # Adjust epochs and layers as needed
        print("Training network heads")
        model.train(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Training Resnet layer 4+")
        model.train(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=50,
                    layers='4+')

        # Training - Stage 3
        # Finetune layers from ResNet stage 3 and up
        print("Training Resnet layer 3+")
        model.train(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=100,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = APCDataset()
        dataset_val.load_apc(args.dataset)
        dataset_val.prepare()
        print("Running APC evaluation on {} images.".format(args.limit))
        evaluate_apc(model, config, dataset_val, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
