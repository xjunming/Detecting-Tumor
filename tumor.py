"""
Mask R-CNN

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python tumor.py train --dataset=/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python tumor.py train --dataset=./dataset/ --subset=train --weights=./model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

    # Resume training a model that you had trained earlier
    python tumor.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python tumor.py detect --dataset=/dataset --subset=train --weights=<last or /path/to/weights.h5>

    # detect the dicom
    python tumor.py detect --dataset=./dataset/ --subset=val --weights=./model/dataset20190417T1119/mask_rcnn_dataset_0040.h5
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os,sys,json,datetime,re
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import pandas as pd
import cv2 as cv

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
TUMOR_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_tumor.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/tumor/")


def _find_data_files(search_path):
    lstFiles = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(search_path):
        for filename in fileList:
            if ".dcm" in filename.lower() in filename.lower():  # check whether the file's DICOM
                lstFiles.append(os.path.join(dirName, filename))
    # print(lstFiles)
    return lstFiles

VAL_IMAGE_IDS = _find_data_files('./dataset/val/')

############################################################
#  Configurations
############################################################

class tumorConfig(Config):
    """Configuration for training on the tumor segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "dataset"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + Positive + Negative

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between tumor and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class tumorInferenceConfig(tumorConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

def read_csvfile(search_path):
  lstFiles = []  # create an empty list
  for dirName, subdirList, fileList in os.walk(search_path):
    for filename in fileList:
      if ".csv" in filename.lower() in filename.lower():  # check whether the file's DICOM
        lstFiles.append(os.path.join(dirName, filename))
  csv_file_name = lstFiles[0]
  # print(csv_file_name)
  df = pd.read_csv(csv_file_name,encoding='GB2312',header=0,delimiter=',', index_col=2,names=['gender', 'age','PosOrNeg'])
  df = df.drop(['gender', 'age'], axis=1)
  df.replace(to_replace='-', value=0, inplace=True)
  df.replace(to_replace='+', value=1, inplace=True)
  return df

df = read_csvfile('./dataset')
############################################################
#  Dataset
############################################################

class tumorDataset(utils.Dataset):

    def _find_data_files(self, search_path):
        lstFiles = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(search_path):
            for filename in fileList:
                if ".dcm" in filename.lower() in filename.lower():  # check whether the file's DICOM
                    lstFiles.append(os.path.join(dirName, filename))
        # print(lstFiles)
        return lstFiles

    def load_tumor(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset tumor, and the class tumor
        self.add_class("tumor", 1, "Positive")
        self.add_class("tumor", 2, "Negative")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val"]
        subset_dir = "train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            print('loading data from ',dataset_dir.replace('train','val'))
            image_ids = self._find_data_files(dataset_dir.replace('train','val'))
        else:
            # Get image ids from directory names
            image_ids = self._find_data_files(dataset_dir)
            if subset == "train":
                image_ids = list(set(image_ids))
                # print(image_ids)

        # Add images
        print(df)
        for image_id in image_ids:
            if subset == "train":
                patientID = int(re.findall(r"train\\(.+?)\\",image_id)[0])
            if subset == "val":
                patientID = int(re.findall(r"val\\(.+?)\\",image_id)[0])
            # print(re.findall(r"val\\(.+?)\\",image_id))
            if df.PosOrNeg[patientID]:
                self.add_image(
                    "tumor",
                    image_id=image_ids.index(image_id),
                    path=image_id)
            else:
                self.add_image(
                    "tumor",
                    image_id=image_ids.index(image_id),
                    path=image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # print(info)
        # Get mask directory from image path
        # mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])))

        # Read mask files from .png image
        mask = []
        # for f in next(os.walk(mask_dir))[2]:
        #     if f.endswith(".png"):
        # print(info['path'].replace('.dcm','_mask.png'))
        m = skimage.io.imread(info['path'].replace('.dcm','_mask.png')).astype(np.bool)
        mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance.
        # print(info['source'])
        if info['source']=='Negative':
            # print(np.ones([mask.shape[-1]], dtype=np.int32)*2)
            return mask, np.ones([mask.shape[-1]], dtype=np.int32)*2
        else:
            return mask, np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tumor":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = tumorDataset()
    dataset_train.load_tumor(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = tumorDataset()
    dataset_val.load_tumor(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################
# 预测的文件一定要在./dataset/val文件夹下
def getPatientID(path_list):
    return [int(re.findall(r"val\\(.+?)\\",path)[0]) for path in path_list]

def initPatientType(patientID):
  patient_dict = {}
  for ID in patientID:
    patient_dict[ID] = []
  return patient_dict

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = tumorDataset()
    dataset.load_tumor(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    patient_dict = initPatientType(getPatientID([dataset.image_info[image_id]["path"] for image_id in dataset.image_ids]))
    scores_dict = initPatientType(getPatientID([dataset.image_info[image_id]["path"] for image_id in dataset.image_ids]))
    PositiveOrnegative = []
    for image_id in dataset.image_ids:
        print(image_id,'/',len(dataset.image_ids))
        # Load image and run detection
        image = dataset.load_image(image_id)
        # print('Max',np.amax(image),'shape',image.shape)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        dcm_path = dataset.image_info[image_id]["path"]
        if os.path.exists(dcm_path):
            os.remove(dcm_path)
        if r["masks"].any():
            # print('Max',np.amax(r["masks"][0]),'shape',r["masks"][0].shape)
            pass
        else:
            cv.imwrite(dcm_path.replace('.dcm', '_mask.png'), np.zeros([512,512,1]))
            # print('没有检测到肿瘤')
            continue
        if r["scores"][0]<0.68:
            # print('概率小于0.68')
            cv.imwrite(dcm_path.replace('.dcm', '_mask.png'), np.zeros([512, 512, 1]))
            continue
        # print(r['rois'], r['class_ids'],
        #     dataset.class_names, r['scores'], r['scores'][0])
        # Encode image to RLE. Returns a string of multiple lines
        # source_id = dataset.image_info[image_id]["id"]
        # 检测图像的路径
        # print(dataset.image_info[image_id]["path"])
        dcm_path = dataset.image_info[image_id]["path"]
        # 输出预测的掩模图像
        mask_array = np.array(r["masks"][:, :, 0])*255
        cv.imwrite(dcm_path.replace('.dcm','_mask.png'), mask_array)
        # 每张图片预测出来的阳性或阴性
        ID = getPatientID(list([dataset.image_info[image_id]["path"]]))[0]
        patient_dict[ID].append(r['class_ids'][0])
        scores_dict[ID].append(r['scores'][0])

        # rle = mask_to_rle(source_id, r["masks"], r["scores"])
        # submission.append(rle)
        # # Save image with masks
        # # 数据集的类别 dataset.class_names
        # visualize.display_instances(
        #     image, r['rois'], r['masks'], r['class_ids'],
        #     ['BG','Positive(+)','Negative(-)'], r['scores'],
        #     show_bbox=True, show_mask=True,
        #     title="Predictions")
        # # 保存可视化图形，用于工程或论文
        # plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    # print('patient_dict: ',patient_dict)
    df = pd.DataFrame(index=[i for i in patient_dict], columns=['Positive', 'Negative'])
    for ID in patient_dict:
        if 1 in patient_dict[ID]:
            df.Positive[ID] = np.mean([s for c, s in zip(patient_dict[ID], scores_dict[ID]) if c == 1])
        else:
            df.Positive[ID] = 0
        if 2 in patient_dict[ID]:
            df.Negative[ID] = np.mean([s for c, s in zip(patient_dict[ID], scores_dict[ID]) if c == 2])
        else:
            df.Negative[ID] = 0
    # 替换成提交验证数据的格式
    # df.replace(to_replace=1, value='+', inplace=True)
    # df.replace(to_replace=2, value='-', inplace=True)
    df.to_csv(os.path.join(submit_dir, "pre_classification.csv"))

    # submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    # file_path = os.path.join(submit_dir, "submit.csv")
    # with open(file_path, "w") as f:
    #     f.write(submission)
    # print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
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
        config = tumorConfig()
    else:
        config = tumorInferenceConfig()
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
        weights_path = TUMOR_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
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
              "Use 'train' or 'detect'".format(args.command))
