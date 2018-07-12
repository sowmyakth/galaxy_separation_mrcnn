import os
import sys
import numpy as np
# import pickle
# import dill
# import random
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['figure.subplot.wspace'] = 0.05
matplotlib.rcParams['figure.subplot.hspace'] = 0.05
import matplotlib.pyplot as plt
import h5py

# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/NN_blend/Mask_RCNN'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# path to images
DATA_PATH = '/scratch/users/sowmyak/lavender'

CODE_PATH = '/home/users/sowmyak/NN_blend/scripts'
# MODEL_PATH = '/scratch/users/sowmyak/lavender/logs/blend_final_again20180608T2004/mask_rcnn_blend_final_again_0080.h5'
MODEL_PATH = '/scratch/users/sowmyak/lavender/logs/blend_final_again20180608T2004/mask_rcnn_blend_final_again_0100.h5'
#MODEL_PATH =None
sys.path.append(CODE_PATH)
import display
import train as train
#import train_final_again as train
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from train import InputConfig
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
# from mrcnn.model import log

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class InferenceConfig(InputConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# image, image_meta, class_ids, bbox, mask, debl_image, mult_image
def overall_perfomance(model, dataset_val, inference_config):
    image_ids = dataset_val.image_ids# np.random.choice(dataset_val.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask, _, _ =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        # molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        if (gt_mask.shape[-1] == 0) or (r['masks'].shape[-1] == 0):
            print(image_id, " skipped")
            continue
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    print("mAP: ", np.mean(APs))
    print("prec: ", np.mean(precisions))
    print("recall: ", np.mean(recalls))
    print("overlap: ", np.mean(overlaps))


def evaluate(dataset_val):
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    inference_config = InferenceConfig()
    # added for testing
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    #model_path = model.find_last()[1]
    if MODEL_PATH:
        model_path = MODEL_PATH
    else:
        model_path = model.find_last()[1]
    # model_path = '/home/users/sowmyak/NN_blend/Mask_RCNN/logs/blend420180607T1217/mask_rcnn_blend4_0009.h5'
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    #overall_perfomance(model, dataset_val, inference_config)
    #fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    #visualize.plot_precision_recall(AP, precisions, recalls, ax=ax)
    #fig.savefig('roc_curve')
    #for i in range(5):
    for i in [267]:
        test_rand_image(model, dataset_val, inference_config, i)


def test_rand_image(model, dataset_val, inference_config, image_id):
    #image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_debl, gt_mult =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    print("mul_image", gt_mult.shape)
    #log("original_image", original_image)
    #log("image_meta", image_meta)
    #log("gt_class_id", gt_class_id)
    #log("gt_bbox", gt_bbox)
    #log("gt_mask", gt_mask)
    results = model.detect([original_image], verbose=1)
    r = results[0]
    print("maax mask", np.max(r['masks']))
    if (gt_mask.shape[-1] == 0) or (r['masks'].shape[-1] == 0):
        print(image_id, " skipped")
        return
    #fig, axarr = plt.subplots(1, 2, figsize=(12, 10))
    #visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                            dataset_val.class_names, ax=axarr[0], limits=[20, 100])
    #plt.savefig('disp1')
    #visualize.display_instances(original_image, r['rois'], r['masks'],
    #                            r['class_ids'], dataset_val.class_names,
    #                            r['scores'], ax=axarr[1], limits=[20, 100])
    #fig.savefig('true_output_' + str(image_id))
    #fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # axarr[0].imshow(original_image.astype(np.uint8()), interpolation='none')
    #axarr[0].imshow(original_image, interpolation='none')
    #axarr[0].set_xlim([20, 100])
    #axarr[0].set_ylim([20, 100])
    #axarr[0].axis('off')
    visualize.display_debl_input(original_image, r['masks'], r['debl'], gt_mult[:,:,0],
                                 r['class_ids'], dataset_val.class_names,
                                     limit=5)
    #visualize.display_differences(original_image, gt_bbox, gt_class_id,
    #                              gt_mask, r['rois'], r['class_ids'],
    #                              r['scores'], r['masks'],
    #                              dataset_val.class_names, ax=axarr[1],
    #                              limits=[20, 108], mask_alpha=0.1)
    #fig.savefig('debl_output_' + str(image_id))
    plt.savefig('debl_output_' + str(image_id))


def main():
    dataset_val = train.ShapesDataset()
    dataset_val.load_data(training=False)
    dataset_val.prepare()
    config = train.InputConfig()
    config.display()
    # Validation dataset
    image_ids = np.random.choice(dataset_val.image_ids, 5)
    image_ids = [267]
    for image_id in image_ids:
        image = dataset_val.load_image(image_id)
        mask, class_ids = dataset_val.load_mask(image_id)
        debl = dataset_val.load_debl_image(image_id)
        mult_image = dataset_val.load_mult_image(image_id)[:, :, 0]
        print(mult_image.shape)
        # visualize.display_sep_masks(image, mask, class_ids, dataset_val.class_names,
        #                            limit=2)
        visualize.display_debl_input(image, mask, debl, mult_image,
                                     class_ids, dataset_val.class_names,
                                     limit=6)
        plt.savefig("input_debl_" +str(image_id))

    evaluate(dataset_val)


if __name__ == "__main__":
    main()
