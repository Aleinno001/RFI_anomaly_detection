from groundingdino.util.inference import load_model, load_image, predict, annotate
from matplotlib import pyplot as plt
from ultralytics import SAM, YOLOE
from torchvision.ops import box_convert
import torch
import numpy as np
import os
import cv2

import utility
from main import RailAnomalyDetector

def show_masks(imag, point_coords=None, box_coords=None, input_labels=None, borders=True,
               savefig=False, save_path=None, save_name=None, show=True):
    plt.clf()
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # plt.figure(figsize=(10, 10))
        plt.imshow(imag)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            # points
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            utility.show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        elif len(scores) == 1:
            plt.title(f"Score: {score:.3f}", fontsize=14)
        plt.axis('off')
        if savefig and save_path is not None and save_name is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"./{save_path}/{str(save_name[:-4])}_{i}.png", bbox_inches='tight', pad_inches=0,
                        dpi=plt.gcf().dpi)
        if show:
            plt.show()
    plt.close()

# main
if __name__ == '__main__':

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    # Inizialization of the models
    mobsamModel = SAM("./models/mobilesam/mobile_sam.pt")
    yoloeModel = YOLOE("./models/yoloe/yoloe-11l-seg.pt")
    samModel = SAM("./models/sam2.1/sam2.1_l.pt")
    model = load_model("configs/grounding_dino/GroundingDINO_SwinT_OGC.py",
                       "models/grounding_dino/groundingdino_swint_ogc.pth", device.type)

    # Setting the keyword that YOLOE will use to detect
    background_search_prompt = ["continuos line","path","railway tracks", "train tracks", "the rails", "near the rails", "railway", "object", "obstacle","rail"]
    foreground_search_prompt = ["the obstacle", "the object", "the anomaly"]

    # Directories of input for the raw images, of output for the processed images, of input for the ground truth
    IMAGE_INPUT_DIR = "./input_images"
    IMAGE_OUTPUT_DIR = "./output_images"
    GROUND_TRUTH_DIR = "./ground_truth"

    # creating the background box
    background_box = np.array([0, 256, 640, 512])

    # Setting statistics
    gd_iou, h_iou, otsu_iou = 0, 0, 0
    gd_dice, h_dice, otsu_dice = 0, 0, 0
    gd_precision, h_precision, otsu_precision = 0, 0, 0
    gd_recall, h_recall, otsu_recall = 0, 0, 0

    n_frames = 0
    skipped = 0

    # Extract frames
    frame_names = [
        p for p in os.listdir(IMAGE_INPUT_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # Sorting frames
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    #TODO Generata la cosa con YOLOE poi devo rigenerare una maschera dei binari interpolata fatta densa piena a prescindere dagli ostacoli, poi con le maschere di mobileSAM (tolte quelle piu piccole di un tot) ne verifico la intersezopne con la maschera di prima e le salvo
    # IMAGE ANALYSIS
    try:
        for p in frame_names:


            print('----------------------')
            image = cv2.imread(os.path.join(IMAGE_INPUT_DIR, p))
            TEXT_PROMPT = "object ."
            BOX_TRESHOLD = 0.35
            TEXT_TRESHOLD = 0.25
            ground_truth_prefix = "gt_"
            print('---------------------------')
            image_source, gd_image = load_image(IMAGE_INPUT_DIR + '/' + p)
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            gd_boxes, logits, phrases = predict(
                model=model,
                image=gd_image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device=device.type,
            )
            h, w, _ = image_source.shape
            gd_boxes = gd_boxes * torch.Tensor([w, h, w, h])
            gd_boxes = box_convert(boxes=gd_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            for box in gd_boxes:
                if utility.is_contained(box, background_box, 1):

                    # salvataggio dell'immagine
                    plt.imshow(image)
                    utility.show_box(box, plt.gca())
                    save_path = "debug/gd2/"
                    save_name = "test"
                    savefig = True
                    show = True
                    if savefig and save_path is not None and save_name is not None:
                        os.makedirs(save_path, exist_ok=True)
                        plt.savefig(f"./{save_path}/{str(save_name[:-4])}_{0}.png", bbox_inches='tight', pad_inches=0,
                                    dpi=plt.gcf().dpi)
                    if show:
                        plt.show()
                    plt.close()

            print("Grounding Dino - frame: ")
            '''
            ground_truth = cv2.imread(
                os.path.join(GROUND_TRUTH_DIR, ground_truth_prefix + os.path.splitext(p)[0] + ".png"),
                cv2.IMREAD_GRAYSCALE)

            if cv2.countNonZero(ground_truth) < 170:
                print("oggetto troppo piccolo")
                skipped += 1
                continue

            if n_frames % 20 == 0:
                print("frame saltati: ", skipped)
                '''

            name = str(p)
            print("Analyzing image:", name)
            print()

            image = cv2.imread(os.path.join(IMAGE_INPUT_DIR, p))
            # Setting YOLOE for the detection of the rails (first prompt)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            print("First YOLOE detection")
            yoloeModel.set_classes(background_search_prompt, yoloeModel.get_text_pe(background_search_prompt))
            results = yoloeModel.predict(image)

            results[0].show()

            # Retrieving boxes
            boxes = results[0].boxes

            torch.cuda.empty_cache()

            coordinates = []
            i = 0
            for box in boxes:
                coordinates.append(box.xyxy[0])  # get box coordinates in (left, top, right, bottom) format
                i = i + 1

            for coord in coordinates:
                #results = samModel.predict(image, bboxes=[int(coord[0].item()), int(coord[1].item()),
                #                                          int(coord[2].item()), int(coord[3].item())])
                #results[0].show()

                # Copying image to crop inside the bottom half box for faster detection
                img_copy = image.copy()
                img_copy = img_copy[int(coord[1].item()):int(coord[3].item()),
                           int(coord[0].item()):int(coord[2].item())]

                print("Second SAMMobile detection")
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)

                results = mobsamModel.predict(img_copy)

                results[0].show()

            # -----------------------------------------------------
    except(KeyboardInterrupt, SystemExit):
        print("Exiting...")
        print('iou', gd_iou, h_iou, otsu_iou)
        print('dice', gd_dice, h_dice, otsu_dice)
        print('precision', gd_precision, h_precision, otsu_precision)
        print('recall', gd_recall, h_recall, otsu_recall)
        print('n_frames ', n_frames)
        print('skipped ', skipped)