from ultralytics import YOLOE, SAM
import torch
import numpy as np
import os
import cv2
from PIL import Image

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


    #Inizialization of the models
    yoloeModel = YOLOE("./models/yoloe/yoloe-11l-seg.pt")
    samModel = SAM("./models/sam2.1/sam2.1_t.pt")

    #Setting the keyword that YOLOE will use to detect
    background_search_prompt =["object","obstacle","train tracks","the rails","near the rails"]
    foreground_search_prompt = ["the obstacle","the object","the anomaly"]

    #Directories of input for the raw images, of output for the processed images, of input for the ground truth
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

    # IMAGE ANALYSIS
    try:
        for p in frame_names:

            print('----------------------')
            image = cv2.imread(os.path.join(IMAGE_INPUT_DIR, p))
            ground_truth_prefix = "gt_"
            ground_truth = cv2.imread(os.path.join(GROUND_TRUTH_DIR, ground_truth_prefix+os.path.splitext(p)[0]+".png"),
                                      cv2.IMREAD_GRAYSCALE)

            if cv2.countNonZero(ground_truth) < 170:
                print("oggetto troppo piccolo")
                skipped += 1
                continue

            if n_frames % 20 == 0:
                print("frame saltati: ", skipped)

            name = str(p)
            print("Analyzing image:", name)
            print()

            #Copying image to crop inside the bottom half box for faster detection
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #img_copy = image.copy()
            #img_copy = img_copy[background_box[1]:background_box[3], background_box[0]:background_box[2]]

            #Setting YOLOE for the detection of the rails (first prompt)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print("First YOLOE detection")
            yoloeModel.set_classes(background_search_prompt, yoloeModel.get_text_pe(background_search_prompt))
            results = yoloeModel.predict(image)

            results[0].show()


            #Retrieving boxes
            boxes = results[0].boxes
            coordinates = []
            i=0
            print(boxes[0].xyxy[0])
            for box in boxes:
                coordinates.append(box.xyxy[0])  # get box coordinates in (left, top, right, bottom) format
                i = i+1

            # Copying image to crop inside the bottom half box for faster detection
            img_copy = image.copy()
            img_copy = img_copy[int(coordinates[0][1].item()):int(coordinates[0][3].item()), int(coordinates[0][0].item()):int(coordinates[0][2].item())]
            # Setting YOLOE for the detection of the anomaly (second prompt)
            print("Second YOLOE detection")
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            yoloeModel.set_classes(foreground_search_prompt, yoloeModel.get_text_pe(foreground_search_prompt))
            results = yoloeModel.predict(img_copy)

            results[0].show()

            #-----------------------------------------------------
            '''
            rd.image_predictor.set_image(image)

            m_gd, m_h, m_l = None, None, None
            n_frames += 1

            # ## Grounding DINO - Finding objects in the bounding box
            image_source, gd_image = load_image(folder_path + '/' + name)

            gd_boxes, logits, phrases = predict(
                model=model,
                image=gd_image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device=device.type,
            )
            # annotated_frame = annotate(image_source=image_source, boxes=gd_boxes, logits=logits,
            #                            phrases=phrases)
            # cv2.imwrite("debug/Grounding_Dino/" + name, annotated_frame)

            h, w, _ = image_source.shape
            gd_boxes = gd_boxes * torch.Tensor([w, h, w, h])
            gd_boxes = box_convert(boxes=gd_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            for box in gd_boxes:
                if utility.is_contained(box, background_box, 1):
                    m_gd, s_gd, _ = rd.image_predictor.predict(
                        box=box,
                        multimask_output=False,
                    )
                    # salvataggio dell'immagine
                    utility.show_masks(image, m_gd, s_gd, borders=False, show=False, savefig=True,
                                       save_path="debug/gd2/",
                                       save_name=name, box_coords=box)

                    d_gd = utility.segmentation_metrics(m_gd, ground_truth)
                    for metric in d_gd:
                        if metric == 'IoU':
                            gd_iou += d_gd[metric]
                        elif metric == 'Dice':
                            gd_dice += d_gd[metric]
                        elif metric == 'Precision':
                            gd_precision += d_gd[metric]
                        elif metric == 'Recall':
                            gd_recall += d_gd[metric]

            print("Grounding Dino - frame: ", n_frames, ", IoU: ", gd_iou / n_frames, ", Dice: ", gd_dice / n_frames,
                  ", Precision: ", gd_precision / n_frames, ", Recall: ", gd_recall / n_frames)

            # GRID of points

            points, labels = utility.create_grid(background_box, points_per_row=[3, 4, 5])

            masks, scores, logits = rd.image_predictor.predict(
                point_coords=points,
                point_labels=labels,
                # box=background_box,
                multimask_output=False,
                return_logits=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            # print(masks.shape) # (3, 512, 640)
            utility.show_masks(image, masks > 0, scores, input_labels=labels, point_coords=points, show=False,
                               savefig=True,
                               save_path='debug/grid/', save_name=p)

            l = np.squeeze(masks)
            l[l < 0] = 0
            # for i in range(4):
            #
            #     mask, score, _ = rd.image_predictor.predict(
            #         point_coords=points[i:i+5],
            #         point_labels=labels[i:i+5],
            #         box=background_box,
            #         return_logits=True,
            #         multimask_output=False,
            #     )
            #     log += mask

            # boxes = traditional_detection(im)
            # for box in boxes:
            #
            #     m, s, _ = rd.image_predictor.predict(
            #         box=box,
            #         multimask_output=False
            #     )
            #     utility.show_masks(image, m, s, box_coords=box)
            #     input("Press Enter to continue...")
            #

            # utility.show_masks(image, mask,score,box_coords=background_box)
            # seg_im = draw_mask(image, mask)
            # cv2.imshow(name, seg_im)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # Step 3: Ensure it's in 2D format (H, W) for OpenCV
            # logits_mask_2d = np.squeeze(log)  # Shape: (H, W)

            sigmoid_mask = utility.sigmoid(l)  # Values are now in range [0, 1]

            # Step 2: Convert to uint8 (0-255 range for OpenCV)
            sigmoid_mask_2d = (sigmoid_mask * 255).astype(np.uint8)  # Shape: (1, H, W)

            sig = (sigmoid_mask_2d > 127).astype(np.uint8)  # binarify the mask to search for holes

            coord_holes, holes_labels = utility.find_holes(sig, 100)
            if len(coord_holes) > 0:

                m_h, s_h, _ = rd.image_predictor.predict(
                    point_coords=coord_holes,
                    point_labels=holes_labels,
                    multimask_output=False,
                )
                utility.show_masks(image, m_h, s_h, borders=False, show=False, savefig=True, point_coords=coord_holes,
                                   input_labels=holes_labels, save_path='debug/holes/', save_name=name)

                d_h = utility.segmentation_metrics(m_h, ground_truth)
                for metric in d_h:
                    if metric == 'IoU':
                        h_iou += d_h[metric]
                    elif metric == 'Dice':
                        h_dice += d_h[metric]
                    elif metric == 'Precision':
                        h_precision += d_h[metric]
                    elif metric == 'Recall':
                        h_recall += d_h[metric]

            print("Holes - frame: ", n_frames, ", IoU: ", h_iou / n_frames, ", Dice: ", h_dice / n_frames,
                  ", Precision: ", h_precision / n_frames, ", Recall: ", h_recall / n_frames)

            # Binarify with otsu

            thresh_val, binary_mask = cv2.threshold(sigmoid_mask_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            binary_mask[binary_mask > 127] = 1

            if binary_mask.dtype != np.uint8:
                binary_mask = binary_mask.astype(np.uint8)

            coord_holes, o_labels = utility.find_holes(binary_mask, 100)
            if len(coord_holes) > 0:
                m_o, s_o, _ = rd.image_predictor.predict(
                    point_coords=coord_holes,
                    point_labels=o_labels,
                    multimask_output=False,
                )
                utility.show_masks(image, m_o, np.array([0]), borders=False, point_coords=coord_holes,
                                   input_labels=o_labels,
                                   show=False, savefig=True, save_path='debug/otzu/', save_name=name)

                d_o = utility.segmentation_metrics(m_o, ground_truth)
                for metric in d_o:
                    if metric == 'IoU':
                        otsu_iou += d_o[metric]
                    elif metric == 'Dice':
                        otsu_dice += d_o[metric]
                    elif metric == 'Precision':
                        otsu_precision += d_o[metric]
                    elif metric == 'Recall':
                        otsu_recall += d_o[metric]

            print("Otsu - frame: ", n_frames, ", IoU: ", otsu_iou / n_frames, ", Dice: ", otsu_dice / n_frames,
                  ", Precision: ", otsu_precision / n_frames, ", Recall: ", otsu_recall / n_frames)
                  
                  '''
    except(KeyboardInterrupt, SystemExit):
        print("Exiting...")
        print('iou', gd_iou, h_iou, otsu_iou)
        print('dice', gd_dice, h_dice, otsu_dice)
        print('precision', gd_precision, h_precision, otsu_precision)
        print('recall', gd_recall, h_recall, otsu_recall)
        print('n_frames ', n_frames)
        print('skipped ', skipped)