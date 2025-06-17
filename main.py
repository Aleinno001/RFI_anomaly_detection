from groundingdino.util.inference import load_model, load_image, predict, annotate

import numpy as np
from torchvision.ops import box_convert
import torch
import torchvision
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor

import utility

'''
1. problema della detection di anomalie nel termico, contesto di uso
2. come funziona SAM2
3. esperimenti fatti per riconoscere la presenza delle anomalie ( a volte SAM2 è in grado di beccarli, altre volte 
    guardando i logits possiamo restituirli)
4. risultati sul dataset sintetico FOSH

:Rilevazione anomalie:
    Non Deep: con Otzu o Blob detection (entrambe di OpenCV)
    Deep: SAM^2
    
    
'''

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class RailAnomalyDetector:
    def __init__(self, model_cfg_path, sam2_checkpoint, device):
        self.device = device
        self.sam2 = build_sam2(model_cfg_path, sam2_checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.sam2,
            points_per_side=32,  # Increase for more detailed segmentation
            pred_iou_thresh=0.7,  # Adjust threshold based on your needs 0.86 def
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2
        )
        self.image_predictor = SAM2ImagePredictor(self.sam2)
        self.video_predictor = build_sam2_video_predictor(model_cfg_path, sam2_checkpoint, device=device)

    def generate_masks_automatically(self, image, show=True, save=False, save_path=None, save_name=None, ret=False):
        """
        Returns:
           masks:list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """
        print(f"Generating masks...")
        masks = self.mask_generator.generate(image)
        anomalies = []
        for mask in masks:
            # if 300 < mask['point_coords'][0][1] < 450:
            # print(mask['point_coords'], mask['predicted_iou'], mask['stability_score'])
            anomalies.append(mask)

        # plt.figure(figsize=(20, 20))

        if show:
            plt.imshow(image)
            plt.axis('off')
            utility.show_anns(masks, False)
            plt.show()
        if save and save_path is not None and save_name is not None:
            plt.savefig(f"./{save_path}/{save_name[:-4]}.png")
        if ret:
            return anomalies

    def generate_mask_from_points(self, image, points, labels, show=True, save=False, save_path=None, save_name=None,
                                  return_logits=False, multimask_output=True, ret=False):
        """
        Returns:
          masks:(np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          scores:(np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          logits:(np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        self.image_predictor.set_image(image)

        masks, scores, logits = self.image_predictor.predict(
            point_coords=points,
            point_labels=labels,
            return_logits=return_logits,
            multimask_output=multimask_output,
        )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # plt.imshow(logits.squeeze(), cmap='viridis')  # Usa una colormap a tua scelta (es. 'hot', 'Reds')
        # plt.colorbar(label='Valore dei Logits')
        # plt.title("Logits della Maschera")
        # plt.axis('off')
        # plt.show()

        if show:
            utility.show_masks(image, masks, scores, point_coords=points, input_labels=labels, savefig=save,
                               save_path=save_path, save_name=save_name, show=show)
        if save and save_path is not None and save_name is not None:
            plt.savefig(f"./{save_path}/{save_name[:-4]}.png")
        if ret:
            return masks, scores, logits

    def generate_mask_from_boxes(self, image, boxes, show=True, save=False, save_path=None, save_name=None,
                                 return_logits=False,
                                 multimask_output=True, ret=False):
        """
        Returns:
          masks:(np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          scores:(np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          logits:(np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        self.image_predictor.set_image(image)
        # embedding = self.image_predictor.get_image_embedding()
        # # Flatten spatial dimensions
        # embedding_flat = embedding.reshape(256, -1).T  # Shape: (H*W, C)

        # # Apply PCA to reduce 256 channels to 3 (RGB)
        # pca = PCA(n_components=3)
        # embedding_pca = pca.fit_transform(embedding_flat)
        #
        # # Reshape back to (H, W, 3)
        # embedding_pca = embedding_pca.reshape(64, 64, 3)
        #
        # # Normalize for display
        # embedding_pca = (embedding_pca - embedding_pca.min()) / (embedding_pca.max() - embedding_pca.min())
        #
        # plt.imshow(embedding_pca)
        # plt.title("PCA Reduced Embedding")
        # plt.show()

        masks, scores, logits = self.image_predictor.predict(
            box=boxes,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        if show:
            plt.imshow(logits.squeeze(), cmap='viridis')  # Usa una colormap a tua scelta (es. 'hot', 'Reds')
            plt.colorbar(label='Valore dei Logits')
            plt.title("Logits della Maschera")
            plt.axis('off')
            plt.show()

            plt.imshow(masks[0])
            utility.show_masks(image, masks, scores, box_coords=boxes, savefig=save, save_path=save_path,
                               save_name=save_name, show=show)
        if save and save_path is not None and save_name is not None:
            plt.savefig(f"./{save_path}/{save_name[:-4]}.png")
        if ret:
            return masks, scores, logits
        return None

    def video_analysis(self, video_dir, segmented_video_dir, dino_model):
        try:
            # Extract frames
            frame_names = [
                p for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            # Sorting frames
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))
            # Setting the inference_state
            inference_state = self.video_predictor.init_state(video_path=video_dir)

            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

            # ## Grounding DINO - Finding objects in the bounding box
            dino_boxes, phrases, dino_score = utility.grounding_Dino_analyzer(video_dir + '/' + frame_names[0],
                                                                              dino_model,
                                                                              'railway . object .', device)
            max_score_railway = 0
            railway_box = None
            object_points = []
            for i, phrase in enumerate(phrases):
                if phrase == 'railway' and dino_score[i] > max_score_railway:
                    railway_box = dino_boxes[i]
                    max_score_railway = dino_score[i]
                if phrase == 'object':
                    x_min, y_min, x_max, y_max = dino_boxes[i]
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2
                    object_points.append([x_center, y_center])

            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=railway_box,
            )
            id_objects = []
            for obj_point in object_points:
                print('obj_point ', obj_point)
                ann_obj_id += 1
                id_objects.append(ann_obj_id)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=[obj_point],
                    labels=np.array([1], np.int32),
                )

            idx_frame = 0
            video_segments = {}  # video_segments contains the per-frame segmentation results

            while idx_frame < len(frame_names):
                last_masks = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                        inference_state, start_frame_idx=idx_frame):

                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                    # plt.close('all')
                    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
                    #
                    # for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    #     utility.show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)
                    #
                    # plt.show()

                    idx_frame += 1
                    if idx_frame % 15 == 0:
                        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                            last_masks[out_obj_id] = out_mask
                        break

                # Riutilizziamo Grounding DINO ogni 15 frame per capire se ci sono nuovi oggetti
                obj_boxes, _, obj_scores = utility.grounding_Dino_analyzer(video_dir + '/' + frame_names[idx_frame - 1],
                                                                           dino_model, 'object .', device)
                new_boxes = []
                # controlliamo se i box contengono maschere già segmentate
                for obj_box in obj_boxes:
                    contained = False
                    for idx, mask in last_masks.items():
                        print('lll', mask.shape)
                        if utility.is_mask_in_box(mask, obj_box):
                            print("C'era di già")
                            contained = True
                            break
                    if contained:
                        continue
                    else:
                        new_boxes.append(obj_box)

                for idx_frame_proc in range(idx_frame - 15, idx_frame):
                    plt.close('all')
                    plt.imshow(Image.open(os.path.join(video_dir, frame_names[idx_frame_proc])))

                    for out_obj_id, out_mask in video_segments[idx_frame_proc].items():
                        utility.show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)
                    plt.savefig(os.path.join(segmented_video_dir, frame_names[idx_frame_proc]),
                                bbox_inches='tight', pad_inches=0)
                    # plt.show()

                for obj_box in new_boxes:
                    print("new box ", obj_box)
                    ann_obj_id += 1

                    x_min, y_min, x_max, y_max = obj_box
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2

                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=idx_frame,
                        obj_id=ann_obj_id,
                        points=[[x_center, y_center]],
                        labels=np.array([1], np.int32),
                    )
                # segmented_img,_ = draw_mask_2(np.array(Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))), out_mask_logits[0].cpu().numpy())
                # im = Image.fromarray(segmented_img)
                # save_path = os.path.join(segmented_video_dir, f"frame_{out_frame_idx:03d}.png")
                # im.save(save_path)# max_logit_value = out_mask_logits[0].max().item()

                # plt.imshow(out_mask_logits.squeeze(), cmap='viridis')
                # plt.colorbar(label='Valore dei Logits')
                # plt.title("Logits della Maschera")
                # plt.axis('off')
                # plt.show()
                # if out_frame_idx == 2:
                #     print("Annotiamo l'anomalia")
                #     # Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
                #     pp = np.array([[550, 370]], dtype=np.float32)
                #     # for labels, `1` means positive click and `0` means negative click
                #     ll = np.array([0], np.int32)
                #     _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                #         inference_state=inference_state,
                #         frame_idx=out_frame_idx,
                #         obj_id=ann_obj_id,
                #         points=pp,
                #         labels=ll,
                #     )
                # plt.figure(figsize=(9, 6))
                # plt.title(f"frame {out_frame_idx}")
                # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
                # utility.show_mask_v((out_mask_logits[0] > threshold).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
                # if out_frame_idx == 100:
                #     plt.close('all')

            # vis_frame_stride = 1
            # plt.close("all")
            # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            #     # image = np.asarray(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            #         utility.show_mask_v(out_mask, plt.gca(), obj_id=out_obj_id)
            #         # segmented_img = draw_mask(image, out_mask)
            #         # im = Image.fromarray(segmented_img)
            #         # save_path = os.path.join(segmented_video_dir, f"frame_{out_frame_idx:03d}.png")
            #         # im.save(save_path)
            #     plt.savefig(os.path.join(segmented_video_dir, frame_names[out_frame_idx]))
        except(KeyboardInterrupt, SystemExit):
            print("Exiting...")


def get_centers_from_mask(mask, min_area=100):
    """
    Data una maschera, ritorna le coordinate (centroide) delle aree con area maggiore di min_area.

    Args:
        mask (np.ndarray): Immagine in scala di grigi o binaria.
                          Gli oggetti di interesse dovrebbero avere valori > 0.
        min_area (int): Soglia minima dell'area (in pixel) per considerare la regione.

    Returns:
        np.ndarray: Array di coordinate (x, y) dei centroidi delle aree rilevate.
                   Shape: (N, 2) dove N è il numero di aree trovate.
    """
    # Controllo input
    if mask is None or mask.size == 0:
        return np.array([])

    # Assicuriamoci che la maschera sia 2D
    if mask.ndim > 2:
        mask = np.squeeze(mask)

    # Convertiamo a uint8 se necessario
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Binarizziamo la maschera
    # _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Trova le componenti connesse
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    centers = []
    # Il label 0 è il background, partiamo da 1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_area:
            center = [int(centroids[i][0]), int(centroids[i][1])]
            centers.append(center)

    return np.array(centers)


if __name__ == '__main__':

    ## SETTING THE MODELS

    ## SAM2

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

    # loading sam paramters
    sam2_checkpoint = "models/sam2.1/sam2.1_hiera_tiny.pt"
    model_cfg_path = "configs/sam2.1/sam2.1_hiera_t.yaml"

    np.random.seed(3)

    ##  Grounding DINO

    model = load_model("configs/grounding_dino/GroundingDINO_SwinT_OGC.py",
                       "models/grounding_dino/groundingdino_swint_ogc.pth", device.type)
    # IMAGE_PATH = "inserimento_oggetti/inserimento_oggetti_00064.jpg"
    TEXT_PROMPT = "object ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    # Where to retrive images
    folder_path = "inserimento_oggetti"

    # creating the background box
    background_box = np.array([0, 256, 640, 512])

    rd = RailAnomalyDetector(model_cfg_path, sam2_checkpoint, device=device)

    # Setting statistics
    gd_iou, h_iou, otsu_iou = 0, 0, 0
    gd_dice, h_dice, otsu_dice = 0, 0, 0
    gd_precision, h_precision, otsu_precision = 0, 0, 0
    gd_recall, h_recall, otsu_recall = 0, 0, 0

    n_frames = 0
    skipped = 0

    # Extract frames
    frame_names = [
        p for p in os.listdir(folder_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # Sorting frames
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    # FOR VIDEO ANALYSIS
    # rd.video_analysis('s_c_f','seg_new_video', model)

    # IMAGE ANALYSIS
    try:
        for p in frame_names:

            print('----------------------')
            image = cv2.imread(os.path.join(folder_path, p))
            ground_truth = cv2.imread(utility.find_corresponding_segmentation(p,
                                                                              'segmentazione_oggetti'),
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
    except(KeyboardInterrupt, SystemExit):
        print("Exiting...")
        print('iou', gd_iou, h_iou, otsu_iou)
        print('dice', gd_dice, h_dice, otsu_dice)
        print('precision', gd_precision, h_precision, otsu_precision)
        print('recall', gd_recall, h_recall, otsu_recall)
        print('n_frames ', n_frames)
        print('skipped ', skipped)
