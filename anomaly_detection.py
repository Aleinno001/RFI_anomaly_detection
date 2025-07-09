import numpy
import pandas as pd
import seaborn as sns
from PIL.ImageChops import offset
from groundingdino.util.inference import load_model, load_image
from scipy.interpolate import interp1d, make_interp_spline, splprep, splev, UnivariateSpline
import os
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gc
import time
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor
from ultralytics import SAM
from scipy.signal import savgol_filter

import utility
from gpu_utility import set_device


def read_config(config_path):
    import configparser

    # Set default values
    config = {
        'sam2_checkpoint': "./models/sam2.1/sam2.1_hiera_tiny.pt",
        'sam2_cfg_path': "./configs/sam2.1/sam2.1_hiera_t.yaml",
        'groundingdino_checkpoint': "./models/grounding_dino/groundingdino_swint_ogc.pth",
        'groundingdino_cfg_path': "./configs/grounding_dino/GroundingDINO_SwinT_OGC.py"
    }

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return config

    # Read config file
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path)

        # Update config with values from file
        if 'MODEL_PATHS' in parser:
            for key in config:
                if key in parser['MODEL_PATHS']:
                    config[key] = parser['MODEL_PATHS'][key]

        print(f"Config loaded from {config_path}")
    except Exception as e:
        print(f"Error reading config file: {e}")
        print("Using default values.")

    return config


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Video anomaly detection using SAM2 and GroundingDINO")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="./config.ini",
        help="Path to configuration file with model paths (default: ./config.ini)"
    )

    # Required arguments
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        default="./test_video/test_video_san_donato.mp4",
        help="Path to input video file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./test_video/output",
        help="Path to save output frames (default: ./test_video/output)"
    )

    # Model parameters
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.22,
        help="Box threshold for GroundingDINO (default: 0.22)"
    )

    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.18,
        help="Text threshold for GroundingDINO (default: 0.18)"
    )

    # save frames ?
    parser.add_argument(
        "--save_frames",
        action="store_true",
        default=False,
        help="Save frames with detected objects (default: False)"
    )

    # show frames ?
    parser.add_argument(
        "--show_frames",
        action="store_true",
        default=True,
        help="Show frames with detected objects (default: False)"
    )

    # Additional options
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times, e.g., -vvv)"
    )

    return parser.parse_args()

def extract_main_internal_railway_points_and_labels(image_source, gd_box, rails_masks):
    #Provo a fare la media tra: metà dell'iimagine, metà della gd box, metà tra i binari di canny

    width = image_source.shape[1]
    height = image_source.shape[0]
    gd_width = gd_box[2]-gd_box[0]
    x = gd_box[0]

    # -----------------CANNY EDGES--------------------
    image_cropped = image_source[int(gd_box[1]):int(gd_box[3]), int(gd_box[0]):int(gd_box[2])]
    img_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=60, threshold2=90)  # Canny Edge Detection
    # FIXME Come fare per evitare che gli edges rilevati non siano altro che le rails?
    # -----------------HUGS EDGES--------------------
    x = int(gd_box[0])
    y = int(gd_box[1])
    hc = image_cropped.shape[0]
    wc = image_cropped.shape[1]

    # Overlaps the edge detection of the cropped GD detection box onto the fullsize image, full black so that there are no other contours except the ones of Canny
    blacked_image = np.zeros_like(image_source)
    blacked_image = cv2.cvtColor(blacked_image, cv2.COLOR_BGR2GRAY)
    blacked_image[y:y + hc, x:x + wc] = edges

    #Find contours from the edge image
    contours, hierarchy = cv2.findContours(blacked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Draw contours on a blank canvas (for visualization)
    output = np.zeros_like(blacked_image)
    cv2.drawContours(output, contours, -1, (255), 1)

    #Filter meaningful curves by length or area
    meaningful_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > 200]

    overlay = image_source.copy()
    cv2.drawContours(overlay, meaningful_contours, -1, (0, 255, 0), 2)

    black_and_mask = np.zeros_like(image_source)
    cv2.drawContours(black_and_mask, meaningful_contours, -1, (0, 255, 0), 2)

    mask_image = None
    #Getting only the rail detection mask
    #TODO visto che ho accesso alle maschere, devo mette i punti rossi dentro gli stacoli e i punti verdi toglierli se dentro le maschere
    for obj_id, mask in rails_masks.items():
        h, w = mask.shape[-2:]
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        break

    # ----------------Extracting some points from mask -------------------
    # Creating a black image with only the curves of the mask

    black_and_mask = np.zeros_like(image_source)
    cv2.drawContours(black_and_mask, meaningful_contours, -1, (0, 255, 0), 2)

    #Here if previous mask was not calculated (ex. first frame), it tries to prompt some points centered in the rails
    if mask_image is None:
        #Calculating three different "middle" points to get the center between rails
        gdbox_midde_point = x + int(gd_width / 2)
        image_midde_point = int(width / 2)
        average_points = []
        average_levels = []
        for j in range(y+hc,y+hc-int(0.03*y+hc),-1):
            for i in range(x,x+wc,3):
                if black_and_mask[j][i][1] == 255:
                    average_points.append(i)
            if len(average_points) > 0:
                average_levels.append(int(sum(average_points) / len(average_points)))

        if len(average_levels) > 0:
            edges_rails_midde_point = int(sum(average_points) / len(average_points))
            #Faccio la media dei tre centri calcolati sopra
            average_midde_point = int((gdbox_midde_point + image_midde_point + edges_rails_midde_point) / 3)
        else:
            average_midde_point = int((gdbox_midde_point + image_midde_point) / 2)

        points = []
        points.append([average_midde_point, height-10])
        points.append([average_midde_point, height-50])
        points.append([average_midde_point-20, height - 50])
        points.append([average_midde_point+20, height - 50])
        points.append([average_midde_point, height - 90])
        points.append([average_midde_point, height - 130])
        points.append([average_midde_point+20, height - 130])
        points.append([average_midde_point-20, height - 130])
        points.append([average_midde_point, height - 170])
        labels = np.ones(len(points))
    else:
        x_mask_points = np.array([])
        mask_middle_points = np.array([])
        avg_array = np.array([])
        y_of_avg_array = np.array([])
        for j in range(y + hc, y, -10):
            for i in range(x, x + wc, 10):
                if mask_image[j][i][0] != 0 or mask_image[j][i][1] != 0 or mask_image[j][i][2] != 0:
                    x_mask_points = np.append(x_mask_points, [i])
            if len(x_mask_points) > 0:
                avg_x = int(sum(x_mask_points) / len(x_mask_points))
                mask_middle_points = np.append(mask_middle_points, [avg_x, j])
                avg_array = np.append(avg_array, avg_x)
                y_of_avg_array = np.append(y_of_avg_array, j)
            x_mask_points = np.array([])

        #Calculating smooth curve
        xhat = savgol_filter(avg_array, 10, 3)
        savol_array = []
        x_avg_array_filtered = np.array(xhat)
        for i in range(len(y_of_avg_array)):
            savol_array.append([x_avg_array_filtered[i], y_of_avg_array[i]])

        #From curve generating point prompts at both sides
        savol_array_left = savol_array.copy()
        savol_array_expanded = []
        savol_array_expanded_negative = []
        for i in range(len(savol_array)):        #TODO per migliorare i point prompt, potrei mettere dei punti a label negativa a destra e sinistra
            if i>0 and i<len(savol_array_left)-1 and (i%6) == 0:
                #savol_array_expanded.append(savol_array[i])
                current_y_in_rail_box = savol_array[i][1] - y
                savol_array_expanded.append([savol_array[i][0]-int(current_y_in_rail_box*0.12), savol_array[i][1]])
                savol_array_expanded.append([savol_array[i][0]+int(current_y_in_rail_box*0.12), savol_array[i][1]])
                savol_array_expanded_negative.append([savol_array[i][0] - int(current_y_in_rail_box * 0.90), savol_array[i][1]])
                savol_array_expanded_negative.append([savol_array[i][0] + int(current_y_in_rail_box * 0.90), savol_array[i][1]])
        points = np.array(np.concatenate((savol_array_expanded, savol_array_expanded_negative)))
        labels = np.ones(len(savol_array_expanded))
        neg_labels = np.zeros(len(savol_array_expanded_negative))
        labels = np.concatenate((labels, neg_labels))

    return points, labels

def smooth_curve_from_points(rail_points_x, rail_points_y):
    rail_points_x = np.array(rail_points_x)
    rail_points_y = np.array(rail_points_y)
    coeffs = np.polyfit(rail_points_y, rail_points_x, deg=3)  # Polinomio di grado 3
    poly = np.poly1d(coeffs)
    # Definizione di un intervallo più fitto per la curva
    y_smooth = np.linspace(rail_points_y.min(), rail_points_y.max(), 200)
    x_fit = poly(y_smooth)
    xy_array = np.column_stack((x_fit, y_smooth))
    return xy_array


def refine_mask(mask, previous_mask=None):
    h, w = mask.shape[-2:]
    black_image = np.zeros((h, w), dtype=np.uint8)

    # Convert your mask to uint8 and pad it properly
    mask = mask.astype('uint8')

    #Somplifying image with blur and morphology, removing noise
    # FloodFill requires the mask to be (H+2, W+2)
    padded_mask = np.zeros((h + 2, w + 2), dtype='uint8')
    padded_mask[1:h + 1,
    1:w + 1] = mask
    cv2.floodFill(black_image, padded_mask, (0, 0), 255)
    black_and_white_mask_image = cv2.bitwise_not(black_image)

    # blur
    blur = cv2.GaussianBlur(black_and_white_mask_image, (0, 0), sigmaX=3, sigmaY=3)
    # otsu threshold
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    #----------Cleaning and obtaining filled and crisp mask in BW----------
    black_image = np.zeros((h, w), dtype=np.uint8)
    padded_mask = np.zeros((h + 2, w + 2), dtype='uint8')
    padded_mask[1:h + 1,
    1:w + 1] = morph
    cv2.floodFill(black_image, padded_mask, (0, 0), 255)
    black_and_white_mask_image = cv2.bitwise_not(black_image)
    #----------------checking old mask------------
    if previous_mask is not None:
        previous_mask_image = previous_mask*255
        previous_mask_image = previous_mask_image.astype(black_and_white_mask_image.dtype)

    #Removing mask "islands"or other noise not connected to the main detected object
    contours, _ = cv2.findContours(black_and_white_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    main_contour = max(contours, key=cv2.contourArea)
    main_contour = main_contour[:, 0, :]  # Remove nesting

    black_image = np.zeros_like(black_and_white_mask_image)
    cv2.drawContours(black_image, [main_contour], -1, (255, 255, 255), 3)
    cv2.fillPoly(black_image, pts=[main_contour], color=(255, 255, 255))

    lrail_p_x = []
    lrail_p_y = []
    rrail_p_x = []
    rrail_p_y = []
    for j in range(h-1,0,-3):
        i=0
        left_x_temp_points = []
        right_x_temp_points = []
        while i<w:
            if black_image[j][i]==255:
                left_x_temp_points.append(i)
                i+=1
                while i<w and black_image[j][i]==255:
                    i+=1
                right_x_temp_points.append(i-1)
            else:
                i+=1
        if len(left_x_temp_points)>0 and len(right_x_temp_points)>0:
            left_x_temp_points = np.array(left_x_temp_points)
            lrail_p_x.append(left_x_temp_points.min())
            lrail_p_y.append(j)
            right_x_temp_points = np.array(right_x_temp_points)
            rrail_p_x.append(right_x_temp_points.max())
            rrail_p_y.append(j)

    xy_array_left_rail = smooth_curve_from_points(lrail_p_x, lrail_p_y)

    xy_array_right_rail = smooth_curve_from_points(rrail_p_x, rrail_p_y)

    xy_array_right_rail = xy_array_right_rail[::-1]
    poly_points = np.array(xy_array_left_rail,dtype=np.int32)
    poly_points = np.concatenate((poly_points, np.array(xy_array_right_rail,dtype=np.int32)))
    poly_points = poly_points.reshape((-1, 1, 2))
    polygonal_mask = np.zeros_like(black_image)
    cv2.polylines(polygonal_mask, [poly_points], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.fillPoly(polygonal_mask, pts=[poly_points], color=(255, 255, 255))

    #FIXME da mettere in un metodo
    #Using past mask for compensation of great variations in the mask
    if previous_mask is not None:
        intersection = cv2.bitwise_and(polygonal_mask, previous_mask_image)
        result = cv2.bitwise_xor(intersection, polygonal_mask)
        scan_height = int(3*h/4)    #Checking only the bottom part of the mask, wich is the part that sholud be the steadiest
        #Intersected mask (difference between old and new mask) pixel counting
        white_px = 0
        for j in range(scan_height,h):
            for i in range(w):
                if result[j][i] == 255:
                    white_px += 1
        #Total old mask pixel counting
        tot_px = 0
        for j in range(scan_height,h):
            for i in range(w):
                if previous_mask_image[j][i] == 255:
                    tot_px += 1
        difference_percentage = white_px / tot_px
        if difference_percentage>0.05:      #Using old mask bottom part if the canche is too drastic
            old_mask_copy = previous_mask_image.copy()
            for j in range(scan_height):
                for i in range(w):
                    old_mask_copy[j][i] = 0
            fixed_image = cv2.bitwise_or(polygonal_mask, old_mask_copy)
            polygonal_mask = fixed_image
    #Expanding the mask at the edges for better coverage of the rails
    blur = cv2.GaussianBlur(polygonal_mask, (0, 0), sigmaX=4, sigmaY=1)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)[1]
    a = thresh.astype(bool)
    return a

# main
def main():
    # Parse command line arguments
    args = parse_args()

    # Read config file
    config = read_config(args.config)

    # Set the device
    device = set_device()
    print(f"using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Load the SAM2 model
    sam2_checkpoint = config['sam2_checkpoint']
    sam2_cfg_path = config['sam2_cfg_path']
    sam2 = build_sam2(sam2_cfg_path, sam2_checkpoint, device=device)
    #image_predictor = SAM2ImagePredictor(sam2)
    #video_predictor = build_sam2_video_predictor(sam2_cfg_path, sam2_checkpoint, device=device)

    #FIXME da scrivere meglio
    video_predictor_rails = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", "models/sam2.1/sam2.1_hiera_tiny.pt", device=device)

    # Load the GroundingDINO model
    groundingdino_checkpoint = config['groundingdino_checkpoint']
    groundingdino_cfg_path = config['groundingdino_cfg_path']
    groundingdino = load_model(groundingdino_cfg_path, groundingdino_checkpoint, device=device)
    TEXT_PROMPT = "object ."
    BOX_TRESHOLD = args.box_threshold
    TEXT_TRESHOLD = args.text_threshold
    #FIXME da scrivere meglio
    BACKGROUND_PROMPT = "one train track."
    OBSTACLE_PROMPT = "all things ."
    BOX_TRESHOLD_RAILS = 0.25
    TEXT_TRESHOLD_RAILS = 0.15
    BOX_TRESHOLD_OBSTACLES = 0.20
    TEXT_TRESHOLD_OBSTACLES = 0.16

    print(f"Loaded SAM2 model from {sam2_checkpoint}")
    print(f"Loaded GroundingDINO model from {groundingdino_checkpoint}")
    print(f'Using box threshold: {BOX_TRESHOLD}, text threshold: {TEXT_TRESHOLD}')
    print(f'Input video: {args.input_video}')
    print(f'Output path: {args.output_path}')

    # Create a temporary directory for frame storage
    temp_dir = os.path.join("temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Open the video stream
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise Exception(f"Could not open video stream: {args.input_video}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video stream at {fps} FPS, resolution: {width}x{height}")

    # Object tracking variables
    ann_obj_id = 1  # Object ID counter
    ann_all_obj_id = 1
    ann_rail_id = 1
    last_masks = {}  # Store the last known mask for each object
    last_masks_rails = {}  # Store the last known mask for each object
    last_masks_obstacles = {}  # Store the last known mask for each object
    frame_idx = 0
    railway_box = None  # Store railway box for future reference
    main_railway_box = None

    test = 0

    try:
        while True:
            print(f"\n--- Processing frame {frame_idx} ---")
            start_time = time.time()

            # Read the next frame
            success, frame = cap.read()
            if not success:
                print("End of stream or error reading frame")
                break

            # Clear temp directory
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))

            # Save current frame to temp directory
            frame_path = os.path.join(temp_dir, "000000.jpg")
            cv2.imwrite(frame_path, frame)

            # Convert to RGB for visualization
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Initialize new state for this frame
            torch.cuda.empty_cache()
            gc.collect()
            #inference_state = video_predictor.init_state(video_path=temp_dir)
            inference_state_rails = video_predictor_rails.init_state(video_path=temp_dir)

            # Process based on frame index
            if frame_idx == 0:
                '''
                # First frame: detect railway and objects with Grounding DINO
                dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, 'railway . object .', device, BOX_TRESHOLD=BOX_TRESHOLD,
                    TEXT_TRESHOLD=TEXT_TRESHOLD
                )

                # Find railway box and object points
                max_score_railway = 0
                object_points = []

                for i, phrase in enumerate(phrases):
                    if phrase == 'railway' and dino_scores[i] > max_score_railway:
                        railway_box = dino_boxes[i]
                        max_score_railway = dino_scores[i]
                    elif phrase == 'object':
                        x_min, y_min, x_max, y_max = dino_boxes[i]
                        object_points.append([(x_min + x_max) // 2, (y_min + y_max) // 2])

                print(f"Found railway: {railway_box is not None}, objects: {len(object_points)}")

                # Add railway to tracking
                if railway_box is not None:
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=ann_obj_id,
                        box=railway_box,
                    )

                    # Store railway mask
                    last_masks[ann_obj_id] = (out_mask_logits[0] > 0).cpu().numpy()

                # Add detected objects to tracking
                for obj_point in object_points:
                    ann_obj_id += 1
                    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=ann_obj_id,
                        points=[obj_point],
                        labels=np.array([1], np.int32),
                    )

                    # Store object mask
                    idx = list(out_obj_ids).index(ann_obj_id) if ann_obj_id in out_obj_ids else 0
                    last_masks[ann_obj_id] = (out_mask_logits[idx] > 0).cpu().numpy()
                '''

                #DETECTION OF THE MAIN RAILWAY AND THE OBSTACLES
                dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, BACKGROUND_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_RAILS,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_RAILS, show=False,
                )#TODO Rimuovere lo show=True

                #TODO tra le bounding box togliere quelle che sono troppo grosse

                # Find main railway box and object points
                max_score_railway = 0
                main_box_area = 0
                #FIXME come faccio a fare che prenda solo il binario centrale, perchè alcune volte prende tutti i binari
                #for i,box in enumerate(dino_boxes):
                #    if dino_scores[i] > max_score_railway:
                #        main_railway_box = box
                #        max_score_railway = dino_scores[i]

                max_score_railway = 0
                for i,box in enumerate(dino_boxes):
                    if main_railway_box is None:
                        main_railway_box = box
                        max_score_railway = dino_scores[i]
                    else:
                        dino_box_width = box[2] - box[0]
                        dino_box_center = (box[0] + box[2]) // 2
                        image_center = width // 2
                        dino_abs_distance_from_center = abs(dino_box_center - image_center)
                        if dino_box_width >= int(0.75*width):       #FIXME controllare se il 75% va bene
                            if dino_abs_distance_from_center < int(0.25*width):     #FIXME controlare se il 25% va bene
                                if dino_scores[i] > max_score_railway:
                                    main_railway_box = box
                                    max_score_railway = dino_scores[i]

                '''        
                #Picking the smallest Dino box
                for i,box in enumerate(dino_boxes):
                    if main_railway_box is None:
                        main_railway_box = box
                    else:
                        gd_width = box[2] - box[0]
                        gd_height = box[3] - box[1]
                        actual_box_area = gd_width * gd_height
                        main_width = main_railway_box[2] - main_railway_box[0]
                        main_height = main_railway_box[3] - main_railway_box[1]
                        main_box_area = main_width * main_height
                        if actual_box_area < main_box_area:
                            main_railway_box = box
                '''

                dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, OBSTACLE_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_OBSTACLES,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_OBSTACLES, show=False,
                )  # TODO Rimuovere lo show=true

                all_obstacles_points = []

                for i,box in enumerate(dino_boxes):
                    x_min, y_min, x_max, y_max = box
                    all_obstacles_points.append([(x_min + x_max) // 2, (y_min + y_max) // 2])

                print(f"Found main railway: {main_railway_box is not None}, all obstacles: {len(all_obstacles_points)}")

                # Add railway to tracking
                if main_railway_box is not None:
                    #points, labels = extract_main_railway_points_and_labels(frame_rgb, main_railway_box,frame_idx)
                    points, labels = extract_main_internal_railway_points_and_labels(frame_rgb, main_railway_box,last_masks_rails)

                    _, out_obj_ids, out_mask_logits = video_predictor_rails.add_new_points_or_box(      #TODO la maschera generata andrebbe espandsa dai lati per includere i binari e contorni, magari anche blurrarla
                        inference_state=inference_state_rails,
                        frame_idx=0,
                        obj_id=ann_rail_id,
                        points=points,
                        labels=labels,
                        box=main_railway_box,
                    )

                    # Store railway mask
                    last_masks_rails[ann_rail_id] = refine_mask((out_mask_logits[0] > 0).cpu().numpy())

                # Add detected objects to tracking
                for obj_point in all_obstacles_points:
                    ann_rail_id += 1
                    _, out_obj_ids, out_mask_logits = video_predictor_rails.add_new_points_or_box(
                        inference_state=inference_state_rails,
                        frame_idx=0,
                        obj_id=ann_rail_id,
                        points=[obj_point],
                        labels=np.array([1], np.int32),
                    )

                    # Store object mask
                    idx = list(out_obj_ids).index(ann_rail_id) if ann_rail_id in out_obj_ids else 0
                    last_masks_rails[ann_rail_id] = (out_mask_logits[idx] > 0).cpu().numpy()


            else:
                '''
                # For non-first frames, transfer objects from previous frame
                for obj_id, mask in last_masks.items():
                    # Convert mask to proper format and find center point
                    mask_array = np.asarray(mask)
                    if mask_array.ndim > 2:
                        mask_array = mask_array.squeeze()
                        if mask_array.ndim > 2:
                            mask_array = mask_array[0]

                    # Find non-zero coordinates (points inside the mask)
                    y_indices, x_indices = np.where(mask_array > 0)

                    if len(y_indices) > 0:
                        # Use center of mass as representative point
                        center_y = int(np.mean(y_indices))
                        center_x = int(np.mean(x_indices))

                        # Special handling for railway (can use box instead of point)
                        if obj_id == 1 and railway_box is not None:
                            _, _, _ = video_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=obj_id,
                                box=railway_box,
                            )
                        else:
                            # Add object using its center point
                            _, _, _ = video_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=obj_id,
                                points=[[center_x, center_y]],
                                labels=np.array([1], np.int32),
                            )
                '''
                #FIXME da scrivere meglio
                #For non-first frames, transfer objects from previous frame

                for obj_id, mask in last_masks_rails.items():
                    # Convert mask to proper format and find center point
                    mask_array = np.asarray(mask)
                    if mask_array.ndim > 2:
                        mask_array = mask_array.squeeze()
                        if mask_array.ndim > 2:
                            mask_array = mask_array[0]

                    # Find non-zero coordinates (points inside the mask)
                    y_indices, x_indices = np.where(mask_array > 0)

                    if len(y_indices) > 0:
                        # Use center of mass as representative point
                        center_y = int(np.mean(y_indices))
                        center_x = int(np.mean(x_indices))
                        # Add object using its center point
                        # Special handling for railway (can use box instead of point)
                        if obj_id == 1 and main_railway_box is not None:
                            #points, labels = extract_main_railway_points_and_labels(frame_rgb, main_railway_box,frame_idx)
                            points, labels = extract_main_internal_railway_points_and_labels(frame_rgb,main_railway_box,last_masks_rails)
                            _, _, _ = video_predictor_rails.add_new_points_or_box(
                                inference_state=inference_state_rails,
                                frame_idx=0,
                                obj_id=obj_id,
                                points=points,
                                labels=labels,
                                box=main_railway_box,
                            )
                        else:
                            _, _, _ = video_predictor_rails.add_new_points_or_box(
                                inference_state=inference_state_rails,
                                frame_idx=0,
                                obj_id=obj_id,
                                points=[[center_x, center_y]],
                                labels=np.array([1], np.int32),
                            )

            '''
            # Propagate all objects in current frame
            result = next(video_predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0
            ))
            _, out_obj_ids, out_mask_logits = result

            # Update all masks for next frame
            for i, obj_id in enumerate(out_obj_ids):
                last_masks[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()
            '''

            #FIXME da scrivere meglio
            # Propagate all objects in current frame
            result_rails = next(video_predictor_rails.propagate_in_video(
                inference_state_rails,
                start_frame_idx=0
            ))
            _, out_obj_ids, out_mask_logits = result_rails


            # Update all masks for next frame
            for i, obj_id in enumerate(out_obj_ids):
                if obj_id == 1: #Fixme mettere se è la maschera binario
                    last_masks_rails[obj_id] = refine_mask((out_mask_logits[i] > 0).cpu().numpy(),last_masks_rails[obj_id])  #FIXME non fa refine al binario ma solo a tutto il resto
                else:
                    last_masks_rails[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()

            # Check for new objects periodically
            if (frame_idx % 15 == 0 and frame_idx > 0):
                '''
                dino_boxes, phrases, _ = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, 'object .', device, BOX_TRESHOLD=BOX_TRESHOLD,
                    TEXT_TRESHOLD=TEXT_TRESHOLD
                )

                # Check each detected object
                for i, phrase in enumerate(phrases):
                    if phrase == 'object':
                        x_min, y_min, x_max, y_max = dino_boxes[i]
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2

                        # Check if this object is already tracked
                        already_tracked = False
                        for mask in last_masks.values():
                            mask_array = np.asarray(mask)
                            if mask_array.ndim > 2:
                                mask_array = mask_array.squeeze()
                                if mask_array.ndim > 2:
                                    mask_array = mask_array[0]

                            # Check if point is inside any existing mask
                            if (0 <= int(center_y) < mask_array.shape[0] and
                                    0 <= int(center_x) < mask_array.shape[1] and
                                    mask_array[int(center_y), int(center_x)]):
                                already_tracked = True
                                break

                        # Add new object if not already tracked
                        if not already_tracked:
                            ann_obj_id += 1
                            print(f"New object {ann_obj_id} detected at frame {frame_idx}")

                            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=ann_obj_id,
                                points=[[center_x, center_y]],
                                labels=np.array([1], np.int32),
                            )

                            # Re-propagate with the new object
                            result = next(video_predictor.propagate_in_video(
                                inference_state,
                                start_frame_idx=0
                            ))
                            _, out_obj_ids, out_mask_logits = result

                            # Update masks dictionary
                            for i, obj_id in enumerate(out_obj_ids):
                                last_masks[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()
                '''
                # DETECTION OF THE MAIN RAILWAY AND THE OBSTACLES
                dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, BACKGROUND_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_RAILS,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_RAILS, show=False,
                )  # TODO Rimuovere lo show=true
                # TODO tra le bounding box togliere quelle che sono troppo grosse
                # Find main railway box and object points
                # FIXME come faccio a fare che prenda solo il binario centrale, perchè alcune volte prende tutti i binari

                for i, box in enumerate(dino_boxes):
                    if main_railway_box is None:
                        main_railway_box = box
                    else:
                        gd_width = box[2] - box[0]
                        gd_height = box[3] - box[1]
                        actual_box_area = gd_width * gd_height
                        main_width = main_railway_box[2] - main_railway_box[0]
                        main_height = main_railway_box[3] - main_railway_box[1]
                        main_box_area = main_width * main_height
                        if actual_box_area < main_box_area:
                            main_railway_box = box

                #FIXME da scrivere meglio
                dino_boxes, phrases, _ = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, OBSTACLE_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_OBSTACLES,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_OBSTACLES, show = False,
                )#TODO rimuovere show=true


                # Check each detected object
                for i, phrase in enumerate(phrases):
                    if phrases[i] == 'all things': #TODO gestire la rail separata
                        x_min, y_min, x_max, y_max = dino_boxes[i]
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2

                        # Check if this object is already tracked
                        already_tracked = False
                        for mask in last_masks_rails.values():
                            mask_array = np.asarray(mask)
                            if mask_array.ndim > 2:
                                mask_array = mask_array.squeeze()
                                if mask_array.ndim > 2:
                                    mask_array = mask_array[0]

                            #TODO da gestire il fatto del contenuto in un altra maschera

                            # Check if point is inside any existing mask
                            if (0 <= int(center_y) < mask_array.shape[0] and
                                    0 <= int(center_x) < mask_array.shape[1] and
                                    mask_array[int(center_y), int(center_x)]):
                                already_tracked = True
                                break


                        # Add new object if not already tracked
                        if not already_tracked:
                            ann_rail_id += 1
                            print(f"New object {ann_rail_id} detected at frame {frame_idx}")

                            _, out_obj_ids, out_mask_logits = video_predictor_rails.add_new_points_or_box(
                                inference_state=inference_state_rails,
                                frame_idx=0,
                                obj_id=ann_rail_id,
                                points=[[center_x, center_y]],
                                labels=np.array([1], np.int32),
                            )

                            # Re-propagate with the new object
                            result_rails = next(video_predictor_rails.propagate_in_video(
                                inference_state_rails,
                                start_frame_idx=0
                            ))
                            _, out_obj_ids, out_mask_logits = result_rails

                            # Update masks dictionary
                            for i, obj_id in enumerate(out_obj_ids):
                                if obj_id == 1:
                                    last_masks_rails[obj_id] = refine_mask((out_mask_logits[i] > 0).cpu().numpy(),last_masks_rails[obj_id])
                                else:
                                    last_masks_rails[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()

            # Create visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(frame_rgb)
            for obj_id, mask in last_masks_rails.items():
                utility.show_mask_v(mask, plt.gca(), obj_id=obj_id)
                break
            if args.show_frames:  # show the plt image using OpenCV
                cv2.imshow("Processed video frame", utility.plt_figure_to_cv2( plt.gcf()))
                key = cv2.waitKey(1)
                if key == ord('q'):
                    raise KeyboardInterrupt
            if args.save_frames:  #to remove True in args.save_frames
                plt.savefig(os.path.join(args.output_path, f"frame_{frame_idx:06d}.jpg"))
            plt.close()

            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"Frame processed in {processing_time:.2f}s")

            # Increment frame counter
            frame_idx += 1      #FIXME per refreshare la segmentazione di sam2 posso detectare il cambio dei binari con una IA e refreshare lì, invece che ogni 15 frame

            # Clear memory for next iteration
            #del inference_state
            del inference_state_rails
            gc.collect()
            torch.cuda.empty_cache()

    except (KeyboardInterrupt, SystemExit):
        print("Exiting gracefully...")
    finally:
        # Release resources
        cap.release()

        # Clean up temp directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

        if args.show_frames:
            cv2.destroyAllWindows()

        print(f"Processing completed or interrupted after {frame_idx} frames")


if __name__ == "__main__":
    main()
