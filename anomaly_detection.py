import numpy
from PIL.ImageChops import offset
from groundingdino.util.inference import load_model, load_image
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

def extract_main_railway_points_and_labels(image_source, gd_boxes):
    # -----------------CANNY EDGES--------------------
    # Convert to graycsale
    image_cropped = image_source[int(gd_boxes[1]):int(gd_boxes[3]), int(gd_boxes[0]):int(gd_boxes[2])]

    img_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=90)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # -----------------HUGS EDGES--------------------
    x = int(gd_boxes[0])
    y = int(gd_boxes[1])
    hc = image_cropped.shape[0]
    wc = image_cropped.shape[1]

    # Overlaps the edge detection of the cropped GD detection box onto the fullsize image, full black so that there are no other contours except the ones of Canny
    blacked_image = np.zeros_like(image_source)
    blacked_image = cv2.cvtColor(blacked_image, cv2.COLOR_BGR2GRAY)
    blacked_image[y:y + hc, x:x + wc] = edges

    # Step 3: Find contours from the edge image
    contours, hierarchy = cv2.findContours(blacked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Draw contours on a blank canvas (for visualization)
    output = np.zeros_like(blacked_image)
    cv2.drawContours(output, contours, -1, (255), 1)

    # Optional: Filter meaningful curves by length or area
    meaningful_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > 400]

    # Step 5: Draw meaningful contours separately
    # filtered_output = np.zeros_like(img_gray)
    # cv2.drawContours(filtered_output, meaningful_contours, -1, (255), 1)
    # cv2.imshow('Countours', filtered_output)
    # cv2.waitKey(0)
    # Overlay curves on the original color image
    overlay = image_source.copy()
    cv2.drawContours(overlay, meaningful_contours, -1, (0, 255, 0), 2)  # Green lines

    # Showing final edge detection result of the rails
    # cv2.imshow('Countours', overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ----------------Extracting some points from mask -------------------
    # Creating a black image with only the curves of the mask

    black_and_mask = np.zeros_like(image_source)
    cv2.drawContours(black_and_mask, meaningful_contours, -1, (0, 255, 0), 2)
    points = []
    # Trying negative label points outside the main rails to refine sam segmentation
    neg_points = []
    found = True

    #Version 2
    #TODO IDea: trovo prima a parte i due punti della base dele rail, poi trovo i punti dopo scorrendo l'indice su height maggiore e la x tra la x precedente +- 20 px
    #Finding the two points of the rails at the base of the image
    left_rail_base_point = [int(wc/2),hc]
    right_rail_base_point = [int(wc/2),hc]
    found = False
    scanning_height = y + hc
    while found == False:
        for i in range(0,int(wc/2),1):
            if black_and_mask[scanning_height][x+i][1]==255:
                left_rail_base_point[1] = scanning_height
                left_rail_base_point[0] = i
                found = True
                break
    found = False
    while found == False:
        for i in range(wc,int(wc/2),-1):
            if black_and_mask[scanning_height][x+i][1]==255:
                right_rail_base_point[1] = scanning_height
                right_rail_base_point[0] = i
                found = True
                break

    number_of_left_points = 0
    boundaries = numpy.empty(2)
    while found == True:
        found = False
        for i in range(left_rail_base_point[0]-25, left_rail_base_point[0]+25, 1):
            if black_and_mask[scanning_height][x + i][1] == 255:
                boundaries[0]=i
                j=i
                while black_and_mask[scanning_height,x+j][1]==255:
                    j=j+1
                boundaries[1]=j-1
                left_rail_base_point[0]=int((boundaries[0]+boundaries[1])/2)
                left_rail_base_point[1]=scanning_height
                points.append([x+left_rail_base_point[0], left_rail_base_point[1]])
                offset = 4
                neg_points.append([x + i - offset * 10, scanning_height])
                number_of_left_points = number_of_left_points + 1
                found = True
                break
        if scanning_height <= (y + hc / 2):
            found=False
        scanning_height = scanning_height - 25
    number_of_right_points = 0
    scanning_height = y + hc
    found = True
    while found == True:
        found = False
        for i in range(right_rail_base_point[0]+25, right_rail_base_point[0]-25, -1):
            if black_and_mask[scanning_height][x + i][1] == 255:
                boundaries[0] = i
                j = i
                while black_and_mask[scanning_height,x+j][1]==255:
                    j=j-1
                boundaries[1] = j + 1
                right_rail_base_point[0] = int((boundaries[0] + boundaries[1]) / 2)
                right_rail_base_point[1] = scanning_height
                points.append([x+right_rail_base_point[0], right_rail_base_point[1]])
                offset = 4
                neg_points.append([x + i + offset * 10, scanning_height])
                number_of_right_points = number_of_right_points + 1
                found = True
                break
        if scanning_height <= (y + hc / 2):
            found = False
        scanning_height = scanning_height - 25
    #Searching for couples of points to then add middlepoints between rails
    cicles = number_of_left_points
    if number_of_left_points>number_of_right_points:
        cicles = number_of_right_points
    for i in range(cicles):
        height = points[i][1]
        for j in range(number_of_right_points):
            if points[number_of_left_points+j][1] == height:
                width = points[number_of_left_points+j][0] - points[i][0]
                print(width)
                points.append([points[i][0]+int(width/2),height])
                points.append([points[i][0]+int(width/3),height])
                points.append([points[i][0]+int(2*width/3), height])
                break


    #------end vrsion 2
    '''
    scanning_height = y + hc
    while found == False:
        for i in range(0, int(wc / 2), 1):
            if black_and_mask[scanning_height][x + i][1] == 255:  # FIXME ricavare pixel
                points.append([x + i, scanning_height])
                # Adding two extra points around the first to improve segmentation accuracy on the rails
                offset = 4  # FIXME da fare parametrico
                points.append([x + i, scanning_height - offset * 3])
                # points.append([x+i-offset,scanning_height])   #FIXME da erificare se aggiungendo l'offset sbordo dalla box di grounding dino
                points.append([x + i + offset, scanning_height])

                neg_points.append([x + i - offset * 10, scanning_height])
                found = True
                break
        scanning_height = scanning_height - 1
    scanning_height = y + hc
    found = False
    while found == False:
        for i in range(wc, int(wc / 2), -1):
            if black_and_mask[scanning_height][x + i][1] == 255:
                points.append([x + i, scanning_height])
                # Adding two extra points around the first to improve segmentation accuracy on the rails
                offset = 4
                points.append([x + i, scanning_height - offset * 3])
                points.append([x + i - offset, scanning_height])
                # points.append([x + i + offset, scanning_height])

                neg_points.append([x + i + offset * 10, scanning_height])
                found = True
                break
        scanning_height = scanning_height - 1
    # Middle point between the two rails at the base
    # FIXME da verificare se sono state trovate entramnbe le rail, non è detto perche edge highlight potrebbe non funzionare
    # FIXME l'offset verticale deve essere parametrico
    points.append([x + int(wc / 2), y + hc])
    points.append([x + int(wc / 3), y + hc])
    points.append([x + int(wc * 2 / 3), y + hc])
    points.append([x + int(wc / 3), y + hc - 20])
    points.append([x + int(wc / 2), y + hc - 20])
    points.append([x + int(wc * 2 / 3), y + hc - 20])
    points.append([x + int(wc / 2) - int(wc / 4), y + hc - 40])
    points.append([x + int(wc / 2) + int(wc / 4), y + hc - 40])
    points.append([x + int(wc / 2) - int(wc / 4), y + hc - 60])
    points.append([x + int(wc / 2) + int(wc / 4), y + hc - 60])
    points.append([x + int(wc / 2), y + hc - 80])
    '''
    # TODO Da provare a promptare i punti nel dettaglio dei punto appartenenti alla rail e alla carreggiata nel mezzo

    pos_labels = np.ones(len(points))
    neg_labels = np.zeros(len(neg_points))
    labels = np.concatenate((pos_labels, neg_labels))

    points = np.concatenate((points, neg_points))

    IMAGE_COPY = image_source.copy()
    i=0
    for p in points:
        if i < (len(points)-len(neg_points)):
            cv2.circle(IMAGE_COPY, (int(p[0]),int(p[1])), 2, (0, 255, 0),thickness=2)
        else:
            cv2.circle(IMAGE_COPY, (int(p[0]), int(p[1])), 2, (0, 0, 255), thickness=2)
        i = i+1
    cv2.imshow("Visualizing POiNTS", IMAGE_COPY)
    cv2.waitKey(0)

    return points, labels

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
    video_predictor_rails = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", "models/sam2.1/sam2.1_hiera_small.pt", device=device)

    # Load the GroundingDINO model
    groundingdino_checkpoint = config['groundingdino_checkpoint']
    groundingdino_cfg_path = config['groundingdino_cfg_path']
    groundingdino = load_model(groundingdino_cfg_path, groundingdino_checkpoint, device=device)
    TEXT_PROMPT = "object ."
    BOX_TRESHOLD = args.box_threshold
    TEXT_TRESHOLD = args.text_threshold
    #FIXME da scrivere meglio
    BACKGROUND_PROMPT = "one train tracks."
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
    last_masks = {}  # Store the last known mask for each object
    last_masks_rails = {}  # Store the last known mask for each object
    frame_idx = 0
    railway_box = None  # Store railway box for future reference
    main_railway_box = None

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
                    frame_path, groundingdino, "one train tracks. all things.", device, BOX_TRESHOLD=BOX_TRESHOLD_OBSTACLES,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_OBSTACLES
                )

                # Find main railway box and object points
                max_score_railway = 0
                all_obstacles_points = []

                for i, phrase in enumerate(dino_boxes):
                    if phrases[i] == 'one train tracks' and dino_scores[i] > max_score_railway:
                        main_railway_box = dino_boxes[i]
                        max_score_railway = dino_scores[i]
                    elif phrases[i] == 'all things':
                        x_min, y_min, x_max, y_max = dino_boxes[i]
                        all_obstacles_points.append([(x_min + x_max) // 2, (y_min + y_max) // 2])

                print(f"Found main railway: {main_railway_box is not None}, all obstacles: {len(all_obstacles_points)}")

                # Add railway to tracking
                points, labels = extract_main_railway_points_and_labels(frame_rgb,main_railway_box)
                if main_railway_box is not None:
                    _, out_obj_ids, out_mask_logits = video_predictor_rails.add_new_points_or_box(
                        inference_state=inference_state_rails,
                        frame_idx=0,
                        obj_id=ann_all_obj_id,
                        points=points,
                        labels=labels,
                    )

                    # Store railway mask
                    last_masks_rails[ann_all_obj_id] = (out_mask_logits[0] > 0).cpu().numpy()

                # Add detected objects to tracking
                for obj_point in all_obstacles_points:
                    ann_all_obj_id += 1
                    _, out_obj_ids, out_mask_logits = video_predictor_rails.add_new_points_or_box(
                        inference_state=inference_state_rails,
                        frame_idx=0,
                        obj_id=ann_all_obj_id,
                        points=[obj_point],
                        labels=np.array([1], np.int32),
                    )

                    # Store object mask
                    idx = list(out_obj_ids).index(ann_obj_id) if ann_obj_id in out_obj_ids else 0
                    last_masks_rails[ann_obj_id] = (out_mask_logits[idx] > 0).cpu().numpy()


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

                        # Special handling for railway (can use box instead of point)
                        points, labels = extract_main_railway_points_and_labels(frame_rgb,main_railway_box)
                        if obj_id == 1 and main_railway_box is not None:
                            _, _, _ = video_predictor_rails.add_new_points_or_box(
                                inference_state=inference_state_rails,
                                frame_idx=0,
                                obj_id=obj_id,
                                points=points,
                                labels=labels,
                            )
                        else:
                            # Add object using its center point
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
                last_masks_rails[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()

            # Check for new objects periodically
            if (frame_idx % 15 == 0 and frame_idx > 0) or (frame_idx<5 and frame_idx>0):
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
                #FIXME da scrivere meglio
                dino_boxes, phrases, _ = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, OBSTACLE_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_RAILS,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_RAILS
                )

                # Check each detected object
                for i, phrase in enumerate(phrases):
                    if phrase == 'all things':
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

                            # Check if point is inside any existing mask
                            if (0 <= int(center_y) < mask_array.shape[0] and
                                    0 <= int(center_x) < mask_array.shape[1] and
                                    mask_array[int(center_y), int(center_x)]):
                                already_tracked = True
                                break

                        # Add new object if not already tracked
                        if not already_tracked:
                            ann_all_obj_id += 1
                            print(f"New object {ann_all_obj_id} detected at frame {frame_idx}")

                            _, out_obj_ids, out_mask_logits = video_predictor_rails.add_new_points_or_box(
                                inference_state=inference_state_rails,
                                frame_idx=0,
                                obj_id=ann_all_obj_id,
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
                                last_masks_rails[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()

            # Create visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(frame_rgb)
            for obj_id, mask in last_masks_rails.items():
                utility.show_mask_v(mask, plt.gca(), obj_id=obj_id)
            if args.show_frames:  # show the plt image using OpenCV
                cv2.imshow("Processed video frame", utility.plt_figure_to_cv2( plt.gcf()))
                key = cv2.waitKey(1)
                if key == ord('q'):
                    raise KeyboardInterrupt
            if args.save_frames:
                plt.savefig(os.path.join(args.output_path, f"frame_{frame_idx:06d}.jpg"))
            plt.close()

            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"Frame processed in {processing_time:.2f}s")

            # Increment frame counter
            frame_idx += 1

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
