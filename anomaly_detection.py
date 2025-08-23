from groundingdino.util.inference import load_model
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gc
import time
from sam2.build_sam import build_sam2_video_predictor

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

    # add ground thruth path
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        default="./test_video/ground_truth.txt",
        help="Path to ground truth files (three directories: main_railway, safe_obstacles,dangerous_obstacles)"
    )

    # abilitate accuracy testing
    parser.add_argument(
        "--accuracy_test",
        action="store_true",
        default=False,
        help="Test accuracy of the model (default: False)"
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

    video_predictor_rails = build_sam2_video_predictor(sam2_cfg_path, sam2_checkpoint, device=device)

    # Load the GroundingDINO model
    groundingdino_checkpoint = config['groundingdino_checkpoint']
    groundingdino_cfg_path = config['groundingdino_cfg_path']
    groundingdino = load_model(groundingdino_cfg_path, groundingdino_checkpoint, device=device)
    BOX_TRESHOLD = args.box_threshold
    TEXT_TRESHOLD = args.text_threshold

    RAILWAY_PROMPT = "one train track."
    OBSTACLE_PROMPT = ["all objects.","all humans.","all animals."]
    GROUND_PROMPT = "all railways. ground."
    BOX_TRESHOLD_RAILS = 0.25
    TEXT_TRESHOLD_RAILS = 0.15
    BOX_TRESHOLD_OBSTACLES = 0.40 #40
    TEXT_TRESHOLD_OBSTACLES = 0.60  #60
    BOX_TRESHOLD_GROUND = 0.30
    TEXT_TRESHOLD_GROUND = 0.30

    print(f"Loaded SAM2 model from {sam2_checkpoint}")
    print(f"Loaded GroundingDINO model from {groundingdino_checkpoint}")
    print(f'Using box threshold: {BOX_TRESHOLD}, text threshold: {TEXT_TRESHOLD}')
    print(f'Input video: {args.input_video}')
    print(f'Output path: {args.output_path}')

    # Create a temporary directory for frame storage
    temp_dir = os.path.join("temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    temp_main_railway_dir = os.path.join("temp_main_railway")
    temp_safe_obstacles_dir = os.path.join("temp_safe_obstacles")
    temp_dangerous_obstacles_dir = os.path.join("temp_dangerous_obstacles")

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
    ann_rail_id = 0
    last_masks_rails = {}  # Store the last known mask for each object
    frame_idx = 0
    main_railway_box = None
    ground_box = None

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
            inference_state_rails = video_predictor_rails.init_state(video_path=temp_dir)

            # Process based on frame index
            if frame_idx == 0:
                #DETECTION OF THE GROUND
                dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, GROUND_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_GROUND,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_GROUND, show=False,
                )

                max_score_railway = 0
                for i, box in enumerate(dino_boxes):
                 if dino_scores[i] > max_score_railway:
                        ground_box = box
                        max_score_railway = dino_scores[i]

                #DETECTION OF THE MAIN RAILWAY AND THE OBSTACLES
                dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, RAILWAY_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_RAILS,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_RAILS, show=False,
                )

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
                        if dino_box_width >= int(0.75*width):
                            if dino_abs_distance_from_center < int(0.25*width):
                                if dino_scores[i] > max_score_railway:
                                    main_railway_box = box
                                    max_score_railway = dino_scores[i]
                all_obstacles_points = []
                #---------prova di fare più detection singole per ogni parola--
                for class_name in OBSTACLE_PROMPT:
                    dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                        frame_path, groundingdino, class_name, device, BOX_TRESHOLD=BOX_TRESHOLD_OBSTACLES,
                        TEXT_TRESHOLD=TEXT_TRESHOLD_OBSTACLES, show=False,
                    )  # TODO Rimuovere lo show=true

                    for i, box in enumerate(dino_boxes):
                        x_min, y_min, x_max, y_max = box
                        x_center = (x_min + x_max) // 2
                        y_center = (y_min + y_max) // 2
                        if (x_center, y_center) not in all_obstacles_points:
                            all_obstacles_points.append([x_center, y_center])

                print(f"Found ground: {ground_box is not None}, Found main railway: {main_railway_box is not None}, all obstacles: {len(all_obstacles_points)}")

                ann_rail_id += 1
                # Add railway to tracking
                if main_railway_box is not None:
                    points, labels = utility.extract_main_internal_railway_points_and_labels(frame_rgb, main_railway_box,last_masks_rails)

                    _, out_obj_ids, out_mask_logits = video_predictor_rails.add_new_points_or_box(
                        inference_state=inference_state_rails,
                        frame_idx=0,
                        obj_id=ann_rail_id,
                        points=points,
                        labels=labels,
                        box=main_railway_box,
                    )

                    # Store railway mask
                    last_masks_rails[ann_rail_id] = utility.refine_mask((out_mask_logits[0] > 0).cpu().numpy())
                    #last_masks_rails[ann_rail_id] = (out_mask_logits[0] > 0).cpu().numpy()

                # Add detected objects to tracking
                for obj_point in all_obstacles_points:
                    if utility.is_point_inside_box(obj_point, ground_box):
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
                        temp_mask = (out_mask_logits[idx] > 0).cpu().numpy()
                        if utility.is_mask_an_obstacle(temp_mask, last_masks_rails[1],ground_box):
                            last_masks_rails[ann_rail_id] = temp_mask
                        else:
                            video_predictor_rails.remove_object(inference_state_rails, ann_rail_id)

            else:
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
                            points, labels = utility.extract_main_internal_railway_points_and_labels(frame_rgb,main_railway_box,last_masks_rails)
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

            # Propagate all objects in current frame
            result_rails = next(video_predictor_rails.propagate_in_video(
                inference_state_rails,
                start_frame_idx=0
            ))
            _, out_obj_ids, out_mask_logits = result_rails


            # Update all masks for next frame
            for i, obj_id in enumerate(out_obj_ids):
                if obj_id == 1:
                    last_masks_rails[obj_id] = utility.refine_mask((out_mask_logits[i] > 0).cpu().numpy(),last_masks_rails[obj_id])
                    #last_masks_rails[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()
                else:
                    last_masks_rails[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()

            # Check for new objects periodically
            if (frame_idx % 15 == 0 and frame_idx > 0):
                # DETECTION OF THE GROUND
                dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, GROUND_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_GROUND,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_GROUND, show=False,
                )

                max_score_railway = 0
                for i, box in enumerate(dino_boxes):
                    if dino_scores[i] > max_score_railway:
                        ground_box = box
                        max_score_railway = dino_scores[i]

                # DETECTION OF THE MAIN RAILWAY AND THE OBSTACLES   #TODO da mettere in un metodo perchè viene fatto lo stesso codice più in alto
                dino_boxes, phrases, dino_scores = utility.grounding_Dino_analyzer(
                    frame_path, groundingdino, RAILWAY_PROMPT, device, BOX_TRESHOLD=BOX_TRESHOLD_RAILS,
                    TEXT_TRESHOLD=TEXT_TRESHOLD_RAILS, show=False,
                )

                # Find main railway box and object points
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

                dino_boxes = []
                phrases = []
                for class_name in OBSTACLE_PROMPT:
                    t_dino_boxes, t_phrases, _ = utility.grounding_Dino_analyzer(
                        frame_path, groundingdino, class_name, device, BOX_TRESHOLD=BOX_TRESHOLD_OBSTACLES,
                        TEXT_TRESHOLD=TEXT_TRESHOLD_OBSTACLES, show = False,
                    )
                    i=0
                    for box in t_dino_boxes:
                        if all(not np.array_equal(box, item) for item in dino_boxes):
                            dino_boxes.append(box)
                            phrases.append(t_phrases[i])
                        i+=1

                # Check each detected object
                for i, phrase in enumerate(phrases):
                    if phrases[i] != 'one train track':
                        x_min, y_min, x_max, y_max = dino_boxes[i]
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2

                        # Check if this object is already tracked
                        already_tracked = False
                        for mask in last_masks_rails.values():
                            mask = np.array(mask, dtype=np.uint8)
                            mask = mask.squeeze()
                            if mask[int(center_y), int(center_x)] > 0:
                                already_tracked = True
                                break

                        # Add new object if not already tracked and is inside the railway area
                        if (not already_tracked) and utility.is_point_inside_box( [center_x,center_y],ground_box):
                            ann_rail_id += 1
                            print(f"New object {ann_rail_id} detected at frame {frame_idx}")

                            _, out_obj_ids, out_mask_logits = video_predictor_rails.add_new_points_or_box(
                                inference_state=inference_state_rails,
                                frame_idx=0,
                                obj_id=ann_rail_id,
                                points=[[center_x, center_y]],
                                labels=np.array([1], np.int32),
                            )

                            if not utility.is_mask_an_obstacle((out_mask_logits[i] > 0).cpu().numpy(), last_masks_rails[1],ground_box):
                                video_predictor_rails.remove_object(inference_state_rails, ann_rail_id)

                            # Re-propagate with the new object
                            result_rails = next(video_predictor_rails.propagate_in_video(
                                inference_state_rails,
                                start_frame_idx=0
                            ))
                            _, out_obj_ids, out_mask_logits = result_rails

                            # Update masks dictionary
                            for i, obj_id in enumerate(out_obj_ids):
                                if obj_id == 1:
                                    last_masks_rails[obj_id] = utility.refine_mask((out_mask_logits[i] > 0).cpu().numpy(),last_masks_rails[obj_id])
                                    #last_masks_rails[obj_id]=(out_mask_logits[i] > 0).cpu().numpy()
                                else:
                                    last_masks_rails[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()

            #Removal of tracked masks that probably are not obstacles, but the same railway or other geometries in the image
            idx_to_pop = []
            #for obj_id, mask in last_masks_rails.items():
            obj_id=1
            while obj_id<=len(last_masks_rails):
                mask = last_masks_rails[obj_id]
                if  obj_id!=1 and ((not utility.is_mask_an_obstacle(mask, last_masks_rails[1], ground_box)) or utility.is_mask_duplicate(mask, obj_id, last_masks_rails)):
                    video_predictor_rails.remove_object(inference_state_rails, obj_id)
                    last_masks_rails.pop(obj_id)

                    new_d = {}
                    for key in sorted(last_masks_rails.keys()):
                        if key < obj_id:
                            new_d[key] = last_masks_rails[key]
                        elif key > obj_id:
                            new_d[key - 1] = last_masks_rails[key]
                        # key == k is dropped
                    last_masks_rails = new_d
                obj_id+=1

                    #last_masks_rails[obj_id] = np.zeros_like(mask)
                     #for i in range(obj_id, len(last_masks_rails),1):
                        #if i!=len(last_masks_rails)-1:
                            #last_masks_rails[i] = last_masks_rails[i+1]
                        #else:
                        #    last_masks_rails[i] = np.zeros_like(mask)
            #for idx in idx_to_pop:
            #    last_masks_rails.

            # Create visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(frame_rgb)
            rail_mask = None
            if True:  # accuracy_test
                os.makedirs(temp_main_railway_dir, exist_ok=True)
                os.makedirs(temp_safe_obstacles_dir, exist_ok=True)
                os.makedirs(temp_dangerous_obstacles_dir, exist_ok=True)
            for obj_id, mask in last_masks_rails.items():
                #utility.show_mask_v(mask, plt.gca(), obj_id=obj_id)
                if obj_id != 1 and obj_id != 0:
                    utility.show_anomalies(mask,plt.gca(),rail_mask, True , obj_id,frame_idx) #FIXME al posto di True ci devo mettere args.accuracy_test
                else:
                    utility.show_mask_v(mask, plt.gca(), True, frame_idx, obj_id=obj_id)#FIXME al posto di True ci devo mettere args.accuracy_test
                    rail_mask = mask
            if True:  # show the plt image using OpenCV     args.show_frames
                cv2.imshow("Processed video frame", utility.plt_figure_to_cv2( plt.gcf()))
                key = cv2.waitKey(1)
                if key == ord('q'):
                    raise KeyboardInterrupt
            if False:  #to remove True in args.save_frames
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

        if True:#FIXME al posto di True ci devo mettere args.accuracy_test
            utility.calculate_accuracy(frame_idx,temp_main_railway_dir, temp_safe_obstacles_dir, temp_dangerous_obstacles_dir)
        # Release resources
        cap.release()

        # Clean up temp directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        for file in os.listdir(temp_main_railway_dir):
            os.remove(os.path.join(temp_main_railway_dir, file))
        os.rmdir(temp_main_railway_dir)
        for file in os.listdir(temp_safe_obstacles_dir):
            os.remove(os.path.join(temp_safe_obstacles_dir, file))
        os.rmdir(temp_safe_obstacles_dir)
        for file in os.listdir(temp_dangerous_obstacles_dir):
            os.remove(os.path.join(temp_dangerous_obstacles_dir, file))
        os.rmdir(temp_dangerous_obstacles_dir)

        if args.show_frames:
            cv2.destroyAllWindows()

        print(f"Processing completed or interrupted after {frame_idx} frames")

if __name__ == "__main__":
    main()
