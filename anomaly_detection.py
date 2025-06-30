import numpy
from PIL.ImageChops import offset
from groundingdino.util.inference import load_model, load_image
from scipy.interpolate import interp1d, make_interp_spline, splprep, splev
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

def extract_main_railway_points_and_labels(image_source, gd_boxes, frame_index):
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
    meaningful_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > 300]

    # Step 5: Draw meaningful contours separately
    # filtered_output = np.zeros_like(img_gray)
    # cv2.drawContours(filtered_output, meaningful_contours, -1, (255), 1)
    # cv2.imshow('Countours', filtered_output)
    # cv2.waitKey(0)
    # Overlay curves on the original color image
    overlay = image_source.copy()
    cv2.drawContours(overlay, meaningful_contours, -1, (0, 255, 0), 2)  # Green lines

    # Showing final edge detection result of the rails
    if frame_index == 40 or frame_index == 0:
        cv2.imshow('Countours', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ----------------Extracting some points from mask -------------------
    # Creating a black image with only the curves of the mask

    black_and_mask = np.zeros_like(image_source)
    cv2.drawContours(black_and_mask, meaningful_contours, -1, (0, 255, 0), 2)
    points = []
    # Trying negative label points outside the main rails to refine sam segmentation
    neg_points = []
    found = True

    #Version 2
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
                points.append([points[i][0]+int(width/2),height])
                points.append([points[i][0]+int(width/3),height])
                points.append([points[i][0]+int(2*width/3), height])
                points.append([points[i][0] + int(width / 5), height])
                points.append([points[i][0] + int(4 * width / 5), height])
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

    #TODO Calcolare la prospettiva dell'immagine, sapendo il parallelismo dei due binari e langolo che hanno/un computer vision che lo capisce da solo
    #TODO Magari riesco a risolvere il problema legato al fatto che prende più binari usando la prospettiva


    pos_labels = np.ones(len(points))
    neg_labels = np.zeros(len(neg_points))
    labels = np.concatenate((pos_labels, neg_labels))

    points = np.concatenate((points, neg_points))
    bl_point = neg_points[0]
    br_point = neg_points[0]
    tl_point = neg_points[0]
    tr_point = neg_points[0]
    min_height = 1000
    max_height = 0
    top_points = []
    bottom_points = []
    for i in range(len(neg_points)):
        if neg_points[i][1] < min_height:
            min_height = neg_points[i][1]
        elif neg_points[i][1] > max_height:
            max_height = neg_points[i][1]

    for i in range(len(neg_points)):
        if neg_points[i][1] == min_height:
            top_points.append(neg_points[i])
        elif neg_points[i][1] == max_height:
            bottom_points.append(neg_points[i])

    min_x = 1000
    max_x = 0
    index_tl = 0
    index_tr = 0
    for i in range(len(top_points)):
        if top_points[i][0] < min_x:
            min_x = top_points[i][0]
            index_tl = i
        elif top_points[i][0] > max_x:
            max_x = top_points[i][0]
            index_tr = i
    tr_point = top_points[index_tr]
    tl_point = top_points[index_tl]

    min_x = 1000
    max_x = 0
    index_bl = 0
    index_br = 0
    for i in range(len(bottom_points)):
        if bottom_points[i][0] < min_x:
            min_x = bottom_points[i][0]
            index_bl = i
        elif bottom_points[i][0] > max_x:
            max_x = bottom_points[i][0]
            index_br = i
    br_point = bottom_points[index_br]
    bl_point = bottom_points[index_bl]

    print(bl_point, br_point, tl_point, tr_point)

    #Perspective trensformation
    starting_points = np.array([
        tl_point,
        tr_point,
        bl_point,
        br_point
    ], dtype=np.float32)

    T_width = int(br_point[0] - bl_point[0])
    T_height = int(tl_point[1] - bl_point[1])

    ending_points = np.array([
        [0, 0],
        [T_width, 0],
        [0, T_height],
        [T_width, T_height]
    ], dtype=np.float32)

    # Compute transformation matrix (THIS is the correct function)
    T = cv2.getPerspectiveTransform(starting_points, ending_points)

    # Convert image
    image_to_trasf = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

    # Apply the perspective warp
    imgTrans = cv2.warpPerspective(image_to_trasf, T, (T_width, T_height))

    # Display result
    plt.figure()
    plt.imshow(imgTrans)
    plt.axis("off")
    plt.show()

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

def media_robusta(valori, epsilon=1e-6):
    mediana = np.median(valori)
    distanze = [abs(x - mediana) + epsilon for x in valori]
    pesi = [1 / d for d in distanze]
    somma_pesi = sum(pesi)
    pesi_normalizzati = [p / somma_pesi for p in pesi]
    return sum(x * w for x, w in zip(valori, pesi_normalizzati))

def extract_main_internal_railway_points_and_labels(image_source, gd_box, rails_masks):
    #TODO Come fare per mantenere stabile la forma della mschera dei binari
    #Provo a fare la media tra: metà dell'iimagine, metà della gd box, metà tra i binari di canny

    width = image_source.shape[1]
    height = image_source.shape[0]
    image_midde_point = int(width / 2)
    gd_width = gd_box[2]-gd_box[0]
    gd_height = gd_box[3]-gd_box[1]
    x = gd_box[0]
    y = gd_box[1]
    gdbox_midde_point = x+int(gd_width / 2)

    # -----------------CANNY EDGES--------------------
    # Convert to graycsale
    image_cropped = image_source[int(gd_box[1]):int(gd_box[3]), int(gd_box[0]):int(gd_box[2])]

    img_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=60, threshold2=90)  # Canny Edge Detection
    #Display Canny Edge Detection Image
    #cv2.imshow('Canny Edge Detection', edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

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

    # Step 3: Find contours from the edge image
    contours, hierarchy = cv2.findContours(blacked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Draw contours on a blank canvas (for visualization)
    output = np.zeros_like(blacked_image)
    cv2.drawContours(output, contours, -1, (255), 1)

    # Optional: Filter meaningful curves by length or area
    meaningful_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > 200]

    # Step 5: Draw meaningful contours separately
    # filtered_output = np.zeros_like(img_gray)
    # cv2.drawContours(filtered_output, meaningful_contours, -1, (255), 1)
    # cv2.imshow('Countours', filtered_output)
    # cv2.waitKey(0)
    # Overlay curves on the original color image
    overlay = image_source.copy()
    cv2.drawContours(overlay, meaningful_contours, -1, (0, 255, 0), 2)  # Green lines

    # Showing final edge detection result of the rails
    #cv2.imshow('Countours', overlay)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    black_and_mask = np.zeros_like(image_source)
    cv2.drawContours(black_and_mask, meaningful_contours, -1, (0, 255, 0), 2)

    #From mask of the railway obtaining the points to segment

    mask_image = None

    for obj_id, mask in rails_masks.items():
        h, w = mask.shape[-2:]
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        break

    '''
    x_mask_points = np.array([])
    y_mask_points = np.array([])
    mask_middle_points = np.array([])
    avg_array = np.array([])
    y_of_avg_array = np.array([])
    for j in range(y+hc,y,-3):
        for i in range(x,x+wc,3):
            if black_and_mask[j][i][1] == 255:
                x_mask_points = np.append(x_mask_points, [i])
        if len(x_mask_points) > 0:
            avg_x = int(sum(x_mask_points) / len(x_mask_points))
            mask_middle_points = np.append(mask_middle_points, [avg_x,j])
            avg_array = np.append(avg_array, avg_x)
            y_of_avg_array = np.append(y_of_avg_array, j)
        x_mask_points = np.array([])

    # Reshape the flat array into (N, 2)
    curve_points = mask_middle_points.reshape((-1, 2))

    # Convert to int32 (required by OpenCV)
    curve_points = np.round(curve_points).astype(np.int32)

    # Reshape to (N, 1, 2) as required by cv2.polylines
    curve_points = curve_points.reshape((-1, 1, 2))

    image_copy = np.copy(image_source)
    # Draw the curve
    cv2.polylines(image_copy, [curve_points], isClosed=False, color=(0, 255, 0), thickness=2)

    # Show image
    cv2.imshow("Image with Curve", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    xhat = savgol_filter(avg_array, 51, 3)
    savol_array = []
    x_avg_array_filtered = np.array(xhat)
    for i in range(len(y_of_avg_array)):
        savol_array.append([x_avg_array_filtered[i],y_of_avg_array[i]])

    curve = np.array(savol_array, dtype=np.float32)
    curve = np.round(curve).astype(np.int32)

    # Reshape for cv2.polylines: (N, 1, 2)
    curve = curve.reshape((-1, 1, 2))

    image_copy = np.copy(image_source)
    # Draw the curve
    cv2.polylines(image_copy, [curve], isClosed=False, color=(0, 255, 0), thickness=2)

    # Show image
    cv2.imshow("Image with savol_array", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # TODO Le strisce degli edge potrebbero non essere abbastanza lunghe per superare il limite minimo di 300 (per esempio il binario è interrotto da un ostacolo) o l'edge potrebbe fondersi ad un altro elemento (tipo un ostacolo)

    # ----------------Extracting some points from mask -------------------
    # Creating a black image with only the curves of the mask

    black_and_mask = np.zeros_like(image_source)
    cv2.drawContours(black_and_mask, meaningful_contours, -1, (0, 255, 0), 2)
    points = []
    # Trying negative label points outside the main rails to refine sam segmentation
    neg_points = []
    found = True

    if mask_image is None:
        left_rail_base_point = [int(wc / 2), hc]
        right_rail_base_point = [int(wc / 2), hc]
        foundLeft = False
        scanning_height = y + hc
        #TODO invece di cercare solo due pixel alla base forse è meglio prendere tutta una riga e fare la deia dei pixel degli edge

        average_points = []
        average_levels = []
        for j in range(y+hc,y+hc-int(0.03*y+hc),-1):
            for i in range(x,x+wc,3):
                if black_and_mask[j][i][1] == 255:
                    average_points.append(i)
            if len(average_points) > 0:
                average_levels.append(int(sum(average_points) / len(average_points)))

        '''
        while foundLeft == False and scanning_height > y+hc-20:      #TODO rendere il "20" pixel parametrico
            for i in range(0, int(wc / 2), 1):
                if black_and_mask[scanning_height][x + i][1] == 255:
                    left_rail_base_point[1] = scanning_height
                    left_rail_base_point[0] = i
                    foundLeft = True
                    break
            scanning_height = scanning_height - 1

        foundRight = False
        while foundRight == False and scanning_height > y+hc-20:
            for i in range(wc, int(wc / 2), -1):
                if black_and_mask[scanning_height][x + i][1] == 255:  # FIXME il problema che si blocca è circa qua
                    right_rail_base_point[1] = scanning_height
                    right_rail_base_point[0] = i
                    foundRight = True
                    break
            scanning_height = scanning_height - 1
        '''

        #if foundLeft and foundRight:
        if len(average_levels) > 0:
            #edges_rails_midde_point = x + int((left_rail_base_point[0] + right_rail_base_point[0]) / 2)
            edges_rails_midde_point = int(sum(average_points) / len(average_points))

            #Faccio la media dei tre centri calcolati sopra
            #average_midde_point = media_robusta([gdbox_midde_point, edges_rails_midde_point, image_midde_point])
            average_midde_point = int((gdbox_midde_point + image_midde_point + edges_rails_midde_point) / 3)
        else:
            #average_midde_point = media_robusta([gdbox_midde_point, image_midde_point])
            print(gdbox_midde_point, image_midde_point)
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
    else:
        x_mask_points = np.array([])
        y_mask_points = np.array([])
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

        # Reshape the flat array into (N, 2)
        curve_points = mask_middle_points.reshape((-1, 2))

        # Convert to int32 (required by OpenCV)
        curve_points = np.round(curve_points).astype(np.int32)

        # Reshape to (N, 1, 2) as required by cv2.polylines
        curve_points = curve_points.reshape((-1, 1, 2))

        image_copy = np.copy(image_source)
        # Draw the curve
        cv2.polylines(image_copy, [curve_points], isClosed=False, color=(0, 255, 0), thickness=2)

        # Show image pre filtering
        #cv2.imshow("Image with Curve", image_copy)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        xhat = savgol_filter(avg_array, 10, 3)
        savol_array = []
        x_avg_array_filtered = np.array(xhat)
        for i in range(len(y_of_avg_array)):
            savol_array.append([x_avg_array_filtered[i], y_of_avg_array[i]])

        curve = np.array(savol_array, dtype=np.float32)
        curve = np.round(curve).astype(np.int32)

        # Reshape for cv2.polylines: (N, 1, 2)
        curve = curve.reshape((-1, 1, 2))

        image_copy = np.copy(image_source)
        # Draw the curve
        cv2.polylines(image_copy, [curve], isClosed=False, color=(0, 255, 0), thickness=2)

        # Show image after savol filter
        #cv2.imshow("Image with savol_array", image_copy)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #prendo il savola array e ne aggiungo una copia a destra e sinistra e tolgo gli estremi
        savol_array_left = savol_array.copy()
        savol_array_right = savol_array.copy()
        savol_array_expanded = []
        for i in range(len(savol_array)):
            if i>0 and i<len(savol_array_left)-1 and (i%6) == 0:
                #savol_array_expanded.append(savol_array[i])
                current_y_in_rail_box = savol_array[i][1] - y
                savol_array_expanded.append([savol_array[i][0]-int(current_y_in_rail_box*0.12), savol_array[i][1]])
                savol_array_expanded.append([savol_array[i][0]+int(current_y_in_rail_box*0.12), savol_array[i][1]])
        points = np.array(savol_array_expanded)

    #Diplaying points
    #IMAGE_COPY = image_source.copy()
    #i = 0
    #for p in points:
    #    cv2.circle(IMAGE_COPY, (int(p[0]), int(p[1])), 2, (0, 255, 0), thickness=2)
    #    i = i + 1
    #cv2.imshow("Visualizing POINTS", IMAGE_COPY)
    #cv2.waitKey(0)

    labels = np.ones(len(points))
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
                )#TODO Rimuovere lo show=true

                # Find main railway box and object points
                max_score_railway = 0
                main_box_area = 0
                #FIXME come faccio a fare che prenda solo il binario centrale, perchè alcune volte prende tutti i binari
                #for i,box in enumerate(dino_boxes):
                #    if dino_scores[i] > max_score_railway:
                #        main_railway_box = box
                #        max_score_railway = dino_scores[i]

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
                    last_masks_rails[ann_rail_id] = (out_mask_logits[0] > 0).cpu().numpy()

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
                last_masks_rails[obj_id] = (out_mask_logits[i] > 0).cpu().numpy()

            # Check for new objects periodically
            if (frame_idx % 15 == 0 and frame_idx > 0) or (frame_idx > 0 and frame_idx < 3):
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
            if True:  #to remove True in args.save_frames
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
