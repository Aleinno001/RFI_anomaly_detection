import numpy as np
from groundingdino.util.inference import load_model
from matplotlib import pyplot as plt
from ultralytics import SAM
from torchvision.ops import box_convert
import torchvision.transforms as T
from ultralytics.models.sam import SAM2VideoPredictor
import os
import torch
import gc
import time
import utility
from groundingdino.util.inference import load_image, predict, annotate
from gpu_utility import set_device
import cv2


#main
def main():
    device = set_device()
    print(f"using device: {device}")

    # Directories of input for the raw video, of output for the processed images and others
    VIDEO_INPUT_DIR = "./test_video"
    VIDEO_OUTPUT_DIR = "./test_video/output"
    VIDEO_INPUT = os.path.join(VIDEO_INPUT_DIR, "test_donato.mp4")


    # Create output directory if it doesn't exist
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    groundedModel = load_model("configs/grounding_dino/GroundingDINO_SwinT_OGC.py",
                       "models/grounding_dino/groundingdino_swint_ogc.pth", device.type)
    # Create SAM2VideoPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="./models/sam2.1/sam2.1_s.pt")
    samPredictor = SAM2VideoPredictor(overrides=overrides)

    samModel = SAM("./models/sam2.1/sam2.1_s.pt")       #Small è il più affidabile e capisce meglio quale sia il binario completo
    overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-s.pt", save=False, imgsz=1024)

    BACKGROUND_PROMPT = "one train tracks."
    OBSTACLE_PROMPT = "all things ."
    BOX_TRESHOLD = 0.25
    TEXT_TRESHOLD = 0.15

    last_masks = {}  # Store the last known mask for each object

    # Create a temporary directory for frame storage
    temp_dir = os.path.join("temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Open the video stream
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        raise Exception(f"Could not open video stream: {VIDEO_INPUT}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video stream at {fps} FPS, resolution: {width}x{height}")

    frame_idx = 0

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

            #samPredictor.set_image(frame_path)
            #inference_state = samPredictor.init_state(samPredictor)



            if True:        #FIXME frame_index == 0
                print("Analysis with Grounding Dino")
                image_source, gd_image = load_image(frame_path)


                # First frame: detect railway and objects with Grounding DINO
                gd_boxes, logits, phrases = predict(
                    model=groundedModel,
                    image=gd_image,
                    caption=BACKGROUND_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD,
                    device=device,
                )
                # Annotate for Grounding Dino track finding
                #annotated_frame = annotate(image_source=image_source, boxes=gd_boxes, logits=logits, phrases=phrases)
                #print(phrases, logits)

                #cv2.imshow("Visualizing results of track detection", annotated_frame)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                w = image_source.shape[1]
                h = image_source.shape[0]
                gd_boxes = gd_boxes * torch.Tensor([w, h, w, h])
                gd_boxes = box_convert(boxes=gd_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                #-----------------CANNY EDGES--------------------
                # Convert to graycsale
                image_cropped = image_source[int(gd_boxes[0][1]):int(gd_boxes[0][3]), int(gd_boxes[0][0]):int(gd_boxes[0][2])]

                img_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

                # Blur the image for better edge detection
                img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

                # Canny Edge Detection
                edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=90)  # Canny Edge Detection
                # Display Canny Edge Detection Image
                #cv2.imshow('Canny Edge Detection', edges)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                # -----------------HUGS EDGES--------------------
                x = int(gd_boxes[0][0])
                y = int(gd_boxes[0][1])
                hc = image_cropped.shape[0]
                wc = image_cropped.shape[1]

                #Overlaps the edge detection of the cropped GD detection box onto the fullsize image, full black so that there are no other contours except the ones of Canny
                blacked_image = np.zeros_like(image_source)
                blacked_image = cv2.cvtColor(blacked_image, cv2.COLOR_BGR2GRAY)
                blacked_image[y:y+hc,x:x+wc] = edges

                # Step 3: Find contours from the edge image
                contours, hierarchy = cv2.findContours(blacked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Step 4: Draw contours on a blank canvas (for visualization)
                output = np.zeros_like(blacked_image)
                cv2.drawContours(output, contours, -1, (255), 1)

                # Optional: Filter meaningful curves by length or area
                meaningful_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > 400]

                # Step 5: Draw meaningful contours separately
                #filtered_output = np.zeros_like(img_gray)
                #cv2.drawContours(filtered_output, meaningful_contours, -1, (255), 1)
                #cv2.imshow('Countours', filtered_output)
                #cv2.waitKey(0)
                # Overlay curves on the original color image
                overlay = image_source.copy()
                cv2.drawContours(overlay, meaningful_contours, -1, (0, 255, 0), 2)  # Green lines

                #Showing final edge detection result of the rails
                #cv2.imshow('Countours', overlay)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #----------------Extracting some points from mask -------------------
                #Creating a black image with only the curves of the mask

                black_and_mask = np.zeros_like(image_source)
                cv2.drawContours(black_and_mask, meaningful_contours, -1, (0, 255, 0), 2)
                points = []
                # Trying negative label points outside the main rails to refine sam segmentation
                neg_points = []
                found = False

                scanning_height = y+hc
                while found == False:
                    for i in range(0,int(wc/2),1):
                        if black_and_mask[scanning_height][x+i][1]==255:      #FIXME ricavare pixel
                            points.append([x+i,scanning_height])
                            #Adding two extra points around the first to improve segmentation accuracy on the rails
                            offset = 4  #FIXME da fare parametrico
                            points.append([x+i,scanning_height-offset*3])
                            #points.append([x+i-offset,scanning_height])   #FIXME da erificare se aggiungendo l'offset sbordo dalla box di grounding dino
                            points.append([x+i+offset,scanning_height])

                            neg_points.append([x+i-offset*10,scanning_height])
                            found = True
                            break
                    scanning_height = scanning_height - 1
                scanning_height = y + hc
                found = False
                while found == False:
                    for i in range(wc,int(wc/2),-1):
                        if black_and_mask[scanning_height][x+i][1]==255:
                            points.append([x+i,scanning_height])
                            # Adding two extra points around the first to improve segmentation accuracy on the rails
                            offset = 4
                            points.append([x + i, scanning_height - offset*3])
                            points.append([x + i - offset, scanning_height])
                            #points.append([x + i + offset, scanning_height])

                            neg_points.append([x + i + offset * 10, scanning_height])
                            found = True
                            break
                    scanning_height = scanning_height -1
                #Middle point between the two rails at the base
                #FIXME da verificare se sono state trovate entramnbe le rail, non è detto perche edge highlight potrebbe non funzionare
                #FIXME l'offset verticale deve essere parametrico
                points.append([x+int(wc/2),y+hc])
                points.append([x+int(wc/3),y+hc])
                points.append([x+int(wc*2/3), y + hc])
                points.append([x + int(wc/3), y + hc - 20])
                points.append([x + int(wc/2), y + hc - 20])
                points.append([x+int(wc*2/3), y + hc - 20])
                points.append([x + int(wc / 2) - int(wc/4), y + hc - 40])
                points.append([x + int(wc / 2) +int( wc/4), y + hc - 40])



                #TODO Da provare a promptare i punti nel dettaglio dei punto appartenenti alla rail e alla carreggiata nel mezzo

                pos_labels = np.ones(len(points))
                neg_labels = np.zeros(len(neg_points))
                labels = np.concatenate((pos_labels,neg_labels))

                points = np.concatenate((points,neg_points))
                #results = samModel(image_source,points=points,labels=labels)

                results = samModel.predict(
                    source=image_source,
                    points=[points],  # Wrap the full list in another list: shape (1, N, 2)
                    labels=[labels],  # Same: shape (1, N)
                )

                #Sam2.1 video predictor test
                '''
                if gd_boxes is not None:
                    _, out_obj_ids, out_mask_logits = samPredictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=0,
                        points=[points],
                    )
                    # Store railway mask
                    last_masks[0] = (out_mask_logits[0] > 0).cpu().numpy()
                '''

                #----Grounded DINO for detection of EVERITHING alias obstacles------
                gd_boxes, logits, phrases = predict(
                    model=groundedModel,
                    image=gd_image,
                    caption=OBSTACLE_PROMPT,
                    box_threshold=0.20,
                    text_threshold=0.16,
                    device=device,
                )
                # Annotate for Grounding Dino track finding
                #annotated_frame = annotate(image_source=image_source, boxes=gd_boxes, logits=logits, phrases=phrases)
                #cv2.imshow("Visualizing results of track detection", annotated_frame)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                annotated_image = annotate(image_source=results[0].plot(), boxes=gd_boxes, logits=logits, phrases=phrases)
                # print(phrases, logits)



                #Visualizing points for sam2.1 segmentation
                #IMAGE_COPY = image_source.copy()
                #i=0
                #for p in points:
                #    if i < (len(points)-len(neg_points)):
                #        cv2.circle(IMAGE_COPY, (int(p[0]),int(p[1])), 2, (0, 255, 0),thickness=2)
                #    else:
                #        cv2.circle(IMAGE_COPY, (int(p[0]), int(p[1])), 2, (0, 0, 255), thickness=2)
                #    i = i+1
                #cv2.imshow("Visualizing POiNTS", IMAGE_COPY)
                #cv2.waitKey(0)


                '''
                names = ["train rails","train tracks","all train tracks","all train rails","tracks"]
                yoloeModel.set_classes(names, yoloeModel.get_text_pe(names))
                result = yoloeModel.predict(image_source)
                '''
                #result = samModel(image_source, bboxes = gd_boxes)

                #result[0].show()

                #cv2.imshow("Visualizing results", result[0].plot())
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                '''
                gd_boxes, logits, phrases = predict(
                    model=groundedModel,
                    image=gd_image,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD,
                    device=device,
                )
                annotated_frame = annotate(image_source=overlay, boxes=gd_boxes, logits=logits, phrases=phrases)

                #cv2.destroyAllWindows()
                #cv2.imshow("Visualizing results of obstacle detection", annotated_frame)
                #cv2.waitKey(100)
                #cv2.waitKey(0)
                '''

                #everything_results = predictor(frame_path)
                #everything_results[0].show()

                #results = samModel(image_source)
                #results[0].show()

                #results = mobsamModel(image_source)

                frame_idx = frame_idx + 1


            else:
                print("Analysis in inference and prediction")


            plt.figure(figsize=(8, 6))
            #plt.imshow(frame_rgb)
            plt.imshow(annotated_image)
            # plt.imshow(results[0].plot())
            for obj_id, mask in last_masks.items():
                utility.show_mask_v(mask, plt.gca(), obj_id=obj_id)
            if True:  # show the plt image using OpenCV
                cv2.imshow("Processed video frame", utility.plt_figure_to_cv2(plt.gcf()))
                key = cv2.waitKey(1)
                # cv2.waitKey(0)
                if key == ord('q'):
                    raise KeyboardInterrupt
            if True:    #FIXME da cambiare il true
                plt.savefig(os.path.join(VIDEO_OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg"))
            plt.close()

            # Clear memory for next iteration
            #del inference_state
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

        #FIXME cambiare true con una variabile
        if True:
            cv2.destroyAllWindows()

        print(f"Processing completed or interrupted after {frame_idx} frames")

if __name__ == "__main__":
    main()
