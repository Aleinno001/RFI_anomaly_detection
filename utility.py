import os

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from torchvision.ops import box_convert

from groundingdino.util.inference import load_image, predict, annotate
from matplotlib import pyplot as plt


## PLOTTING & SAVING

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_mask_v(mask, ax, obj_id=None, random_color=False, frame_name=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx + 1)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.axis('off')
    # plt.savefig(f"./seg_frames_2/{str(frame_name)}.jpg", bbox_inches='tight', pad_inches=0)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(imag, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True,
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
            show_box(box_coords, plt.gca())
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


def plt_figure_to_cv2(figure):
    """Convert a matplotlib figure to an OpenCV image."""
    # Draw the figure first
    figure.canvas.draw()

    # Get the ARGB buffer from the figure
    buf = figure.canvas.tostring_argb()
    w, h = figure.canvas.get_width_height()

    # Convert ARGB buffer to numpy array
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

    # Convert ARGB to RGB
    img = img[:, :, 1:]  # Remove alpha channel

    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def produce_video(frame_folder, output_video, fps):
    '''

    :param frame_folder: the folder with the frames of the video to produce
    :param output_video: the name of the output video with the extension .mp4
    :param fps: the integer representing the frames per second

    '''
    # Configura i percorsi
    # frame_folder = "./seg_frames_2"  # Cartella dei frame
    # output_video = "output_video_2.mp4"  # Nome del file video
    # fps = 30  # Frame per secondo

    # Ottieni i file immagine ordinati
    frame_files = sorted(
        [f for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])  # Ordina in base al numero prima del .jpg
    )
    # Controlla se ci sono frame
    if not frame_files:
        raise ValueError("Nessun frame trovato nella cartella!")

    # Leggi il primo frame per ottenere la risoluzione
    first_frame_path = os.path.join(frame_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        raise ValueError(f"Impossibile leggere il frame: {first_frame_path}")

    height, width, _ = first_frame.shape

    # Configura il VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Scrivi ogni frame nel video
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Impossibile leggere il frame: {frame_path}. Ignorato.")
            continue

        video_writer.write(frame)

    # Rilascia il video writer
    video_writer.release()
    print(f"Video salvato correttamente come {output_video}")


def rinomina_files(cartella):
    """
    Rinomina i file in una cartella rimuovendo il prefisso 'sfondi_'

    Args:
        cartella (str): Il percorso della cartella contenente i file da rinominare
    """
    # Verifica che la cartella esista
    if not os.path.exists(cartella):
        print(f"La cartella {cartella} non esiste")
        return

    # Itera sui file nella cartella
    for filename in os.listdir(cartella):
        # Verifica se il file inizia con 'sfondi_'
        if filename.startswith('frame_'):
            # Crea il nuovo nome file rimuovendo 'sfondi_'
            nuovo_nome = filename.replace('frame_', '')

            # Crea i percorsi completi
            vecchio_percorso = os.path.join(cartella, filename)
            nuovo_percorso = os.path.join(cartella, nuovo_nome)

            try:
                # Rinomina il file
                os.rename(vecchio_percorso, nuovo_percorso)
                # print(f"Rinominato: {filename} -> {nuovo_nome}")
            except Exception as e:
                print(f"Errore nel rinominare {filename}: {str(e)}")


def extract_frames(video_path, output_folder):
    """
    Estrae i frame da un video e li salva in una cartella.

    Args:
        video_path (str): Il percorso del file video (es. 'video.mp4').
        output_folder (str): La cartella in cui salvare i frame.
    """
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Apri il video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        # Se non ci sono più frame, interrompi il ciclo
        if not ret:
            break

        # Definisci il nome del file per il frame corrente
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        # Salva il frame come immagine JPEG
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Rilascia la risorsa del video
    cap.release()
    print(f"Salvati {frame_count} frame nella cartella '{output_folder}'.")


## MASK OPERATORS

def recognize(image, mask_generator, predictor, points, labels, savefig=None, filename=None):
    print("Predicting object masks...")
    obj_masks = mask_generator.generate(image)

    print("Predicting railway background...")
    predictor.set_image(image)
    b_masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    b_masks = b_masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    anomalies = []
    for obj in obj_masks:
        if 323 < obj['point_coords'][0][1] < 470 and obj['area'] < 1300:
            # print(mask['point_coords'], mask['predicted_iou'], mask['stability_score'])
            overlap = calculate_overlap(obj['segmentation'], b_masks[0])
            if overlap < 15:
                print(f"Overlap: {overlap:.3f}")
                anomalies.append(obj)
    i = 5
    while len(anomalies) == 0 and i >= 0:
        for obj in obj_masks:
            if 323 < obj['point_coords'][0][1] < 470 and obj['area'] < 1300 and check_mask_containment(b_masks[0], obj[
                'segmentation']):
                # print(mask['point_coords'], mask['predicted_iou'], mask['stability_score'])
                # overlap = calculate_overlap(obj['segmentation'], b_masks[0])
                # if overlap < i:
                print(f"Overlap: {overlap:.3f}")
                anomalies.append(obj)

        i -= 5
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(anomalies)

    plt.axis('off')
    if savefig and filename:
        plt.savefig(f"./recognize_2/{filename}.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    # show_masks(image, b_masks[0], scores, folder_path, point_coords=points, input_labels=labels)


def calculate_overlap(mask1, mask2):
    """
    Calculate overlap percentage between two binary masks.
    Returns overlap as percentage of smaller mask area.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    min_area = min(mask1.sum(), mask2.sum())
    overlap_percentage = (intersection / min_area) * 100 if min_area > 0 else 0
    return overlap_percentage


def check_mask_containment(base_mask, query_mask) -> bool:
    """
    Check if query_mask is contained within the first channel of base_masks.

    Args:
        base_mask: Tensor of shape (B, H, W) containing the base / background mask
        query_mask: Tensor of shape (B, H, W) or (1, H, W) containing the mask to check

    Returns:
        Tensor of shape (B,) containing boolean values indicating if query_mask
        is contained within the base mask for each batch
    """

    is_contained = np.all((query_mask * (1 - base_mask)) == 0)

    base_active_h = np.where(np.any(base_mask, axis=1))[0]
    base_active_w = np.where(np.any(base_mask, axis=0))[0]

    # Find first and last active pixels for query mask
    query_active_h = np.where(np.any(query_mask, axis=1))[0]
    query_active_w = np.where(np.any(query_mask, axis=0))[0]

    bounds_check = False
    if len(base_active_h) > 0 and len(query_active_h) > 0:
        # Check if query mask's active region is within base mask's bounds
        h_check = (base_active_h[0] <= query_active_h[0] and
                   base_active_h[-1] >= query_active_h[-1])
        w_check = (base_active_w[0] <= query_active_w[0] and
                   base_active_w[-1] >= query_active_w[-1])
        bounds_check = h_check and w_check

    meets_criteria = is_contained or bounds_check

    return meets_criteria


def refine_mask(image, image_predictor, points, labels):
    print("Creating masks...")
    image_predictor.set_image(image)
    masks, scores, logits = image_predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    # d = fill_holes_in_mask(masks[0])
    d = advanced_hole_filling(masks[0], 10)
    plt.imshow(d)
    plt.show()


def fill_holes_in_mask(mask):
    """
    Fill holes in a binary segmentation mask using different methods.

    Parameters:
    mask: numpy.ndarray
        Binary input mask where 255 or 1 represents the foreground
        and 0 represents the background/holes

    Returns:
    dict: Dictionary containing results from different filling methods
    """
    # Ensure binary mask
    if mask.max() > 1:
        mask = mask / 255.0
    binary_mask = mask.astype(np.uint8)

    results = {}

    # Method 1: Morphological Closing
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closing_result = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    results['morphological_closing'] = closing_result

    # Method 2: Floodfill
    # Create a copy since floodfill modifies the input
    floodfill_mask = binary_mask.copy()
    height, width = floodfill_mask.shape
    mask_for_flood = np.zeros((height + 2, width + 2), np.uint8)
    # Flood from point (0,0)
    cv2.floodFill(floodfill_mask, mask_for_flood, (0, 0), 1)
    # Invert the image
    floodfill_result = cv2.bitwise_not(floodfill_mask)
    # Combine with original mask
    filled_mask = binary_mask | floodfill_result
    results['floodfill'] = filled_mask

    # Method 3: Contour Filling
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(binary_mask)
    cv2.drawContours(contour_mask, contours, -1, 1, -1)  # -1 means fill the contour
    results['contour_filling'] = contour_mask

    return results


def advanced_hole_filling(mask, min_hole_size=50):
    """
    Advanced hole filling with size-based filtering

    Parameters:
    mask: numpy.ndarray
        Binary input mask
    min_hole_size: int
        Minimum size of holes to fill (in pixels)

    Returns:
    numpy.ndarray: Mask with holes filled based on size criteria
    """
    # Ensure binary mask
    if mask.max() > 1:
        mask = mask / 255.0
    binary_mask = mask.astype(np.uint8)

    # Find holes
    holes = cv2.bitwise_not(binary_mask)

    # Label connected components in holes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)

    # Create output mask
    result = binary_mask.copy()

    # Fill holes based on size
    for label in range(1, num_labels):  # Skip background (label 0)
        if stats[label, cv2.CC_STAT_AREA] < min_hole_size:
            result[labels == label] = 1

    return result


def find_corresponding_segmentation(image_path, segmentation_folder):
    """
    Trova il file di segmentazione corrispondente a un'immagine di input.

    Args:
        image_path (str): Percorso del file immagine originale.
        segmentation_folder (str): Directory contenente le segmentazioni.

    Returns:
        str | None: Percorso del file di segmentazione corrispondente, se esiste. Altrimenti None.
    """
    # Estrai il numero dal nome del file originale
    filename = os.path.basename(image_path)  # Ottieni solo il nome del file
    number_part = filename.split('_')[-1].split('.')[0]  # Ottieni la parte numerica

    # Costruisci il nome del file di segmentazione
    segmentation_filename = f"segmentazione_oggetti_{number_part}.png"
    segmentation_path = os.path.join(segmentation_folder, segmentation_filename)

    # Controlla se il file di segmentazione esiste
    if os.path.exists(segmentation_path):
        return segmentation_path
    else:
        return None  # Se il file non esiste


def segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Calcola IoU, Dice Coefficient, Precision e Recall tra la maschera predetta e la ground truth.

    Args:
        pred_mask (np.ndarray): Maschera predetta (binaria: 0 o 1)
        gt_mask (np.ndarray): Maschera ground truth (binaria: 0 o 1)

    Returns:
        dict: Dizionario con i valori delle metriche {"IoU": float, "Dice": float, "Precision": float, "Recall": float}
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    IoU = intersection / union if union > 0 else 0.0
    Dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0

    TP = intersection  # Veri positivi
    FP = np.logical_and(pred_mask, ~gt_mask).sum()  # Falsi positivi
    FN = np.logical_and(~pred_mask, gt_mask).sum()  # Falsi negativi

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return {"IoU": IoU, "Dice": Dice, "Precision": Precision, "Recall": Recall}


def create_grid(box, points_per_row=None):
    if points_per_row is None:
        points_per_row = [2, 2]

    rows = len(points_per_row)
    step_y = int(((box[3] - box[1]) / rows))
    points = []

    for row in range(rows):
        y = step_y * row + int(step_y / 2) + box[1]
        step_x = int(abs(box[2] - box[0]) / points_per_row[row])
        for i in range(points_per_row[row]):
            x = step_x * i + box[0] + int(step_x / 2)
            points.append([x, y])
    points = np.array(points)

    return points, np.ones(len(points))  # points, labels


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def find_holes(mask, min_hole_size=50):
    """
    Trova i buchi all'interno di un'area nella maschera binaria e restituisce i centroidi dei buchi abbastanza grandi.

    Args:
        mask (np.ndarray): Maschera binaria (1, H, W) di tipo uint8 (valori 0 e 1 o 0 e 255).
        min_hole_size (int): Dimensione minima per considerare un buco valido.

    Returns:
        List[Tuple[int, int]]: Lista di centroidi dei buchi (x, y).
    """
    # Rimuove la prima dimensione se la maschera è (1, H, W) -> diventa (H, W)
    # mask = mask.squeeze()

    # Trova i contorni degli oggetti principali
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crea una maschera nera di sfondo per disegnare gli oggetti trovati
    filled_mask = np.zeros_like(mask)

    # Riempie completamente gli oggetti principali per ottenere solo i buchi
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Trova i buchi confrontando la maschera riempita con l'originale
    holes_mask = np.logical_and(filled_mask > 0, mask == 0).astype(np.uint8) * 255

    # Trova i contorni dei buchi
    hole_contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in hole_contours:
        area = cv2.contourArea(cnt)
        if area > min_hole_size:  # Filtra i buchi troppo piccoli
            M = cv2.moments(cnt)
            if M["m00"] > 0:  # Evita divisioni per zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append([cx, cy])

    return np.array(centroids), np.random.choice(np.arange(2, 51), size=len(centroids), replace=False)


def is_contained(box: np.ndarray, background_box: np.ndarray, threshold: float = 0.9) -> bool:
    """
    Verifica se il box è contenuto per almeno il 90% all'interno del background box.

    Args:
        box (np.ndarray): Array (x_min, y_min, x_max, y_max) del box da verificare.
        background_box (np.ndarray): Array (x_min, y_min, x_max, y_max) del box di sfondo.
        threshold (float): Percentuale minima di contenimento (default 0.9).

    Returns:
        bool: True se il box è contenuto almeno per il 90%, False altrimenti.
    """
    # Coordinate dei due box
    x_min_b, y_min_b, x_max_b, y_max_b = box
    x_min_bg, y_min_bg, x_max_bg, y_max_bg = background_box

    # Calcolo dell'intersezione tra i due box
    x_min_int = max(x_min_b, x_min_bg)
    y_min_int = max(y_min_b, y_min_bg)
    x_max_int = min(x_max_b, x_max_bg)
    y_max_int = min(y_max_b, y_max_bg)

    # Calcolo dell'area del box e dell'intersezione
    box_area = (x_max_b - x_min_b) * (y_max_b - y_min_b)
    inter_area = max(0, x_max_int - x_min_int) * max(0, y_max_int - y_min_int)
    if box_area > 30000:
        return False
    # Controllo la percentuale di contenimento
    return (inter_area / box_area) >= threshold if box_area > 0 else False


def grounding_Dino_analyzer(image, model, caption, device, show=False, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25):
    print("Analysis with Grounding Dino")
    image_source, gd_image = load_image(image)

    gd_boxes, logits, phrases = predict(
        model=model,
        image=gd_image,
        caption=caption,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device,
    )

    if show:
        annotated_frame = annotate(image_source=image_source, boxes=gd_boxes, logits=logits, phrases=phrases)

        cv2.imshow("Visualizing results", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(phrases, logits)

    h, w, _ = image_source.shape
    gd_boxes = gd_boxes * torch.Tensor([w, h, w, h])
    gd_boxes = box_convert(boxes=gd_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return gd_boxes, phrases, logits.cpu().numpy()


def grounding_Dino_analyzer_plt(image_path, model, text_prompt, device, show=False, BOX_TRESHOLD=0.35,
                                TEXT_TRESHOLD=0.25):
    """Analyze image with Grounding DINO to detect objects matching text prompt"""
    # Load image
    image_source, image = load_image(image_path)

    # Run Grounding DINO prediction
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device,
    )

    # Convert boxes to proper format
    h, w, _ = image_source.shape
    boxes_denormalized = boxes * torch.Tensor([w, h, w, h])
    xyxy_boxes = box_convert(boxes=boxes_denormalized, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # Show visualization if requested
    if show:
        # Create visualization without using the labels parameter directly
        annotated_frame = annotate(
            image_source=image_source.copy(),
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )

        # Display with matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_frame)
        plt.axis('off')
        plt.show()

    return xyxy_boxes, phrases, logits.cpu().numpy()


def is_mask_in_box(mask, box, margin=10):
    """
    Verifica se una maschera binaria è contenuta in un box, considerando un margine di tolleranza.

    Args:
        mask (np.ndarray): Maschera binaria di forma (1, H, W)
        box (list/tuple): Coordinate del box in formato [x1, y1, x2, y2]
        margin (int): Margine di tolleranza in pixel da aggiungere al box (default: 10)

    Returns:
        bool: True se la maschera è contenuta nel box allargato, False altrimenti
    """
    # Verifica input
    assert mask.shape[0] == 1, "La maschera deve avere shape (1, H, W)"
    assert len(box) == 4, "Il box deve avere 4 coordinate [x1, y1, x2, y2]"

    # Estrai coordinate del box e applica il margine
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - margin)  # Assicurati di non andare sotto 0
    y1 = max(0, y1 - margin)
    x2 = min(mask.shape[2], x2 + margin)  # Assicurati di non superare i limiti dell'immagine
    y2 = min(mask.shape[1], y2 + margin)

    # Trova le coordinate dei pixel non-zero nella maschera
    y_coords, x_coords = np.where(mask[0] > 0)

    if len(x_coords) == 0:  # Se la maschera è vuota
        return True

    # Verifica se tutti i punti della maschera sono dentro il box allargato
    mask_in_box = (
            (x_coords >= x1).all() and
            (x_coords <= x2).all() and
            (y_coords >= y1).all() and
            (y_coords <= y2).all()
    )

    return mask_in_box

def extract_ground_points_and_labels(image_source, ground_gd_box):
    points = []
    width = image_source.shape[1]
    height = image_source.shape[0]
    gd_width = int(ground_gd_box[2] - ground_gd_box[0])
    gd_height = int(ground_gd_box[3] - ground_gd_box[1])
    x = int(ground_gd_box[0])
    y = int(ground_gd_box[1])

    for j in range(y+gd_height-10, y+60, -120):
        for i in range(x+10,x+gd_width-10,120):
            points.append([i,j])

    labels = np.ones(len(points))

    IMAGE_COPY = image_source.copy()
    for point in points:
        cv2.circle(IMAGE_COPY, (point[0], point[1]), 5, (0, 0, 255), -1)
    cv2.imshow("Image", IMAGE_COPY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points, labels



def extract_main_internal_railway_points_and_labels(image_source, gd_box, rails_masks):
    #Provo a fare la media tra: metà dell'iimagine, metà della gd box, metà tra i binari di canny

    width = image_source.shape[1]
    height = image_source.shape[0]
    gd_width = gd_box[2]-gd_box[0]

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
        if obj_id == 1:
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
        for i in range(len(savol_array)):
            if i>0 and i<len(savol_array_left)-1 and (i%6) == 0:
                #savol_array_expanded.append(savol_array[i])
                current_y_in_rail_box = savol_array[i][1] - y
                savol_array_expanded.append([savol_array[i][0]-int(current_y_in_rail_box*0.12), savol_array[i][1]])
                savol_array_expanded.append([savol_array[i][0]+int(current_y_in_rail_box*0.12), savol_array[i][1]])
                savol_array_expanded_negative.append([savol_array[i][0] - int(current_y_in_rail_box * 1), savol_array[i][1]])
                savol_array_expanded_negative.append([savol_array[i][0] + int(current_y_in_rail_box * 1), savol_array[i][1]])
        # TODO guardando le maschere degli oggetti detectati e metto i punti negativi su di loro, e devo controllare che quelli verdi che hp messo non siano in essi

        #Removing positive points that are inside a detected object and adding negative points for the same objects
        for obj_id, mask in rails_masks.items():
            if obj_id != 1 and obj_id!= 0:
                h, w = mask.shape[-2:]
                binary_mask = mask.reshape(h, w).astype(np.uint8)

                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                temp_mask_image = binary_mask[..., None] * color.reshape(1, 1, -1)

                for p in savol_array_expanded[:]:  # Usa copia per evitare problemi di modifica durante iterazione
                    if temp_mask_image[int(p[1]), int(p[0])].sum() > 0:
                        savol_array_expanded.remove(p)



        #FIXME controllare che ci siano punti verdi, è possibile che non ci siano, quindi usare la maschera calcolata precedentemente
        points = np.array(np.concatenate((savol_array_expanded, savol_array_expanded_negative)))    #FIXME se detecto un oggetto che è il binario stesso poi faccio il pop di tutti i punti verdi nel binario ed esplode
        labels = np.ones(len(savol_array_expanded))
        neg_labels = np.zeros(len(savol_array_expanded_negative))
        labels = np.concatenate((labels, neg_labels))

    return points, labels

def smooth_curve_from_points(rail_points_x, rail_points_y):
    rail_points_x = np.array(rail_points_x)
    rail_points_y = np.array(rail_points_y)
    coeffs = np.polyfit(rail_points_y, rail_points_x, deg=5)  # Polinomio di grado 3    #FIXME da decidere il grado più appropriato
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

    #Simplifying image with blur and morphology, removing noise
    # FloodFill requires the mask to be (H+2, W+2)
    padded_mask = np.zeros((h + 2, w + 2), dtype='uint8')
    padded_mask[1:h + 1,
    1:w + 1] = mask
    cv2.floodFill(black_image, padded_mask, (0, 0), 255)
    black_and_white_mask_image = cv2.bitwise_not(black_image)
    # blur
    blur = cv2.GaussianBlur(black_and_white_mask_image, (0, 0), sigmaX=4, sigmaY=2)
    # otsu threshold
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)[1]

    # Removing mask "islands"or other noise not connected to the main detected object
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    main_contour = max(contours, key=cv2.contourArea)
    main_contour = main_contour[:, 0, :]  # Remove nesting

    black_image = np.zeros_like(black_and_white_mask_image)
    cv2.drawContours(black_image, [main_contour], -1, (255, 255, 255), 3)
    cv2.fillPoly(black_image, pts=[main_contour], color=(255, 255, 255))
    a = black_image.astype(bool)
    return a

def show_anomalies(mask, ax,rail_mask):
     #FIXME infatti se per esempio una persona è in piedi accanto ai binari ma è alta e il binario cirva in lontananza dietro la persona diventa arancione ma non è giusto
    mask = np.array(mask, dtype=np.uint8)
    mask = mask.squeeze()
    rail_mask = np.array(rail_mask, dtype=np.uint8)
    rail_mask = rail_mask.squeeze()
    #Expanding mask to detect near objects to the rails
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.dilate(mask, kernel, iterations=5)
    intersection = cv2.bitwise_and(binary_mask, rail_mask)
    if intersection.sum()>0:
        color = np.array([255 / 255, 136 / 255, 0 / 255, 0.5])
    else:
        color = np.array([234 / 255, 255 / 255, 0 / 255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def is_point_inside_box(point, box):
    y_min = int(box[1])
    y_max = int(box[3])
    result = True
    if y_min<point[1]<y_max:       #FIXME la box del terreno non è ancora affidabile, potrei controllare solo la quota e non orizzontalmente
        result = True
    else:
        result = False
    return result

def is_mask_an_obstacle(mask, rail_mask, railway_box):      #FIXME se un oggetto è in prospettiva grosso allora viene tolto, sbagliato
    result = True
    mask = np.array(mask, dtype=np.uint8)
    mask = mask.squeeze()
    rail_mask = np.array(rail_mask, dtype=np.uint8)
    rail_mask = rail_mask.squeeze()
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3)
    binary_mask =cv2.threshold(blurred_mask, 0, 255, cv2.THRESH_BINARY)[1]
    intersection = cv2.bitwise_and(binary_mask, rail_mask)
    substraction = cv2.bitwise_xor(binary_mask, rail_mask)
    intersected_rail_px_count = intersection.sum()
    substraction_rail_px_count = substraction.sum()
    railway_px_count = rail_mask.sum()
    #Removes the masks that are inside the rail mask and too big, but not protruding, probably is the rail itself
    if intersected_rail_px_count/railway_px_count>0.7 and substraction_rail_px_count<(0.05*railway_px_count):       #FIXME da verificare (non so come) che abbia la forma del binario
        result = False

    x_min = int(railway_box[0])
    x_max = int(railway_box[2])
    y_min = int(railway_box[1])
    y_max = int(railway_box[3])

    #Removes the masks that are not inside the detected box of the entire railway
    black_image = np.zeros_like(mask)
    railway_box_points = np.array([
        [[x_min, y_min]],
        [[x_max, y_min]],
        [[x_max, y_max]],
        [[x_min, y_max]]
    ], dtype=np.int32)
    cv2.fillPoly(black_image, pts=[railway_box_points], color=(255, 255, 255))
    intersection = cv2.bitwise_and(black_image, binary_mask)
    mask_px_count = binary_mask.sum()
    intersected_mask_px_count = intersection.sum()
    if intersected_mask_px_count/mask_px_count<0.5:
        result = False

    #Check if mask is too small
    mask_px_count = binary_mask.sum()
    if mask_px_count<100:
        result = False

    return result

def is_mask_duplicate(mask, obj_id, last_masks_rails):
    result = False
    mask = np.array(mask, dtype=np.uint8)
    mask = mask.squeeze()
    for last_obj_id, last_mask in last_masks_rails.items():
        if last_obj_id!=obj_id and last_obj_id != 1:
            last_mask = np.array(last_mask, dtype=np.uint8)
            last_mask = last_mask.squeeze()
            substraction = cv2.bitwise_xor(mask, last_mask)
            intersection = cv2.bitwise_and(mask, last_mask)
            substraction_px_count = substraction.sum()
            intersection_px_count = intersection.sum()
            last_mask_px_count = last_mask.sum()
            mask_px_count = mask.sum()
            if intersection_px_count/mask_px_count>1 or (intersection_px_count/mask_px_count>0.8 and substraction_px_count < (0.1 * last_mask_px_count)):
                result = True
                break
    return result