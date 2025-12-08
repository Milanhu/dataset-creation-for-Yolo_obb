import cv2
import os
from ultralytics import SAM
import numpy as np
from get_bboxes import get_bboxes
from tqdm import tqdm

# Config
###################################
IMG_DIR = "demo_data/images"                     # path to the base dataset
ANNOT_DIR = "demo_data/annotations"              # path to the base annotation
OUT_DIR = "demo_data/out_image"                  # where the output images shall be saved
OUT_DIR_ANNO = "demo_data/for_training/labels"   # where the new annotations shall be saved
####################################

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_ANNO, exist_ok=True)
image_files = os.listdir(IMG_DIR)

# Load the SAM model
model = SAM("sam_b.pt") # downloads automatically if not found in the folder

#Kernel for edge smoothing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

# Process the images
for pic in tqdm(image_files, desc="Processing images", unit="img", total=len(image_files)):
    
    # path for each image & annotations
    img_path = os.path.join(IMG_DIR, pic)
    base, _ = os.path.splitext(pic)
    base = base + ".txt"
    txt_path = os.path.join(ANNOT_DIR, base)

    # Reading in the given image for later visualization (optional)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Get the bbox coordinates from txt
    bboxes, class_ids = get_bboxes(txt_path,[0,1,2,3],5,[2,3,4,5,6,7,8,9], sep =',')
    # txt_path: path to the given annotation txt
    # [0,1,2,3]: columns containing the bounding box's coordiantes
    # 5: column containing the classes
    #[2,3,4,5,6,7,8,9]: classes that we are interested in
    
    if bboxes is None or len(bboxes) == 0: continue

    # segmentation on all bboxes using SAM
    results = model(img_path, bboxes=bboxes, save=False, verbose = False)

    all_boxes = []

    # getting the mask (convert to booelans to int 0-255 for contours function) for each bbox
    for i in range(len(results[0].masks.data)):
        mask = (results[0].masks.data[i].cpu().numpy() * 255).astype("uint8")

        # Dilation & Erosion to get rid of weak contours in the mask
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        # find the contours and select the largest
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue # in case erosion is too strong
        cnt = max(contours, key=cv2.contourArea)

        # get the minimum area bbox
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)

        # normalize oriented box coordinates to achieve YOLO format
        coords_norm = []
        for x_pt, y_pt in box:
            coords_norm.append(x_pt / w)
            coords_norm.append(y_pt / h)
        
        # get class id aligned with this index
        cls_id = class_ids[i]
        all_boxes.append([cls_id] + coords_norm)
        
        # Draw the bbox to the original picture
        box_int = np.intp(box)
        cv2.drawContours(img, [box_int], 0, (0, 255, 0), 2)

    # Output filenames
    out_name = os.path.splitext(pic)[0] + ".jpg"
    out_path = os.path.join(OUT_DIR, out_name)

    # Save the image with the bboxes on it
    cv2.imwrite(out_path, img)

    # Save the new annotations
    out_ann_name =  os.path.splitext(pic)[0] + ".txt"
    out_ann_path = os.path.join(OUT_DIR_ANNO, out_ann_name)
    with open(out_ann_path, "w") as f:
        for row in all_boxes:
            line = " ".join(f"{v}" for v in row)
            f.write(line + "\n")
    del results

