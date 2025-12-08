import os
import shutil
# Directories
dir1 = "demo_data/images"                       # original image directory
dir2 = "demo_data/for_training/labels"          # directroy for the annotations, name must be labels for the yolo to recogniez it
target_dir = "demo_data/for_training/images"    # directory for the images, that will be used for train/val (need for this: as we can set the classes that we'd like to segment, we might end up with picture that has no annotations, we won't copy those here)

os.makedirs(target_dir, exist_ok=True)

# get file names without extensions
def get_basenames(path):
    basenames = set()
    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        if os.path.isfile(full_path):
            base, _ = os.path.splitext(fname)
            basenames.add(base.lower())
    return basenames

# Get basenames from both directories
names1 = get_basenames(dir1)
names2 = get_basenames(dir2)

# Basenames that exist in both dirs
paired_basenames = names1 & names2


# Copy files from dir1 whose basename has a pair in dir2
for fname in os.listdir(dir1):
    full_path = os.path.join(dir1, fname)
    if os.path.isfile(full_path):
        base, ext = os.path.splitext(fname)
        if base.lower() in paired_basenames:
            dest_path = os.path.join(target_dir, fname)
            shutil.copy2(full_path, dest_path)
    

# Preparing the data for training: splitting it to train (0.7) and val (0.3) sets
images_train_dir = os.path.join(target_dir, "train")
images_val_dir   = os.path.join(target_dir, "val")
ann_train_dir    = os.path.join(dir2, "train")
ann_val_dir      = os.path.join(dir2, "val")

os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(ann_train_dir, exist_ok=True)
os.makedirs(ann_val_dir, exist_ok=True)

# All annotation files (ignore subfolders)
ann_files = [
    f for f in os.listdir(dir2)
    if os.path.isfile(os.path.join(dir2, f))
]

# Map image basename -> image filename from target_dir
img_files = [
    f for f in os.listdir(target_dir)
    if os.path.isfile(os.path.join(target_dir, f))
]
img_by_base = {os.path.splitext(f)[0].lower(): f for f in img_files}

# split at 0.7
split_idx = int(0.7 * len(ann_files))

for i, ann_fname in enumerate(ann_files):
    base, _ = os.path.splitext(ann_fname)
    img_fname = img_by_base.get(base.lower())

    src_img = os.path.join(target_dir, img_fname)
    src_ann = os.path.join(dir2, ann_fname)

    if i < split_idx:
        # train
        dst_img = os.path.join(images_train_dir, img_fname)
        dst_ann = os.path.join(ann_train_dir, ann_fname)
    else:
        # val
        dst_img = os.path.join(images_val_dir, img_fname)
        dst_ann = os.path.join(ann_val_dir, ann_fname)
    
    # Move them
    shutil.move(src_img, dst_img)
    shutil.move(src_ann, dst_ann)