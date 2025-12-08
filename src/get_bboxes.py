import os

def get_bboxes(txt_path, coord_cols, class_column, classes, sep=None):
    """
    Parse an annotation .txt file and return bounding boxes
    for the given classes.

    Parameters
    ----------
    txt_path : str
        Path to the annotation .txt file.
    coord_cols : list[int]
        Indices of the columns that contain coordinates.
    class_column : int
        Index of the column that contains the class id/label.
    classes : list
        List of allowed classes (e.g. [0, 1]).
    sep : str
        Separator for splitting lines.

    Returns
    -------
    bboxes : list[list[float]]
        List of coordinate lists, one per matching object.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Annotation file not found: {txt_path}")

    target_classes = {str(c) for c in classes}
    bboxes = []
    class_ids = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(sep)

            cls_val = parts[class_column]

            # only grab the included classes
            if str(cls_val) not in target_classes:
                continue
            
            # saving and remaping the original class ID's
            orig_cls = int(cls_val)
            remapped_cls = classes.index(orig_cls)

            coords = [float(parts[i]) for i in coord_cols]
            # we need to transform (x,y,w,h) to (x1,y1,x2,y2) for obb
            coords[2] = coords[0] + coords[2]
            coords[3] = coords[1] + coords[3]
            bboxes.append(coords)
            class_ids.append(remapped_cls)

    return bboxes, class_ids

