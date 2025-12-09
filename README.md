# dataset-creation-for-Yolo_obb
This repository provides a simple pipeline to create datasets for [YOLO-OBB](https://github.com/ultralytics/ultralytics) (oriented bounding boxes) from existing datasets that only contain axis-aligned bounding boxes (AABB).

After the new dataset has been created, you can fine-tune a base YOLO-OBB model on it to obtain a better-performing model for your specific use case.

## Get started
You can quickly test the pipeline using the included demo data (intended only to verify functionality, not for training).
### Install dependencies
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```
## Create oriented annotations (Run: ```dataset_creation.py```)

This script creates new OBB annotation files using the following logic:

 1.  For each provided axis-aligned bounding box, perform segmentation using SAM (Segment Anything Model from Ultralytics).

 2.  For each mask, compute the minimum-area bounding box.

 3.  Save the oriented bounding boxes in the YOLO-OBB-compatible format.

Note: This code was written for the VisDrone dataset as provided by Ultralytics.
If your dataset’s annotations use a different format, you will need to adapt the conversion logic in ```dataset_creation.py.```

## Prepare data for training (Run: ```prepare_for_training.py```)
This will get your files, needed for the training, in the right structure. If you wish to modify this, you shall modify the ```data.yaml``` as well.

Note: This creates copies of the images, so that it doesn't modify your base dataset. If you wish to train on huge amounts of images, this is recommended to be modified. 

## Start training YOLO-OBB (Run ```train.py```)
This will start the training; please set the parameters to align with your goals. If GPU is available, definitely use 'device=0' in the argument.

## Result (left: base model; right: custom model)
<img width="1884" height="710" alt="image" src="https://github.com/user-attachments/assets/2a980ccd-9b76-4b53-9b06-609a37c963f2" />
<img width="1888" height="705" alt="image" src="https://github.com/user-attachments/assets/e84b4d38-7ccf-4535-b5e4-efb1cfaea692" />

The base YOLO11-OBB model was pre-trained on the DOTA dataset, which has very different annotations and image characteristics; as a result, it performs poorly on VisDrone. After fine-tuning on the generated VisDrone-OBB dataset, the custom model produces significantly more accurate oriented bounding boxes.
