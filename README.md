# dataset-creation-for-Yolo_obb
This repository's goal is to create datasets for Yolo-obb (oriented bounding box) from existing datasets that contain only axis-aligned bounding boxes. 

After the new dataset has been created, we can train the base Yolo-obb model of choice on it, acquiring a better-performing model for our use case.

# Get started
After downloading the code, we can test it with the included demo data. (Note: just for testing the functionality)

As the zeroth step, please run: pip install -r requirements.txt

# First, run the dataset_creation.py
This will create the new annotation files by the following logic: 

Perform segmentation in each provided bounding box using SAM (Segment Anything Model for Ultralytics) --> Calculate minimum area bounding box for each mask --> Save the annotations in the right format for OBB.

Note: This code was written for VisDrone dataset from Ultralytics. If your dataset's annotations are in a different format, you will need to make modifications. 

# Secondly, run the prepare_for_training.py
This will get your files, needed for the training, in the right structure. If you wish to modify this, you shall modify the data.yaml as well.

Note: This creates copies of the images, so that it doesn't modify your base dataset. If you wish to train on huge amounts of images, this is recommended to be modified. 

# Lastly, run the train.py
This will start the training; please set the parameters to align with your goals.  

# Result
<img width="1884" height="710" alt="image" src="https://github.com/user-attachments/assets/2a980ccd-9b76-4b53-9b06-609a37c963f2" />
<img width="1888" height="705" alt="image" src="https://github.com/user-attachments/assets/e84b4d38-7ccf-4535-b5e4-efb1cfaea692" />

Note: The base Yolo11-obb was trained on the DOTA dataset, with vastly different annotations, that's why it performs poorly.
