# Wall & Road Crack Detection
In This file, I will train a Yolov11 model to detect wall cracks and road cracks, The project will use the Yolov11 segmentation large model version that is previously trained on the COCO dataset. However, I will retrain it on the custom dataset to detect wall and road cracks.

## Dataset Used:

- Dataset Name: Crack Computer Vision Dataset
- URL: https://universe.roboflow.com/apu-jimbr/crack-uuasm

- Dataset Description: The dataset has about 4000 images of cracks in roads, walls, and cement in general.
1. The Number of files in data\test\images is 322.
2. The Number of files in data\test\labels is 322.
3. The Number of files in data\train\images is 7072.
4. The Number of files in data\train\labels is 7072.
5. The Number of files in data\valid\images is 455.
6. The Number of files in data\valid\labels is 455.

## Model Details & Settings:

- Model Name: Yolov11
- Model Version: Segmentation Nano Model
- Epochs: 25
- Input Image Size: 640
- Batch: 32

## Inference Details:
Images gathered from google about wall cracks and road cracks. 10 in total.