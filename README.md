# ADE20K to YOLO Converter

A tool for converting ADE20K dataset to YOLO format for instance segmentation tasks.

## Overview

This project provides tools to:
1. Convert ADE20K dataset annotations to COCO format
2. Further convert COCO format annotations to YOLO format
3. Filter and clean up annotations based on specified categories

## Prerequisites

- Python 3.8+
- ADE20K dataset (2021_17_01 version)
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  pip install ultralytics
  ```

## Project Structure
```
ade20k_to_yolo/
├── convert_to_yolo.py     # Main conversion script
├── ADE20K_To_COCO/        # ADE20K to COCO conversion module
│   └── AdeToCOCO.py       # Core conversion logic
├── dataset/               # Output directory
│   ├── images/           # Converted images
│   ├── labels/           # YOLO format labels
│   └── data.yaml         # YOLO dataset configuration
```

This tool can convert the ADE20K dataset to YOLO format for instance segmentation tasks. It first converts ADE20K to COCO format, then converts COCO format to YOLO format. The tool is designed to filter and process specific object categories based on their frequency in the dataset.

Please :star: the repo if you find it useful :smiley:

## Usage

1. First, prepare the ADE20K dataset:
   ```bash
   # Download ADE20K dataset and extract to project directory
   # Make sure you have index_ade20k.pkl file
   ```

2. Convert ADE20K to COCO format:
   ```bash
   python ADE20K_To_COCO/AdeToCOCO.py --datasetDir path/to/ADE20K --pklPath path/to/index_ade20k.pkl --objectNames wall person tree car plant
   ```
   objectNames: List of target object categories to extract\
   pklPath: Path to index_ade20k.pkl provided by ADE20K\
   datasetDir: Path to ADE20K dataset

3. Convert COCO to YOLO format:
   ```bash
   python convert_to_yolo.py
   ```

4. Train YOLO model:
   ```bash
   python dataset/yolo11_train.py
   ```

## Features

- Supports instance segmentation annotations
- Filters categories based on frequency
- Handles coordinate normalization
- Removes invalid annotations
- Generates statistics for the converted dataset

## Output Format

The converter generates YOLO format labels with the following structure:
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
Where:
- class_id: Integer class identifier (0-based)
- x1,y1,...xn,yn: Normalized polygon coordinates for instance segmentation

## Test Result

The converted dataset can be directly used for YOLO training:
- Supports both instance segmentation and object detection
- Maintains original image quality
- Preserves polygon annotations

## Citation
```
@misc{
    title  = {ade20k_to_yolo},
    author = {Charlie},
    url    = {https://github.com/charlie-C929/ade20k_to_yolo},
    year   = {2024}
}
```