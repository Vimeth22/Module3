# Module 3 – Complete Image Processing Web Application
### Gradients · Keypoints · Contours · ArUco Marker Segmentation

This project is a full Flask-based web application implementing all four parts of Assignment 3 in the Computer Vision course. It processes images using OpenCV and displays the results through an interactive browser interface.

The system supports:  
- Gradient filtering  
- Edge and corner keypoint detection  
- Largest contour extraction  
- ArUco marker segmentation  

All input images are selected directly through the web UI, and all outputs are saved automatically to the `static/output` directory.

## 1. Project Structure

```
Assignment3/
│── app.py
|-- dataset
|-- dataset_aruco
└── static/
    └── output/
├── templates/
   ├── index.html
   ├── result.html
   ├── aruco_index.html
   └── aruco_result.html
|── requirements.txt

## 2. Features Included

### Part 1 – Image Gradients
- Part 1-3 I have uploaded 5 iamges. You can choose any image and see the results.
- Also you can choose your own images too.
- Gradient magnitude  
- Gradient angle  
- Laplacian of Gaussian (LoG)

### Part 2 – Keypoint Detection
- Canny edges  
- Harris corners  

### Part 3 – Object Contour Extraction
- Extracts the largest contour  
- Draws the boundary  

### Part 4 – ArUco Marker Segmentation
- There are 10 images. You can click any of thise and see the output.
- Detects ArUco marker  
- Computes convex hull  
- Displays original + segmented output  

## 3. Installation

```
pip install -r requirements.txt
```

## 4. Running the Application

```
python app.py

```

## 5. Usage

### Parts 1–3
Place images in:
```
dataset
```

### Part 4
Place ArUco images in:
```
dataset_aruco
```

## 6. Output Files

All processed results are saved in:
```
static/output/
```

Filenames include:
- _original  
- _magnitude  
- _angle  
- _log  
- _edges  
- _corners  
- _boundary  
- _segmented  

## 7. Summary

This project integrates four computer vision tasks into one Flask application, supporting gradients, keypoints, contours, and ArUco segmentation.
