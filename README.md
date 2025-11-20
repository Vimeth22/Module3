# Computer Vision – Module 3 (Flask + OpenCV)

This project implements all five tasks from Module 3 and runs **in real time** as a small web app.

## Features

1. **Gradients & LoG**  
   - Computes Sobel gradients (magnitude & angle)  
   - Computes Laplacian of Gaussian (LoG)

2. **Keypoints: Edges & Corners (no ML)**  
   - Edges via gradient-based Canny + non-maximum suppression  
   - Corners via Shi–Tomasi (`cv2.goodFeaturesToTrack`)

3. **Exact Object Boundary**  
   - Otsu thresholding + morphology  
   - Largest connected component + contour extraction  
   - Polygonal simplification (Douglas–Peucker)

4. **Segmentation of a Non-Rectangular Object with ArUco Markers**  
   - Detects ArUco markers on the boundary  
   - Builds a polygon from detected corners (sorted angularly)  
   - Refines foreground with GrabCut using the polygon as the initial mask

5. **Comparison with SAM2 (optional)**  
   - If you have SAM2 installed, the app can call it and compute IoU between our mask and SAM2’s mask.

## How to run

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your dataset**  
   Put **at least 10 images of the same object** (varied angles/distances) in:
   ```
   static/dataset/
   ```

3. **Run**
   ```bash
   python app.py
   ```
   Open http://127.0.0.1:5000

## Notes

- **No deep learning** is used for Tasks 1–4.  
- Task 5 requires SAM2 installed separately. If SAM2 is missing, the UI will show a helpful message and skip it.
- ArUco detection works with OpenCV’s built-in dictionaries (e.g., `DICT_4X4_50`). Make sure markers are visible on the boundary.

