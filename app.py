import cv2
import numpy as np
import os
import glob
from flask import Flask, render_template, url_for, abort

# ---------------------------------------------------------
# Flask app setup and folder configuration
# ---------------------------------------------------------

# Initialize the Flask app
app = Flask(__name__)

# Folder that holds the original images for Parts 1–3
DATASET_DIR = 'dataset'

# Folder that holds the ArUco images for Part 4
ARUCO_DATASET_DIR = 'dataset_aruco' # Folder for part 4

# Folder where all processed images will be saved and served by Flask
OUTPUT_DIR = os.path.join('static', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Processing pipeline for Parts 1–3 (single image)
# ---------------------------------------------------------
def process_and_save_image(filename):
    """
    Runs all processing for Parts 1, 2, and 3 on a single image.
    The function:
      - Reads the input image.
      - Resizes and converts to grayscale.
      - Computes gradient magnitude, angle, and Laplacian.
      - Performs Canny edge detection.
      - Detects Harris corners.
      - Finds the largest contour from edges and draws its boundary.
    All intermediate and final results are saved into OUTPUT_DIR.
    """
    
    input_path = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(input_path):
        # If the requested input image does not exist, stop here
        return None
    
    # Base name used for all output image filenames
    output_name_base = os.path.splitext(filename)[0]
    
    # Define all the files we're going to create
    # Each key corresponds to one visualization we generate
    output_paths = {
        "original": os.path.join(OUTPUT_DIR, f"{output_name_base}_original.jpg"),
        "magnitude": os.path.join(OUTPUT_DIR, f"{output_name_base}_magnitude.jpg"),
        "angle": os.path.join(OUTPUT_DIR, f"{output_name_base}_angle.jpg"),
        "log": os.path.join(OUTPUT_DIR, f"{output_name_base}_log.jpg"),
        "edges": os.path.join(OUTPUT_DIR, f"{output_name_base}_edges.jpg"),
        "corners": os.path.join(OUTPUT_DIR, f"{output_name_base}_corners.jpg"),
        "boundary": os.path.join(OUTPUT_DIR, f"{output_name_base}_boundary.jpg"),
    }
    
    # If the last file exists, assume everything for this image is already done
    if os.path.exists(output_paths["boundary"]):
        return output_name_base
        
    # Read the image from disk
    frame = cv2.imread(input_path)
    if frame is None:
        # Reading failed for some reason (corrupt file, etc.)
        return None

    # ---------------------------------------------------------
    # Preprocessing: resize + convert to grayscale + blur
    # ---------------------------------------------------------

    # Resize so all images are the same size (helps with consistent display)
    h, w = 360, 480
    frame_small = cv2.resize(frame, (w, h))

    # Convert to grayscale for gradient and edge operations
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise before edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # Blur to reduce noise

    # Save the resized original image
    cv2.imwrite(output_paths["original"], frame_small)

    # ---------------------------------------------------------
    # Gradient computation: Sobel operators in x and y
    # ---------------------------------------------------------

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    
    # Magnitude and angle of the gradient
    magnitude = cv2.magnitude(sobelx, sobely)
    angle = cv2.phase(sobelx, sobely)

    # Laplacian of Gaussian (LoG) style operation using Laplacian on blurred image
    log = cv2.Laplacian(blur, cv2.CV_64F, ksize=5)

    # ---------------------------------------------------------
    # Convert to 8-bit images so they can be saved and visualized
    # ---------------------------------------------------------

    mag_display = cv2.convertScaleAbs(magnitude)
    angle_display = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    log_display = cv2.convertScaleAbs(log)
    
    cv2.imwrite(output_paths["magnitude"], mag_display)
    cv2.imwrite(output_paths["angle"], angle_display)
    cv2.imwrite(output_paths["log"], log_display)
    
    
    # ---------------------------------------------------------
    # Canny Edge Detection
    # ---------------------------------------------------------

    canny_edges = cv2.Canny(blur, 100, 200)
    cv2.imwrite(output_paths["edges"], canny_edges)

    # ---------------------------------------------------------
    # Harris Corner Detection
    # ---------------------------------------------------------

    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    # Dilate result for better visualization of corners
    dst = cv2.dilate(dst, None)
    
    # Copy original image and mark detected corners in red
    corner_image = frame_small.copy()
    corner_image[dst > 0.01 * dst.max()] = [0, 0, 255] # Draw red dots
    cv2.imwrite(output_paths["corners"], corner_image)
    
    # ---------------------------------------------------------
    # Boundary extraction via contours on the Canny edges
    # ---------------------------------------------------------

    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boundary_image = frame_small.copy()
    
    if contours:
        # Find and draw only the biggest contour (largest object boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(boundary_image, [largest_contour], -1, (0, 255, 0), 3) 

    # Save the final boundary visualization
    cv2.imwrite(output_paths["boundary"], boundary_image)

    # Return the base name so templates can construct proper filenames
    return output_name_base


# ---------------------------------------------------------
# Processing pipeline for Part 4 (ArUco markers)
# ---------------------------------------------------------
def process_and_save_aruco(filename):
    """
    Finds ArUco markers and draws a convex hull boundary.
    UPDATED for OpenCV 4.7+
    Steps:
      - Load and resize the ArUco image.
      - Detect ArUco markers.
      - Draw the markers and IDs.
      - Compute a convex hull around all marker corners.
      - Draw that hull as a green boundary.
    """
    input_path = os.path.join(ARUCO_DATASET_DIR, filename)
    if not os.path.exists(input_path):
        # If the ArUco image does not exist in the dataset, stop here
        return None
    
    # Add "_aruco" to filename to avoid clashes with regular outputs
    output_name_base = os.path.splitext(filename)[0] + "_aruco"
    
    # We keep an original and a segmented (with hull) version
    output_paths = {
        "original": os.path.join(OUTPUT_DIR, f"{output_name_base}_original.jpg"),
        "segmented": os.path.join(OUTPUT_DIR, f"{output_name_base}_segmented.jpg"),
    }
    
    # If we've already created the segmented version, no need to recompute
    if os.path.exists(output_paths["segmented"]):
        return output_name_base
        
    frame = cv2.imread(input_path)
    if frame is None:
        return None
    
    # Resize for consistent display size
    frame = cv2.resize(frame, (640, 480))
    cv2.imwrite(output_paths["original"], frame)
    
    # ---------------------------------------------------------
    # ArUco detection setup (OpenCV 4.7+ style)
    # ---------------------------------------------------------

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # corners: list of marker corner points
    # ids:     corresponding marker IDs
    # rejected: candidate markers that did not pass detection
    corners, ids, rejected = detector.detectMarkers(frame)
    
    segmented_image = frame.copy()
    
    if ids is not None and len(ids) > 0:
        # Draw all detected markers
        cv2.aruco.drawDetectedMarkers(segmented_image, corners, ids)

        # Gather all marker corners into one big array
        all_marker_corners = np.concatenate(corners)

        # Compute convex hull around all markers to get a global boundary
        hull = cv2.convexHull(all_marker_corners)

        # Draw the convex hull boundary in green
        cv2.drawContours(segmented_image, [hull], -1, (0, 255, 0), 3)

    # Save the segmented ArUco visualization
    cv2.imwrite(output_paths["segmented"], segmented_image)
    
    return output_name_base


# ---------------------------------------------------------
# Flask routes for web interface
# ---------------------------------------------------------

@app.route('/')
def index():
    """Home page for Parts 1-3: lists all images in the dataset folder."""
    # Collect all .jpg/.png images from DATASET_DIR
    image_paths = glob.glob(os.path.join(DATASET_DIR, '*.[jp][pn]g'))
    image_filenames = [os.path.basename(p) for p in image_paths]
    return render_template('index.html', filenames=image_filenames)


@app.route('/view/<filename>')
def view_image(filename):
    """Results page for Parts 1-3: shows processed outputs for one image."""
    # Run processing for this specific image on-demand
    output_name = process_and_save_image(filename)
    if output_name is None:
        # If the image doesn't exist or failed to load, return 404
        return abort(404, "Image not found in dataset.")
    return render_template('result.html', output_name=output_name)


@app.route('/aruco')
def index_aruco():
    """Home page for Part 4: lists all ArUco images."""
    # Collect all .jpg/.png images from ARUCO_DATASET_DIR
    image_paths = glob.glob(os.path.join(ARUCO_DATASET_DIR, '*.[jp][pn]g'))
    image_filenames = [os.path.basename(p) for p in image_paths]
    return render_template('aruco_index.html', filenames=image_filenames)


@app.route('/view_aruco/<filename>')
def view_aruco(filename):
    """Results page for Part 4: shows ArUco detection and convex hull."""
    # Run ArUco processing on-demand
    output_name = process_and_save_aruco(filename)
    if output_name is None:
        # If the ArUco file is missing, show 404
        return abort(404, "ArUco image not found in dataset.")
    return render_template('aruco_result.html', output_name=output_name)


# ---------------------------------------------------------
# Run the Flask development server
# ---------------------------------------------------------
if __name__ == '__main__':
    # run the app
    app.run(debug=True)
