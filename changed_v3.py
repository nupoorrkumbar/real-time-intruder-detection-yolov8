from flask import Flask, render_template, Response, send_file, jsonify
import cv2
import time
import os
import threading
import re

# Import the necessary functions from your modules
import iptechniques
from ultralytics import YOLO  

# Initialize the YOLO model globally
model = YOLO('best15.pt')

app = Flask(__name__)

# Video capture object
video_capture = cv2.VideoCapture(0)

IMAGE_DIR = 'captured_images'
captured_image_paths = []

# Function to capture and save images at regular intervals
def capture_images(frame, image_name):
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    # Save image with timestamp
    timestamp = int(time.time())
    image_path = os.path.join(IMAGE_DIR, image_name)
    cv2.imwrite(image_path, frame)
    captured_image_paths.append(image_path)
    iptechniques.process(image_path)

# Generator function to stream video
def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        results = model(frame, conf=0.6, save=True, save_crop=True, iou=0.2)
        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('changed_v3.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def get_latest_image(folder_path):
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if image_files:
            image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
            latest_image_filename = image_files[0]
            latest_image_path = os.path.join(folder_path, latest_image_filename)
            return latest_image_path, latest_image_filename
    return None, None

def g_r_image():
    while True:
        time.sleep(5)
        INTRUDER_DIR = 'runs/detect/predict/crops/intruder'
        intruder_image_path, intruder_image_filename = get_latest_image(INTRUDER_DIR)
        if intruder_image_path:
            frame = cv2.imread(intruder_image_path)
            if frame is not None:
                capture_images(frame, intruder_image_filename)  
                yield (b'--frame2\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')
            else:
                print("Error: Unable to read image")
        else:
            print("Error: No image found")

@app.route('/get_recent_image')
def get_recent_image():
    return Response(g_r_image(), mimetype='multipart/x-mixed-replace; boundary=frame2')

@app.route('/all_image_paths')
def all_image_paths():
    return jsonify({'image_paths': captured_image_paths})

@app.route('/image/original')
def get_original_image():
    image_path = os.path.join("original_images", "Original_image.jpg")
    print(f"Fetching original image from {image_path}")
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/image/medianfiltered')
def get_median_filtered_image():
    image_path = os.path.join("median_filtered_images", "median_filtered_image.jpg")
    print(f"Fetching lowpass image from {image_path}")
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/image/highpass')
def get_highpass_image():
    image_path = os.path.join("highpass_images", "hpf_filtered_image.jpg")
    print(f"Fetching highpass image from {image_path}")
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/image/histogram')
def get_histogram_image():
    image_path = os.path.join("histogram_images", "histogram_equalized_image.jpg")
    print(f"Fetching histogram image from {image_path}")
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/image/edgedetection')
def get_edgedetection_image():
    image_path = os.path.join("edgedetection_images", "edge_detected_image.jpg")
    print(f"Fetching edgedetection image from {image_path}")
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/image/thresholding1')
def get_thresholding1_image():
    image_path = os.path.join("Thresholding1_images", "Threshold1_image.jpg")
    print(f"Fetching thresholding1 image from {image_path}")
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/image/thresholding2')
def get_thresholding2_image():
    image_path = os.path.join("Thresholding2_images", "Threshold2_image.jpg")
    print(f"Fetching thresholding2 image from {image_path}")
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/image/logo')
def get_logo():
    image_path = os.path.join("logo", "logo.png")
    print(f"Fetching LOGO image from {image_path}")
    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run()
