# Intruder Detection System using YOLOv8 and OpenCV  

## Overview  
This project demonstrates a **real-time security surveillance system** that detects and classifies intruders using **YOLOv8 (You Only Look Once)** and **OpenCV**. With security becoming increasingly critical in homes, offices, and public spaces, the system leverages **deep learning, computer vision, and digital image processing** to provide automated intruder detection and visualization.  

The solution includes a **Flask-based GUI** that streams live video, captures intruder snapshots, and displays both original and filtered images for enhanced visibility under different lighting conditions.  

---

## Objectives  
- Detect and classify individuals as **authorized personnel** or **intruders** in real time.  
- Build a **user-friendly interface** for live monitoring.  
- Apply **digital image processing** techniques to improve recognition under varying conditions.  
- Evaluate YOLOv8 performance using standard detection metrics.  

---

## Methodology  

### Dataset Preparation  
- Collected ~5,000 images across **four classes**: three authorized individuals and one intruder class.  
- Used **Roboflow** for annotation with bounding boxes.  
- Split dataset into **70% training, 20% testing, and 10% validation**.  
- Applied augmentations (rotations, brightness adjustments) to increase robustness.  

### Model Training  
- YOLOv8 model configured for **person detection**.  
- Trained for **50 epochs** in Google Colab, using pre-trained weights for fine-tuning.  
- Evaluated using **mAP50** and **mAP50-95** metrics.  

### Model Performance  
- Achieved **mAP50 = 0.995** and **mAP50-95 = 0.831** by epoch 35, showing strong detection performance.  

### **My contribution** - Digital Image Processing  
Applied seven filtering techniques to intruder snapshots to maintain clarity under different lighting and noise conditions:  
- Bicubic Interpolation  
- Median Filtering  
- Ideal High-Pass Filtering  
- Histogram Equalization  
- Edge Detection  
- Otsu Thresholding  
- Adaptive Mean Thresholding  

### GUI Development  
- Developed a **Flask-based interface** for real-time monitoring.  
- Features:  
  - Live streaming from camera feed.  
  - Intruder snapshot display (updated every 5 seconds).  
  - Filtered intruder images for enhanced visualization.  

---

## Challenges Addressed  
- **Computation Power:** Optimized epochs and leveraged Colab GPU.  
- **Appearance Variability:** Collected diverse intruder data (lighting, clothing, posture).  
- **Edge Detection Thresholds:** Automated with Otsu Thresholding to reduce manual tuning.  
- **Real-time GUI Updates:** Implemented watchdog timers to refresh intruder images seamlessly.  
- **Manual Labeling:** Annotated 5,000 images manually due to Roboflow limitations.  

---

## Results  
- Built a working **real-time intruder detection system** with high detection accuracy.  
- Integrated image processing filters to ensure snapshots remain recognizable under varying conditions.  
- Delivered a **GUI interface** for practical, user-friendly monitoring.  

---

## Future Enhancements  
- Expand dataset with more intruder profiles, clothing variations, and environmental conditions.  
- Automate **continuous model updates** with new data.  
- Introduce advanced **behavior analysis and anomaly detection**.  
- Improve GUI interactivity (sliders, parameter tuning, visual heatmaps).  
- Enable **scalability with cloud/edge deployment** for multi-site use.  

---

## Tech Stack  
- **Languages:** Python  
- **Frameworks/Libraries:** OpenCV, YOLOv8, Flask, NumPy, Matplotlib  
- **Tools:** Roboflow (annotation), Google Colab (training)  

---

## Acknowledgments  
Developed as part of the **Digital Image Processing (EDS6397)** course project at the **University of Houston**, under the guidance of **Dr. Lucy Nwosu** 

---
