import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
from datetime import datetime
D0_lowpass = 50 # Cut-off frequency
D0_highpass = 50 # Cut-off frequency
def apply_low_pass_filter(image, D0_lowpass):
    # Apply 2D Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
 
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
 
    # Create Ideal Low Pass filter mask
    lpf_mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if dist <= D0_lowpass:
                lpf_mask[i, j] = 1
 
    # Apply the mask to the centered spectrum
    lpf_filtered = fshift * lpf_mask
 
    # Apply inverse shift
    f_inverse_shifted = np.fft.ifftshift(lpf_filtered)
 
    # Calculate the inverse DFT
    restored_image = np.fft.ifft2(f_inverse_shifted)
    restored_image = np.abs(restored_image)  # Take the magnitude
 
    return restored_image
 
def apply_high_pass_filter(image, D0_highpass):
    # Apply 2D Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
 
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
 
    # Create Ideal High Pass filter mask
    hpf_mask = np.ones((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if dist <= D0_highpass:
                hpf_mask[i, j] = 0
 
    # Apply the mask to the centered spectrum
    hpf_filtered = fshift * hpf_mask
 
    # Apply inverse shift
    hpf_inverse_shifted = np.fft.ifftshift(hpf_filtered)
 
    # Calculate the inverse DFT
    restored_image = np.fft.ifft2(hpf_inverse_shifted)
    restored_image = np.abs(restored_image)  # Take the magnitude
 
    return restored_image
 
def histogram_equalization(image):
    # Perform histogram equalization
    # We then apply histogram equalization using the cv2.equalizeHist() function,
    # which performs contrast adjustment by redistributing pixel intensities.
    # Convert the image to YUV color space
    image_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
 
    # Apply histogram equalization to the Y channel (luminance)
    image_yuv[:,:,0] = cv.equalizeHist(image_yuv[:,:,0])
 
    # Convert the image back to BGR color space
    equalized_image = cv.cvtColor(image_yuv, cv.COLOR_YUV2BGR)
    return equalized_image
 
def resize_image(image, scale_factor):
    resized_image = cv.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
    return resized_image

def apply_edge_detection(image):
    # Apply Gaussian Blur
    blur = cv.GaussianBlur(image, (3,3), 0)
    # Determine the lower and upper thresholds using Otsu's method
    ret, threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lower = 0.5 * ret
    upper = 1.5 * ret
    # Apply Canny edge detection with the automatically determined thresholds
    canny = cv.Canny(image, lower, upper)
    return canny

def apply_average_filter(image, filter_size):
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size **2)
    smoothed_image = cv.filter2D(image, -1, kernel)
    return smoothed_image

def apply_Otsu_Thresholding(image):
    # Apply Otsu's thresholding
    _, otsu_thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return otsu_thresh
 
def apply_AdaptiveMean_Thresholding(image):
    # Apply adaptive thresholding with different methods
    adaptive_mean_thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    return adaptive_mean_thresh

def saveFilteredImg(ImgName, imgObj, filter_type):
    # Create a folder to save the filtered images based on filter_type
    output_folder = os.path.join("Filtered_Images", filter_type)
    os.makedirs(output_folder, exist_ok=True)
    # Save filtered images
    img_output_path = os.path.join(output_folder, ImgName)
    cv.imwrite(img_output_path, imgObj)


def process(imagepath):
    image_path=imagepath
    #image_path = "myImage.jpg"
    #timestamp=image_path.split('\\')[-1].split('.')[0].split("_")[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_image = cv.imread(image_path)
    original_image_RGB = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    gray_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    assert gray_img is not None, "file could not be read, check with os.path.exists()"
    
    D0_lowpass = 50 # Cut-off frequency
    D0_highpass = 50 # Cut-off frequency
    scale_factor = 0.5 # Resize scale factor
    
    # Apply low-pass filter
    lpf_filtered_image = apply_low_pass_filter(gray_img, D0_lowpass)
    
    # Apply high-pass filter
    hpf_filtered_image = apply_high_pass_filter(gray_img, D0_highpass)
    
    # Perform histogram equalization
    histogram_equalized_image = histogram_equalization(original_image)

    edge_detected_image= apply_edge_detection(gray_img)

    Threshold1_image = apply_AdaptiveMean_Thresholding(gray_img)

    Threshold2_image = apply_Otsu_Thresholding(gray_img)

    #Implement a low-pass filter - averaging filter
    median_filtered_image_jp = apply_average_filter(original_image, 10)
    
    # Resize images using bicubic interpolation
    #lpf_resized_image = resize_image(lpf_filtered_image, scale_factor)
    #hpf_resized_image = resize_image(hpf_filtered_image, scale_factor)
    
    # Create a folder to save the filtered images
    original_images = "original_images"
    lowpass_image = "lowpass_images"
    median_filtered_image = "median_filtered_images"
    highpass_image = "highpass_images"
    Histogram_image = "Histogram_images"
    EdgeDetection_image = "EdgeDetection_images"
    Thresholding1_image = "Thresholding1_images"
    Thresholding2_image = "Thresholding2_images"
    
    if not os.path.exists(original_images):
        os.makedirs(original_images, exist_ok=True)
    if not os.path.exists(median_filtered_image):
        os.makedirs(median_filtered_image, exist_ok=True)
    if not os.path.exists(highpass_image):
        os.makedirs(highpass_image, exist_ok=True)
    if not os.path.exists(Histogram_image):
        os.makedirs(Histogram_image, exist_ok=True)
    if not os.path.exists(EdgeDetection_image):
        os.makedirs(EdgeDetection_image, exist_ok=True)
    if not os.path.exists(Thresholding1_image):
        os.makedirs(Thresholding1_image, exist_ok=True)
    if not os.path.exists(Thresholding2_image):
        os.makedirs(Thresholding2_image, exist_ok=True)
    
    # Save filtered images
    originalImg_output_path = os.path.join(original_images, f'Original_image.jpg')
    #lpf_output_path = os.path.join(lowpass_image, f'lpf_filtered_image.jpg')
    median_output_path = os.path.join(median_filtered_image, f'median_filtered_image.jpg')
    hpf_output_path = os.path.join(highpass_image, f'hpf_filtered_image.jpg')
    histogram_equalized_output_path = os.path.join(Histogram_image, f'histogram_equalized_image.jpg')
    EdgeDetection_output_path = os.path.join(EdgeDetection_image, f'edge_detected_image.jpg')
    Thresholding1_output_path = os.path.join(Thresholding1_image, f'Threshold1_image.jpg')
    Thresholding2_output_path = os.path.join(Thresholding2_image, f'Threshold2_image.jpg')
    cv.imwrite(originalImg_output_path, original_image)
    cv.imwrite(median_output_path, median_filtered_image_jp)
    cv.imwrite(hpf_output_path, hpf_filtered_image)
    cv.imwrite(histogram_equalized_output_path, histogram_equalized_image)
    cv.imwrite(EdgeDetection_output_path, edge_detected_image)
    cv.imwrite(Thresholding1_output_path, Threshold1_image)
    cv.imwrite(Thresholding2_output_path, Threshold2_image)
    
    image_name = os.path.basename(image_path)
    saveFilteredImg('Gray_Image_'+image_name, gray_img, 'Gray_Image')
    saveFilteredImg('lpf_filtered_'+image_name, lpf_filtered_image, 'LPF')
    saveFilteredImg('hpf_filtered_'+image_name, hpf_filtered_image, 'HPF')
    saveFilteredImg('histogram_equalized_'+image_name, histogram_equalized_image, 'Histogram_Equalized')
    saveFilteredImg('median_filtered_'+image_name, median_filtered_image_jp, 'Median_Filtered')
    saveFilteredImg('edge_detected_'+image_name, edge_detected_image, 'Edge_Detected')
    saveFilteredImg('otsu_threshold_'+image_name, Threshold2_image, 'Otsu_Threashold')
    saveFilteredImg('adaptiveMean_threshold_'+image_name, Threshold1_image, 'AdaptiveMean_Thresholded')
    #saveFilteredImg('adaptiveGaussian_threshold_'+image_name, adaptiveGaussian_Thresholded, 'AdaptiveGaussian_Thresholded')
    #saveFilteredImg('sauvola_threshold_'+image_name, sauvola_Thresholded, 'Sauvola_Thresholded')
    #saveFilteredImg('niblack_threshold_'+image_name, niblack_Thresholded, 'Niblack_Thresholded')

    #print(f"LPF Filtered and Resized image saved to: {lpf_output_path}")
    print(f"HPF Filtered and Resized image saved to: {hpf_output_path}")
    
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 4, 1)
    # plt.imshow(original_image)
    # plt.title('Original Image')
    # plt.axis('off')
    
    # plt.subplot(1, 4, 2)
    # plt.imshow(hpf_filtered_image, cmap='gray')
    # plt.title('HPF Image')
    # plt.axis('off')
    
    # plt.subplot(1, 4, 3)
    # plt.imshow(lpf_filtered_image, cmap='gray')
    # plt.title('LPF Image')
    # plt.axis('off')
    
    # plt.subplot(1, 4, 4)
    # plt.imshow(histogram_equalized_image)
    # plt.title('Hist EQ Image')
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":

    process(r"C:\Users\racha\OneDrive\Desktop\DIP\captured_images\image_1712892236..jpg")
