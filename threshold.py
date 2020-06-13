import numpy as np
import cv2

def binary_threshold(img, threshold):
    thresholded = np.zeros_like(img, np.uint8)
    thresholded[(img >= threshold[0]) & (img <= threshold[1])] = 255
    return thresholded

def magnitude_threshold(sobel_x, sobel_y, threshold):
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    return binary_threshold(magnitude, threshold)

def direction_threshold(sobel_x, sobel_y, threshold):
    direction = np.arctan2(sobel_x, sobel_y)
    return binary_threshold(direction, threshold)

def threshold_use_gradient(img, th_sobel_x, th_sobel_y, th_magnitude, th_direction):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x_img = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    sobel_y_img = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel_x = np.uint8(255 * sobel_x_img / np.max(sobel_x_img))
    scaled_sobel_y = np.uint8(255 * sobel_y_img / np.max(sobel_y_img))
    
    sobel_x_thresholded = binary_threshold(scaled_sobel_x, th_sobel_x)
    sobel_y_thresholded = binary_threshold(scaled_sobel_y, th_sobel_y)
    magnitude_thresholded = magnitude_threshold(sobel_x_img, sobel_y_img, th_magnitude)
    direction_thresholded = direction_threshold(sobel_x_img, sobel_y_img, th_direction)
    gradient_thresholded = np.zeros_like(sobel_x_thresholded, np.uint8)
    gradient_thresholded[(((sobel_x_thresholded > 0) & (sobel_y_thresholded > 0)) | ((sobel_x_thresholded > 0) & (magnitude_thresholded > 0) & (direction_thresholded > 0)))] = 255
    return gradient_thresholded

def threshold_use_hls(img, th_h, th_l, th_s):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    h_thresholded = binary_threshold(H, th_h)
    l_thresholded = binary_threshold(L, th_l)
    s_thresholded = binary_threshold(S, th_s)
    hls_thresholded = np.zeros_like(H, np.uint8)
    hls_thresholded[((s_thresholded > 0) & (l_thresholded == 0)) | ((s_thresholded == 0) & (h_thresholded > 0) & (l_thresholded > 0))] = 255
    return hls_thresholded