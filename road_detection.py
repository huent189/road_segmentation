import numpy as np
import cv2
from threshold import threshold_use_gradient, threshold_use_hls
from line_utils import search_no_pre_info, seatch_with_pre_info
from visualize import draw_lane

WRAP_IMG_SIZE = (420, 540)
#fine-tunned parameter
th_sobel_x = (35, 100)
th_sobel_y = (30, 255)
th_magnitude = (30, 255)
th_direction = (0.7, 1.3)
th_h = (254, 255)
th_l = (0, 141)
th_s = (0, 255)
perspective_src = np.float32([[218, 92], [299, 40], [374, 40], [472, 92]])
perspective_dst = np.float32([[60, 540], [60, 0], [360, 0], [360, 540]])

def detect_road(frame, pre_left_line = None, pre_right_line = None):
    frame = cv2.resize(frame, None, fx=1/2, fy=1/2 , interpolation=cv2.INTER_AREA)
    rows, cols  = frame.shape[:2]
    ROI = frame[(rows // 2):].copy()
    gradient = threshold_use_gradient(ROI, th_sobel_x, th_sobel_y, th_magnitude, th_direction)
    hls = threshold_use_hls(ROI, th_h, th_l, th_s)
    threshold_combined = np.zeros_like(hls, np.uint8)
    threshold_combined[((gradient > 0) & (hls > 0))] = 255
    M = cv2.getPerspectiveTransform(perspective_src, perspective_dst)
    warp_img = cv2.warpPerspective(threshold_combined, M, WRAP_IMG_SIZE, cv2.INTER_LINEAR)
    if(pre_left_line):
        left_line, right_line = seatch_with_pre_info(warp_img, pre_left_line, pre_right_line)
    else:
        left_line, right_line = search_no_pre_info(warp_img)
    
    wrapped_road_img = draw_lane([warp_img.shape[0], warp_img.shape[1], 3], left_line, right_line)
    inverse_M = cv2.getPerspectiveTransform(perspective_dst, perspective_src)
    road_img = cv2.warpPerspective(wrapped_road_img, inverse_M, (cols, rows // 2))
    color_result = np.zeros_like(frame)
    color_result[(rows//2):, :] = road_img
    # road_img[:rows // 3, :] = [0,0,0]
    return left_line, right_line, cv2.addWeighted(frame, 1, color_result, 0.3, 0)

