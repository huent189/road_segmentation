import numpy as np
import cv2

def draw_lane(output_shape, left_line, right_line, left_color = (0, 0, 255), road_color = (0, 255, 0), right_color = (255, 0, 0)):
    display_img = np.zeros(output_shape)
    window_margin = 30
    left_plot_x, right_plot_x = left_line.x, right_line.x
    ploty = left_line.y

    pts_left = np.array([np.transpose(np.vstack([left_plot_x+window_margin/5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plot_x-window_margin/5, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(display_img, np.int_([pts]), road_color)
    return display_img
