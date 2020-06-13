import numpy as np
from Line import Line
import cv2
POINT_PER_LINE = 27
WINDOW_MARGIN = 30
MIN_PIXEL_REQUIRED = 50
def search_no_pre_info(binary_imgae):
    hist = np.sum(binary_imgae[int(binary_imgae.shape[0] / 2):, :], axis=0)
    mid = np.int(hist.shape[0] / 2)
    left_x_base = np.argmax(hist[:mid])
    right_x_base = np.argmax(hist[mid:]) + mid
    window_height = np.int(binary_imgae.shape[0] / POINT_PER_LINE)
    nonzero = binary_imgae.nonzero()
    nonzero_y, nonzero_x = nonzero[0], nonzero[1]

    current_left_x = left_x_base
    current_right_x = right_x_base
    left_lane_inds = []
    right_lane_inds = []

    for idx in range(POINT_PER_LINE):
        window_y_low = binary_imgae.shape[0] - (idx + 1) * window_height
        window_y_high = binary_imgae.shape[0] - idx * window_height
        window_left_x_min = current_left_x - WINDOW_MARGIN
        window_left_x_max = current_left_x + WINDOW_MARGIN
        window_right_x_min = current_right_x - WINDOW_MARGIN
        window_right_x_max = current_right_x + WINDOW_MARGIN
        left_window_inds = ((nonzero_y <= window_y_high) & (nonzero_y >= window_y_low) &  (nonzero_x >= window_left_x_min) & (nonzero_x <= window_left_x_max)).nonzero()[0]
        right_window_inds = ((nonzero_y <= window_y_high) & (nonzero_x >= window_right_x_min) & (nonzero_y >= window_y_low) & (nonzero_x <= window_right_x_max)).nonzero()[0]

        left_lane_inds.append(left_window_inds)
        right_lane_inds.append(right_window_inds)

        if len(left_window_inds) > MIN_PIXEL_REQUIRED:
            current_left_X = np.int(np.mean(nonzero_x[left_window_inds]))
        if len(right_window_inds) > MIN_PIXEL_REQUIRED:
            current_right_X = np.int(np.mean(nonzero_x[right_window_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    left_x= nonzero_x[left_lane_inds]
    left_y =  nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]
    left_poly = np.polyfit(left_y, left_x, 2)
    right_poly = np.polyfit(right_y, right_x, 2)
    
    left_line = Line()
    right_line = Line()
    left_line.poly = left_poly
    right_line.poly = right_poly

    plot_y = np.linspace(0, binary_imgae.shape[0] - 1, binary_imgae.shape[0])

    left_plot_x = left_poly[0] * plot_y ** 2 + left_poly[1] * plot_y + left_poly[2]
    right_plot_x = right_poly[0] * plot_y ** 2 + right_poly[1] * plot_y + right_poly[2]

    left_line.raw_detected_x.append(left_plot_x)
    right_line.raw_detected_x.append(right_plot_x)

    if len(left_line.raw_detected_x) > 10:
        left_avg_line = left_line.smooth(binary_imgae.shape[0], 10)
        left_avg_fit = np.polyfit(plot_y, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * plot_y ** 2 + left_avg_fit[1] * plot_y + left_avg_fit[2]
        left_line.poly = left_avg_fit
        left_line.x, left_line.y = left_fit_plotx, plot_y
    else:
        left_line.poly = left_poly
        left_line.x, left_line.y = left_plot_x, plot_y

    if len(right_line.raw_detected_x) > 10:
        right_avg_line = right_line.smooth(binary_imgae.shape[0], 10)
        right_avg_fit = np.polyfit(plot_y, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * plot_y ** 2 + right_avg_fit[1] * plot_y + right_avg_fit[2]
        right_line.poly = right_avg_fit
        right_line.x, right_line.y = right_fit_plotx, plot_y
    else:
        right_line.poly = right_poly
        right_line.x, right_line.y = right_plot_x, plot_y

    left_line.start_x= left_line.x[len(left_line.x)-1]
    left_line.end_x = left_line.x[0]
    right_line.start_x = right_line.x[len(right_line.x)-1]
    right_line.end_x = right_line.x[0]

    return left_line, right_line

def seatch_with_pre_info(binary_imgae, pre_left_lane, pre_right_lane):
    nonzero_y, nonzero_x = binary_imgae.nonzero()
    pre_left_poly = pre_left_lane.poly
    pre_right_poly = pre_right_lane.poly
    left_x_min = pre_left_poly[0] * nonzero_y ** 2 + pre_left_poly[1] * nonzero_y + pre_left_poly[2] - WINDOW_MARGIN
    left_x_max = pre_left_poly[0] * nonzero_y ** 2 + pre_left_poly[1] * nonzero_y + pre_left_poly[2] + WINDOW_MARGIN
    right_x_min = pre_right_poly[0] * nonzero_y ** 2 + pre_right_poly[1] * nonzero_y + pre_right_poly[2] - WINDOW_MARGIN
    right_x_max = pre_right_poly[0] * nonzero_y ** 2 + pre_right_poly[1] * nonzero_y + pre_right_poly[2] + WINDOW_MARGIN
    left_inds = ((nonzero_x >= left_x_min) & (nonzero_x <= left_x_max)).nonzero()[0]
    right_inds = ((nonzero_x >= right_x_min) & (nonzero_x <= right_x_max)).nonzero()[0]


    left_x, left_y = nonzero_x[left_inds], nonzero_y[left_inds]
    right_x, right_y = nonzero_x[right_inds], nonzero_y[right_inds]

    current_left_poly = np.polyfit(left_y, left_x, 2)
    current_right_poly = np.polyfit(right_y, right_x, 2)

    plot_y = np.linspace(0, binary_imgae.shape[0] - 1, binary_imgae.shape[0])

    # Fit line
    left_plot_x = current_left_poly[0] * plot_y ** 2 + current_left_poly[1] * plot_y + current_left_poly[2]
    right_plot_x = current_right_poly[0] * plot_y ** 2 + current_right_poly[1] * plot_y + current_right_poly[2]

    left_x_avg = np.average(left_plot_x)
    right_x_avg = np.average(right_plot_x)

    current_left_line = Line()
    current_right_line = Line()
    current_left_line.raw_detected_x.append(left_plot_x)
    current_right_line.raw_detected_x.append(right_plot_x)

    if len(current_left_line.raw_detected_x) > 10:  
        left_avg_line = current_left_line.smooth(binary_imgae.shape[0], 10)
        left_avg_fit = np.polyfit(plot_y, left_avg_line, 2)
        left_fit_plot_x = left_avg_fit[0] * plot_y ** 2 + left_avg_fit[1] * plot_y + left_avg_fit[2]
        current_left_line.poly = left_avg_fit
        current_left_line.x, current_left_line.y = left_fit_plot_x, plot_y
    else:
        current_left_line.poly = current_left_poly
        current_left_line.x, current_left_line.y = left_plot_x, plot_y

    if len(current_right_line.raw_detected_x) > 10: 
        right_avg_line = current_right_line.smooth(binary_imgae.shape[0], 10)
        right_avg_fit = np.polyfit(plot_y, right_avg_line, 2)
        right_fit_plot_x = right_avg_fit[0] * plot_y ** 2 + right_avg_fit[1] * plot_y + right_avg_fit[2]
        current_right_line.poly = right_avg_fit
        current_right_line.x, current_right_line.y = right_fit_plot_x, plot_y
    else:
        current_right_line.poly = current_right_poly
        current_right_line.x, current_right_line.y = right_plot_x, plot_y

    current_left_line.start_x= current_left_line.x[len(current_left_line.x)-1]
    current_left_line.end_x = current_left_line.x[0]
    current_right_line.start_x = current_right_line.x[len(current_right_line.x)-1]
    current_right_line.end_x = current_right_line.x[0]
    
    return current_left_line, current_right_line


