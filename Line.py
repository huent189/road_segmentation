import numpy as np
class Line:
    def __init__(self):
        # self.window_margin = 30
        self.raw_detected_x = []
        self.poly = [np.array([False])]
        # self.radius = None
        self.start_x = None
        self.end_x = None
        self.x = []
        self.y = None 
    def smooth(self, max_row, prev_n_line=3):
        lines = np.squeeze(self.raw_detected_x)
        avg_line = np.zeros(max_row)
        for i, line in enumerate(reversed(lines)):
            if i == prev_n_line:
                break
            avg_line += line

        avg_line /= prev_n_line

        return avg_line