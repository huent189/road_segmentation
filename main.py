import cv2
from road_detection import detect_road

if __name__ == '__main__':
    input_video = "test/sample.mp4"
    cap = cv2.VideoCapture(input_video)
    out = cv2.VideoWriter('test/detection_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,360))      
    left_line = None
    right_line = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            left_line, right_line, road = detect_road(frame, left_line, right_line)
            out.write(road)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
