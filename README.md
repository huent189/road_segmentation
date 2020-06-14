# Overview
When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project we will detect lane lines in images using Python and OpenCV. This project is a part of application we use when participating Digital Race competition
# Project Approach
My pipeline consisted of 5 steps
1. Thresholding
- Using HLS and gradient convert and calculate threshold.
- After that, combine those one into a single image.
2. Bird view transform
- Using wrap perspective transform to transform road image into bird view
3. Lane line finding using sliding window
4. Convert the bird view image in which found lane is drawn to normal view, then combine it to original image.
# Shortcomings:
Since this pipeline use threshold to detect lane line, it is sensitive to environmental changes. We can overcome it by using deep learning based approach
# Project result
You can view the sample input and output in test folder
