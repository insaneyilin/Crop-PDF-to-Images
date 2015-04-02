# USAGE: Put images in the "images" directory
# run the command "python pdfcrop.py"
# results will be in the "output" directory

import os
import numpy
import cv2


files_dir = "./images"
files = os.listdir(files_dir)
output_dir = "./output"

for image_filename in files:
    # detect edges
    image = cv2.imread(files_dir + "/" + image_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    # extract contours
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    poly_contours = []
    for c in cnts[0:6]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            poly_contours.append(approx)
    
    # save results
    index = 0
    for contours in poly_contours:
        rect = cv2.boundingRect(contours)
        cv2.imwrite(output_dir + "/" + image_filename + "_" + str(index) + ".jpg", 
                           image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
        index += 1
        