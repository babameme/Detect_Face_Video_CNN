# 1. Import libraries
import argparse
import dlib
import cv2
import os

IMAGE_SIZE=224

# Use argparse so we can send the image path from the command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
    help="Image to input image to detect faces")
args = vars(ap.parse_args())

dirname = "E:\\Research\\MachineLearning\\Detect_Face_Video_CNN\\data\\other\\"

new_dirname = "E:\\Research\\MachineLearning\\Detect_Face_Video_CNN\\data\\hihi\\"

# Get the dlib frontal face detector
detector = dlib.get_frontal_face_detector()
star = 0
for file in os.listdir(dirname):
    full_path = dirname + file
    print("Processing file: {}".format(full_path))

    # Load the image using OpenCV
    img = cv2.imread(full_path)

    # Pass de loaded image to the `detector`
    dets = detector(img, 1)
    #print("Number of faces detected: {}".format(len(dets)))

    # Iterate over the found faces and draw a rectangle in the original image.
    for i, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        # cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
        crop_img = img[d.top():d.bottom(),d.left():d.right()]
        #print(crop_img.shape)
        if crop_img.shape[0] <= 0 or crop_img.shape[1] <=0:
            continue
        fix_img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        star = star + 1
        print(star)
        cv2.imwrite(new_dirname + str(star) + ".png", fix_img)
        #cv2.imshow("crop_img", crop_img)
        #cv2.waitKey(0)

# 6. Show image with detected faces
# cv2.imshow('image', img)
# cv2.waitKey(0)
cv2.destroyAllWindows()