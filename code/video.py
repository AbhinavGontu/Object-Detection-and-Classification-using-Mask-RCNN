import numpy as np
import argparse
import random
import time
import cv2
import os

os.chdir("C://MASK RCNN PROJECT")


labelsPath='different_classes.txt'
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


weightsPath='inference_graph.pb'
configPath='mask_rcnn_inception.pbtxt'
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

video='traffic.mp4'


def web(boxes,masks):
    
    classID = int(boxes[0, 0, 1, 1])
    confidence = boxes[0, 0, 1, 2]
    (H, W) = image.shape[:2]
    box = boxes[0, 0, 1, 3:7] * np.array([W, H, W, H])
    (startX, startY, endX, endY) = box.astype("int")
    boxW = endX - startX
    boxH = endY - startY
    for i in range(0, boxes.shape[2]):
		# extract the class ID of the detection along with the
		# confidence (i.e., probability) associated with the
		# prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
        if confidence >0.5:
			# scale the bounding box coordinates back relative to the
			# size of the frame and then compute the width and the
			# height of the bounding box
            (H, W) = image.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

			# extract the pixel-wise segmentation for the object,
			# resize the mask such that it's the same dimensions of
			# the bounding box, and then finally threshold to create
			# a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask >0.3)

			# extract the ROI of the image but *only* extracted the
			# masked region of the ROI
            roi = image[startY:endY, startX:endX][mask]

			# grab the color used to visualize this particular class,
			# then create a transparent overlay by blending the color
			# with the ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

			# store the blended ROI in the original frame
            image[startY:endY, startX:endX][mask] = blended

			# draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv2.rectangle(image, (startX, startY), (endX, endY),color, 2)

			# draw the predicted label and associated probability of
			# the instance segmentation on the frame
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(image, text, (startX, startY - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




cap = cv2.VideoCapture(video)



#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('video2.avi',fourcc, 50.0, (640,480))

while cv2.waitKey(1) < 0:
    

    # Get frame from the video
    hasFrame, image = cap.read()
    hasFrame, image = cap.read()
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])



    # Extract the bounding box and mask for each of the detected objects
    web(boxes, masks)
    cv2.imshow("traffic_output", image)



cap.release()
#out.release()

cv2.destroyAllWindows()