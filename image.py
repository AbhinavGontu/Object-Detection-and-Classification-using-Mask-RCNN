import numpy as np
import random
import time
import cv2
import os

os.chdir('C:\MASK RCNN PROJECT')


labelsPath='different_classes.txt'
LABELS = open(labelsPath).read().strip().split("\n")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


weightsPath='inference_graph.pb'
configPath='mask_rcnn_inception.pbtxt'
#Reads a network model stored in TensorFlow framework's format.
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)



image = cv2.imread('image1.jpg')
print(image.shape)

(H, W) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
print(boxes.shape)


#min and max values of x,y coordinates
classID = int(boxes[0, 0, 1, 1])
confidence = boxes[0, 0, 1, 2]
print(classID)
box = boxes[0, 0, 1, 3:7] * np.array([W, H, W, H])
(startX, startY, endX, endY) = box.astype("int")
boxW = endX - startX
boxH = endY - startY


for i in range(0, boxes.shape[2]):

	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]
    print(classID)
	if confidence >0.5:

		(H, W) = image.shape[:2]
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
		boxW = endX - startX
		boxH = endY - startY

		mask = masks[i, classID]
		mask = cv2.resize(mask, (boxW, boxH),
			interpolation=cv2.INTER_NEAREST)
		mask = (mask >0.3)


		roi = image[startY:endY, startX:endX][mask]


		color =COLORS[classID]
		blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")


		image[startY:endY, startX:endX][mask] = blended


		color = [int(c) for c in color]
		cv2.rectangle(image, (startX, startY), (endX, endY),color, 2)


		text = "{}: {:.4f}".format(LABELS[classID], confidence)
		cv2.putText(image, text, (startX, startY - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.imshow("Output", image)


cv2.imwrite('test3_output.jpeg',image)
        
cv2.destroyAllWindows()
