# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from gtts import gTTS
import numpy as np
import pandas as pd
import openpyxl
import datetime 
import imutils
import pyttsx3
import time
import cv2
import os														

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)..[[we done]]

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	count = 0
	
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))
			
	count = len(faces)
	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds,count)

def SpeakText(command):
     
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()
	
	
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame,width=750)
	frame = imutils.resize(frame,height=800)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds,count) = detect_and_predict_mask(frame, faceNet, maskNet)
	
	#initialize the mask and without mask count
	c3 = 0;
	c4 = 0;
	c5 = 0;
	
	#loop the preds to get the values of with_mask and without_mask
	for i in range(count):
		(mask,withoutMask) = preds[i]
		if(mask > withoutMask):
			c3=c3+1
		elif(mask < withoutMask):
			c4=c4+1
		else:
			c5=1
		
	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred	
		
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		
		# include the probability in the label(% percentage)
		#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)..........
		
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
		#facecount collector...
		label2 = "Total Faces: "
		l3 = "Wearing Mask: "
		l4 = "Not Wearing Mask: "
		label2 = "{0:s}{1:d}  {2:s}{3:}  {4:s}{5:}".format(label2,count,l3,c3,l4,c4)
		color=(255,255,255)
		
		#storing Data in .txt file
		current_time = str(datetime.datetime.now())
		label3 = "{0:s}  Total Faces:{1:d} Wearing Mask:{2:d}  Not Wearing Mask:{3:d} \n".format(current_time,count,c3,c4)
		f = open("Datafile.txt", "a")
		f.write(label3)
		f.close()
		
		
		#display in command prompt
		#print(label2)
		
		# display the face count on the Frame.
		cv2.putText(frame,label2, (200,75),					#(image,text,origin,font,font scale,color,thickness)
			cv2.FONT_HERSHEY_SIMPLEX,0.80, color, 2)
			
	# show the output frame
	cv2.imshow("Real Time Face-Mask Detector",frame)
	key = cv2.waitKey(1) & 0xFF
		
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	
# voice over..
s1 = "People Present here is: ",count
s2 = "Awesome , Everyone Wearing their Mask properly" 
s3 = "count of people Wearing Mask is: ",c3
s4 = "count of people Not Wearing Mask ",c4
s5 = "Please Everyone Wear your mask properly."

SpeakText(s1)

if c3==count:
	speakText(s2)
else:
	SpeakText(s3)
if c4 > 0:
	SpeakText(s4)
	SpeakText(s5)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()