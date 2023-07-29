import cv2 
import mediapipe as mp 
import numpy as np
from keras.models import load_model
import time 
from gtts import gTTS
import os
from playsound import playsound
import threading


def playmusic(filepath):
	playsound(filepath)


timesteps=24
labels = np.load('labels.npy')

classes = labels.copy()

labels = []
for i in classes:
	if i!="still":
		labels.append(i)
labels = np.array(labels)

print(labels)
print(classes)

model = load_model('model.h5')
msg = ""
cmsg = ""
in_data = [[0.0]*66]*timesteps

def shifting(lst):
	for i in range(timesteps-1):
		in_data[i] = in_data[i+1]
	in_data[timesteps-1] = lst

def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
		return True 
	return False

def averg():
	s = 0 
	for i in guesses:
		s = s+i
	return int(s/len(guesses))

cap = cv2.VideoCapture(0)

holis = mp.solutions.pose
drawing = mp.solutions.drawing_utils 
holisO = holis.Pose(static_image_mode=False)

final_data = []
lst = []
c = 0

t2 = 0
show_intro = True
time_init = False
play_audio = False

guesses = []

while True:
	stime = time.time()

	_, frame = cap.read()

	window = np.zeros((940,940,3), dtype="uint8")

	frame = cv2.flip(frame, 1)
	res = holisO.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

	drawing.draw_landmarks(frame, res.pose_landmarks, holis.POSE_CONNECTIONS)

	if res.pose_landmarks:
		finalres = res.pose_landmarks.landmark

	if res.pose_landmarks and inFrame(finalres):
		lst = []
		for i in res.pose_landmarks.landmark:
			lst.append(i.x - finalres[0].x)
			lst.append(i.y - finalres[0].y)

		shifting(lst)
		pred = model.predict(np.array([in_data]))

		print(classes[np.argmax(pred)])

		if classes[np.argmax(pred)] != "still" and pred[0][np.argmax(pred)] > 0.8:
			if len(guesses) < 20:
				guesses.append(np.argmax(pred))
			else:
				s = 0 
				for i in guesses:
					s = s+i 
				print(s/len(guesses))
				print(classes)
				print(classes[int(s/len(guesses))])
				break

		print(pred)
		cv2.putText(window, classes[np.argmax(pred)], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

		#cv2.putText(frame, classes[np.argmax(pred)], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


		# if pred[0][np.argmax(pred)]>0.8 and classes[np.argmax(pred)]!="still":
		# 	msg = classes[np.argmax(pred)]

		# 	cv2.putText(window, msg + classes[rand], (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2)


	else:
		cv2.putText(frame, "Make Sure full body in frame", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

	etime = time.time()

	cv2.putText(frame, f"{int(1/(etime-stime))}", (50,340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

	window[420:900, 170:810, :] = frame

	cv2.imshow("window", window)

	if cv2.waitKey(1) == 27:
		cap.release()
		cv2.destroyAllWindows()
		break
	