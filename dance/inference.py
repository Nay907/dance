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

times = 0

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

points = 0

was = ""
did = ""

prev_rand = -1

while True:
	stime = time.time()

	_, frame = cap.read()

	window = np.zeros((940,940,3), dtype="uint8")

	cv2.putText(window, f"score : {points}", (670,90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 3)
	
	if was!="":
		cv2.putText(window, f"was : {was}", (160,290), cv2.FONT_HERSHEY_SIMPLEX, 1.6 , (0,0,255), 3)
		cv2.putText(window, f"did : {did}", (160,340), cv2.FONT_HERSHEY_SIMPLEX, 1.6 , (0,0,255), 3)

	frame = cv2.flip(frame, 1)
	res = holisO.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

	frame = cv2.blur(frame, (12,12))
	#frame = np.zeros_like(frame)
	drawing.draw_landmarks(frame, res.pose_landmarks, holis.POSE_CONNECTIONS,
							connection_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=6 ),
							 landmark_drawing_spec=drawing.DrawingSpec(color=(255,0,0), circle_radius=3, thickness=3))

	if show_intro:
		if not(time_init):
			t1 = time.time()
			time_init=True

			rand = np.random.randint(0, labels.shape[0])
			while prev_rand==rand:
				rand = np.random.randint(0, labels.shape[0])


		if abs(time.time() - t1) < 3:  
			cv2.putText(window, "Ready ?", (250,450), cv2.FONT_HERSHEY_DUPLEX, 3, (0,255,0), 3)
		
		else:
			cv2.putText(window, "Listen and guess dance moves !", (100,450), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,255,0), 2)
			if not(play_audio):
				t = threading.Thread(target=playmusic, args=[f"music/{labels[rand]}.wav"])
				t.start()
				play_audio=True
				t2 = time.time()
		if abs(t2-time.time()) > 2 and play_audio:
			show_intro=False

	else:
		window[420:900, 170:810, :] = frame
		cv2.putText(window, "Now it's your turn !", (130,210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
		if res.pose_landmarks:
			finalres = res.pose_landmarks.landmark

		if res.pose_landmarks and inFrame(finalres):
			lst = []
			for i in res.pose_landmarks.landmark:
				lst.append(i.x - finalres[0].x)
				lst.append(i.y - finalres[0].y)

			shifting(lst)
			pred = model.predict(np.array([in_data]))


			cv2.putText(frame, classes[np.argmax(pred)], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

			# print("================ ", pred[0][np.argmax(pred)], classes[np.argmax(pred)])
			times = times + 1
			if classes[np.argmax(pred)] != "still" and pred[0][np.argmax(pred)] > 0.70:
					
				if len(guesses) < 10:
					if times > 24:
						print(times, classes[np.argmax(pred)], pred[0][np.argmax(pred)])
						guesses.append(np.argmax(pred))

				else:
					s_dict={}
					for i in guesses:
						if i not in s_dict:
							s_dict[i] = 0
						else:
							s_dict[i] = s_dict[i]+1

					mx = -1
					s = -1
					for k,v in s_dict.items():
						if v > mx:
							mx=v
							s=k

					if classes[s] == labels[rand]:
						points = points+1
						print("YOU WON >>>>>>>>>>>>>>>>>>>>>")
					else:
						points = points - 1
						print("YOU LOOSE -------------------")

					was = labels[rand]
					did = classes[s]
					print("="*100)
					show_intro=True
					time_init=False
					play_audio=False
					guesses=[]
					times=0

					prev_rand = rand

		else:
			cv2.putText(frame, "Make Sure full body in frame", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

		etime = time.time()

		cv2.putText(frame, f"{int(1/(etime-stime))}", (50,340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

	

	cv2.imshow("window", window)

	if cv2.waitKey(1) == 27:
		cap.release()
		cv2.destroyAllWindows()
		break
	