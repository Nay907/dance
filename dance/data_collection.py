import cv2
import mediapipe as mp 
import numpy as np 
import time

def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
		return True 
	return False

data_name = input("Enter the dance step song name : ")
data_size = 8
timesteps = 24

print(data_name, data_size, timesteps)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'dancing/{data_name}.avi', fourcc, 20.0, (640,480))
anim_done=False

cap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
pose = mp.solutions.pose
pose_d = pose.Pose(static_image_mode=False)

c = 0
t=0

a = []
b = []

while True:
	_, frm = cap.read()

	anim = np.zeros_like(frm)

	frm = cv2.flip(frm, 1)

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	res = pose_d.process(rgb)

	lst = []

	if res.pose_landmarks:
		finalres = res.pose_landmarks.landmark

	if res.pose_landmarks and inFrame(finalres):	

		for i in res.pose_landmarks.landmark:
			lst.append(i.x - finalres[0].x)
			lst.append(i.y - finalres[0].y)

		if c< timesteps:
			a.append(lst)
			c = c+1
			cv2.line(frm, (10,10), (int(c*25),10), (255,0,0), 3)

		else:
			b.append(a)
			t = t+1
			c=0
			a=[]

		drawing.draw_landmarks(anim, res.pose_landmarks, pose.POSE_CONNECTIONS)

		out.write(anim)


	else:
		cv2.putText(frm, "Make Sure full body in frame", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	

	drawing.draw_landmarks(frm, res.pose_landmarks, pose.POSE_CONNECTIONS)


	cv2.putText(frm, str(t), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
	cv2.imshow("Window", frm)


	if cv2.waitKey(1) == 27 or t==data_size:
		cap.release()
		cv2.destroyAllWindows()
		break

b = np.array(b)
np.save(f"{data_name}.npy", b)
print(b.shape)
print("Done with data collection for "+data_name)