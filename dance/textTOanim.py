import numpy as np 
import cv2
import numpy as np 
import os  

def texttoanimation(string):
	anims = []
	for npy in os.listdir():
		if npy.split('.')[-1] == 'npy' and not(npy.split('.')[0]=="labels"): 
			anims.append(npy.split('.')[0])

	print("="*50)
	print(anims)

	txt = string.split(' ')

	for i in txt:
		if i in anims:
			cap = cv2.VideoCapture('animations/'+i+'.avi')

			while True : 
				ret, frm = cap.read()

				if ret:
					cv2.putText(frm, i, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
					cv2.imshow("window", frm)

					cv2.waitKey(80)

				else:
					cap.release()
					break
		else:
			print("do not have animation for "+i)
	cv2.destroyAllWindows()