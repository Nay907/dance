import tkinter as tk
from data_collection import data_collection
from data_prep_train import data_prep
from inference import inference
import tkinter.font as font
from tkinter import Text
from textTOanim import texttoanimation
import speech_recognition as sr

window = tk.Tk()
window.title("sign language")
window.geometry("480x640")

def show():
	if T.get("1.0", END) != "" and T1.get("1.0", END) != "" and T2.get("1.0", END) != "":
		data_collection(T.get("1.0", END), int(T1.get("1.0", END)), int(T2.get("1.0", END)))
	else:
		print("Enter some value")
		errlbl['text'] = "Fill values of all fields!"

def train():
	if T1.get("1.0", END) != "":
		data_prep(int(T1.get("1.0", END)))

	else:
		print("Enter some value")
		errlbl['text'] = "Fill values of Data Size fields!"

def inf():
	if T2.get("1.0", END) != "":
		inference(int(T2.get("1.0", END)))

	else:
		print("Enter some value")
		errlbl['text'] = "Fill values of Timesteps fields!"

def texttoanim():
	if T4.get("1.0", END) != "":
		texttoanimation(T4.get("1.0", END))

	else:
		print("Enter some value")
		errlbl['text'] = "Fill values of Text fields!"

def voicetoanim():
	r = sr.Recognizer()
	with sr.Microphone() as m:
		print("Start speaking")
		r.adjust_for_ambient_noise(m)
		errlbl['text'] = "Start Speaking for 3 sec!"
		try:
			audio = r.listen(m, timeout=3)
			text = r.recognize_google(audio)
			errlbl['text'] = "you spoke : " + text
			print("you spoke : ", text)
			texttoanimation(text)
		except:
			errlbl['text'] = "unable to understand try again !"

frame1 = tk.Frame(window)
lbl = tk.Label(frame1, text="Data Name ")
lbl2 = tk.Label(frame1, text="Data Size ")
lbl3 = tk.Label(frame1, text="Timesteps ")

errlbl = tk.Label(frame1, text="",  fg="red")

lbl4 = tk.Label(frame1, text="Enter Text ")

T = Text(frame1, height = 3, width = 20)
T1 = Text(frame1, height = 3, width = 20)
T2 = Text(frame1, height = 3, width = 20)

T4 = Text(frame1, height = 3, width = 20)

T.grid(row=1, column=1)
T1.grid(row=2, column=1)
T2.grid(row=3, column=1)

T4.grid(row=5, column=1, pady=(20, 10))

lbl.grid(row=1, column=0)
lbl2.grid(row=2, column=0)
lbl3.grid(row=3, column=0)

lbl4.grid(row=5, column=0)
errlbl.grid(row=7, columnspan=3)

END="end-1c"
btn_font = font.Font(size=15)
btn1 = tk.Button(frame1,fg="red", text="add data",command=show, height=5, width=10)
btn1['font'] = btn_font
btn1.grid(row=0, column=0, padx=(5,5), pady=(10,10))

btn2 = tk.Button(frame1,fg="orange", text="train",command=train, height=5, width=10)
btn2['font'] = btn_font
btn2.grid(row=0, column=1, padx=(5,5), pady=(10,10))

btn3 = tk.Button(frame1,fg="green", text="run",command=inf, height=5, width=10)
btn3['font'] = btn_font
btn3.grid(row=0, column=2, padx=(5,5), pady=(10,10))

btn4 = tk.Button(frame1,fg="green", text="text to anim",command=texttoanim, height=5, width=10)
btn4['font'] = btn_font
btn4.grid(row=6, columnspan=3, padx=(5,5), pady=(10,10))

btn5 = tk.Button(frame1,fg="green", text="voice to anim",command=voicetoanim, height=5, width=10)
btn5['font'] = btn_font
btn5.grid(row=8, columnspan=3, padx=(5,5), pady=(10,10))

frame1.pack()
window.mainloop()