# USAGE
# python pi_facial_landmarks.py

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from playsound import playsound
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# load the input image and convert it to grayscale
image = cv2.imread("example.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("[INFO] starting video stream thread...")
vs = VideoStream(0).start()
time.sleep(1.0)
frameCount=0
inicio_de_tiempo=None
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		contador=0
		aperturaOjoDerecho=0
		aperturaOjoIzquierdo=0
		anchoOjoIzquierdo=0
		anchoOjoDerecho=0
		aperturaBoca=0
		anchoBoca=0
		distanciaNariz=0
		alarma = False

		posicionesX=[]
		posicionesY=[]
		for (x, y) in shape:

			posicionesX.append(x)
			posicionesY.append(y)
			#CONTORNO FACIAL
			if(contador>=0 and contador<17):
				cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
				cv2.putText(frame, str(contador), (x+3, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
			#CEJAS
			if(contador>17 and contador<27):
				cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
				cv2.putText(frame, str(contador), (x+3, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			#NARIZ
			if(contador>26 and contador<36):
				cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
				cv2.putText(frame, str(contador), (x+3, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,255 ), 2)
			#OJOS
			#if(contador>35 and contador<48):
			if(contador==36 or contador==39 or contador==42 or contador==45):
				cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
				cv2.putText(frame, str(contador), (x+3, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
				print('Coordenadas x='+str(x)+' y= '+ str(y))
			#BOCA
			if(contador>47 and contador<68):
				cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
				cv2.putText(frame, str(contador), (x+3, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			contador=contador+1
		#print(str(len(posicionesOjoIzquierdoX)))
		print('Parpado superior izquierdo X='+str(posicionesX[37])+' Y='+str(posicionesY[37]))
		print('Parpado inferior izquierdo X='+str(posicionesX[41])+' Y='+str(posicionesY[41]))
		print('Parpado superior derecho X='+str(posicionesX[44])+' Y='+str(posicionesY[44]))
		print('Parpado inferior derecho X='+str(posicionesX[46])+' Y='+str(posicionesY[46]))
		print('Boca labio superior X='+str(posicionesX[62])+' Y='+str(posicionesY[62]))
		print('Boca labio inferior X='+str(posicionesX[66])+' Y='+str(posicionesY[66]))
		print('Distancia labios '+ str(posicionesY[66]-posicionesY[62]))
		print('Distancia ojo izquierdo '+ str(posicionesY[41]-posicionesY[37]))
		print('Distancia ojo derecho '+ str(posicionesY[46]-posicionesY[44]))
		print('Tamaño del arreglo x '+ str(len(posicionesX)))
		#<Distancia entre el parpado superior e inferior izquiero
		aperturaOjoIzquierdo = math.sqrt((posicionesY[37]-posicionesY[41])**2)
		print('Distancia ojo izquierdo '+ str(aperturaOjoIzquierdo))
		#Distancia entre el parpado superior e inferior derecho
		aperturaOjoDerecho = math.sqrt((posicionesY[44]-posicionesY[46])**2)
		print('Distancia ojo derecho '+ str(aperturaOjoDerecho))
		#<Distancia entre el ojo derecho
		anchoOjoDerecho = math.sqrt((posicionesX[42]-posicionesX[45])**2)
		print('Ancho ojo derecho '+ str(anchoOjoDerecho))
		#Distancia entre el ojo izquierdo 
		anchoOjoIzquierdo = math.sqrt((posicionesX[36]-posicionesX[39])**2)
		print('ancho ojo izquierdo '+ str(anchoOjoIzquierdo))
		
		#Distancia entre el labio superior e inferior 
		aperturaBoca = math.sqrt((posicionesY[62]-posicionesY[66])**2)
		print('Distancia boca '+ str(aperturaBoca))
		#Distancia nariz
		distanciaNariz = math.sqrt((posicionesY[27]-posicionesY[30])**2)
		print('Distancia nariz '+ str(distanciaNariz))
		#Ancho de boca
		anchoBoca = math.sqrt((posicionesX[67]-posicionesX[54])**2)
		print('Ancho de boca '+ str(anchoBoca))
		#Bostezo
		if(aperturaBoca>=(anchoBoca*0.8)):
			print('Bostezo')
		#Alerta ojos
		if(aperturaOjoDerecho<=(anchoOjoDerecho*0.2)):
			print('Alerta ojo derecho')
			print('Conteo de cuadros '+str(frameCount))
			if inicio_de_tiempo is None:
				inicio_de_tiempo = time.time()
			if not alarma:
				alarma = True

				# check to see if an alarm file was supplied,
				# and if so, start a thread to have the alarm
				# sound played in the background
				tiempo_final = time.time() 
				tiempo_transcurrido = tiempo_final - inicio_de_tiempo
				print('Tiempo dormido '+str(tiempo_transcurrido))
				if (tiempo_transcurrido>=1):
					t = Thread(target=reproducirAlarma,args=())
					t.deamon = True
					t.start()
		else:
			ALARM_ON = False
			inicio_de_tiempo=None

		#Alerta ojos
		if(aperturaOjoIzquierdo<=(anchoOjoIzquierdo*0.2)):
			print('Alerta ojo izquierdo')
		
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	#frameCount+1
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	def reproducirAlarma():
	# play an alarm sound
		playsound('alarm.wav')
		