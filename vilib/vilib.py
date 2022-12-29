#!/usr/bin/env python3
import os
import time
import datetime

print('Launching ...')
import cv2
import numpy as np
from threading import Thread
from time import sleep
import tflite_runtime.interpreter as tflite
from pyzbar import pyzbar
from flask imporwt Flask, render_template, Response

kernel_5 = np.ones((5,5),np.uint8)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

def human_face_detect(img):
    resize_img = cv2.resize(img, (320,240), interpolation=cv2.INTER_LINEAR)         # In order to reduce the amount of calculation, resize the image to 320 x 240 size
    gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)    # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)    # Detect faces on grayscale images
    face_num = len(faces)   # Number of detected faces
    if face_num  > 0:
        for (x,y,w,h) in faces:
            
            x = x*2   # Because the image is reduced to one-half of the original size, the x, y, w, and h must be multiplied by 2.
            y = y*2
            w = w*2
            h = h*2
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # Draw a rectangle on the face
    
    return img

def color_detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Red color rangle  169, 100, 100 , 189, 255, 255
    lower_range = np.array([169,100,100])
    upper_range = np.array([189,255,255])

    mask = cv2.inRange(hsv, lower_range, upper_range)
    morphologyEx_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5,iterations=1) 
    _tuple = cv2.findContours(morphologyEx_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)      
    # compatible with opencv3.x and openc4.x
    if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
    else:
        contours, hierarchy = _tuple
    color_area_num = len(contours) # Count the number of contours

    if color_area_num > 0: 
        for i in contours:    # Traverse all contours
            x,y,w,h = cv2.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object

            # Draw a rectangle on the image (picture, upper left corner coordinate, lower right corner coordinate, color, line width)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
            cv2.putText(img,'red',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)# Add character description

    return img

class Vilib(object): 
    def __init__(self, src=0):
        self.color_detect = False
        self.human_face_detect = False
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            sleep(1/30)

    
    def show_frame(self):
        # Display frames in main program
        while True:
            if self.color_detect == True:
                self.frame = color_detect(self.frame)
            elif self.human_face_detect == True:
                self.frame = human_face_detect(self.frame)
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            cv2.waitKey(int(1/30*1000))