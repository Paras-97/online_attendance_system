# Importing required libraries
from flask import Flask, render_template, request
from flask.wrappers import Request, Response
import numpy as np
import cv2
from datetime import datetime
#import tensorflow as tf
from keras.models import load_model
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
model = load_model("saved_model/model.h5")

def create_app():

    def check_offline(online, results):
        if online == []:
            return list(results.values())
        ret = []
        for item in results:
            if results[item] not in online:
                ret.append(results[item])
        return ret

    
    def get_timestamp():
        dateTimeObj = datetime.now()
        day = str(dateTimeObj.year)+ '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.day)
        time = str(dateTimeObj.hour) + ':' + str(dateTimeObj.minute) 
        return (day + ' '+ time)  

    def gen_frames():  
        results = {0:'Meet', 1:'Paras', 2:'Sahil'}
        online = {}
        offline = {}
        max_val_ind = 0
        fa_img = np.zeros((1,224,224,3),dtype=np.float32)
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
                # Find faces in image using classifier
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
                # Draw rectangle around the faces
                on = []
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255,255), 2)  
                    face_img = frame[y:y + h, x:x + w]
                    face_img =  cv2.resize(face_img, (224,224), interpolation = cv2.INTER_AREA)

                    fa_img[0,:,:,:] = face_img/223

                    y_pred = model.predict(fa_img)

                    max_val = max(y_pred[0])
                    max_val_ind = int(list(y_pred[0]).index(max_val))
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.putText(frame,results[max_val_ind],(x,y-2), font, 0.8, (0,255,255)) 
                    on.append(results[max_val_ind])
                #print(on)    
                    off = check_offline(on, results)
                    timestamp = get_timestamp()
                    online = {timestamp:on}
                    offline = {timestamp:off}
                    #print("Online: ", online)
                    #print("offline: ", offline)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    app=Flask(__name__)
    @app.route('/')
    def home():
        return render_template("base.html")
    
    @app.route('/video_feed')
    def video_feed():
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
       

    return app