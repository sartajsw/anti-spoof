import cv2
import numpy as np
import sqlite3
from PIL import Image
from matplotlib import pyplot as plt
import math

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
sampleNum=0
def insertorUpdate(ID,NAME):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(ID)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name="+str(NAME)+" where id="+str(ID)
    else:
        cmd="Insert INTO People(ID,NAME) Values("+str(ID)+","+str(NAME)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

     
    
ID=raw_input('Enter User id')
NAME=raw_input('Enter your name')
insertorUpdate(ID,NAME)
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataSet/User."+ID+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100);
    cv2.imshow( "Face",img);
    cv2.waitKey(1);
    if(sampleNum>200):
        break;
cam.release()

cv2.destroyAllWindows()
