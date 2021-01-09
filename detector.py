import cv2
import numpy as np
import sqlite3
from PIL import Image
from matplotlib import pyplot as plt
import math
import pandas

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")
"""
def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent
ret,img=cam.read();
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
colorIm=np.array(img)
greyIm=np.array(gray)
"""
"""
def luminance(im):
   #im = Image.open(im_file)
   #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
   #fin = np.zeros((800, 800))
   #fin=cv2.normalize(im,  fin, 0, 255, cv2.NORM_MINMAX)
   #cv2.imshow("norm gs", fin)
   #cv2.imshow("normal face", im)
   im=Image.fromarray(im)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   return (0.299*(r) + 0.587*(g) + 0.114*(b))
print luminance(colorIm)
"""
"""
N=5
S=greyIm.shape
E=np.array(greyIm)
for row in range(S[0]):
        for col in range(S[1]):
                Lx=np.max([0,col-N])
                Ux=np.min([S[1],col+N])
                Ly=np.max([0,row-N])
                Uy=np.min([S[0],row+N])
                region=greyIm[Ly:Uy,Lx:Ux].flatten()
                E[row,col]=entropy(region)

plt.subplot(1,3,1)
plt.imshow(colorIm)

plt.subplot(1,3,2)
plt.imshow(greyIm, cmap=plt.cm.gray)
plt.subplot(1,3,3)

plt.imshow(E,cmap=plt.cm.jet)

plt.xlabel('Entropy in 10x10 neighbourhood')
plt.colorbar()

plt.show()
"""

"""
def image_entropy(img):
    calculate the entropy of an image
    histogram = img.histogram()
    histogram_length = sum(histogram)
 
    samples_probability = [float(h) / histogram_length for h in histogram]
 
    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
a=image_entropy(img)
print a
"""
"""
def ent2(img):
    img=Image.fromarray(img)
    histogram = img.histogram()
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram]
    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
"""
def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

font = cv2.FONT_HERSHEY_SIMPLEX;
global cropped

while(True):
        ret,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                #cropped = img[y:y+h, x:x+w]
                #cv2.imshow("cropped", cropped)
                id,conf=rec.predict(gray[y:y+h,x:x+w])
                profile=getProfile(id)
                if conf<60:
                        if(profile!=None):
                                cv2.putText(img,str(profile[0]),(x,y+h+30), font, 1,(255,255,255),2);
                                cv2.putText(img,str(profile[1]),(x,y+h+60), font, 1,(255,255,255),2);
                else:
                        cv2.putText(img,"Unknown Person",(x,y+h+30),font,1,(255,255,255),2);
        cv2.imshow( "Face",img);
    
    
   
        if(cv2.waitKey(1)==(ord('q'))):
                break;
  
    
cam.release()

cv2.destroyAllWindows()
