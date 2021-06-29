import dlib
import cv2
from keras.models import load_model
import numpy as np


detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
model = load_model('cnn_mask.h5')
try:
    
    labels_dict = {0:'NO MASK',1:'MASK'}
    color_dict  = {0:(0,0,255),1:(0,255,0)}
    while True:        
        _,frame = cap.read()
        
        gray =  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        
        for face in faces:
            x1 = face.left() 
            y1 = face.top() 
            x2 = face.right() 
            y2 = face.bottom() 
            face_img = gray[y1:y2,x1:x2]
            resized = cv2.resize(face_img,(100,100))
            normalized = resized/255
            reshaped = np.reshape(normalized,(1,100,100,1))
            result = model.predict(reshaped)

            label = round(result[0][0])
            
            cv2.rectangle(frame,(x1,y1),(x2,y2),color_dict[label],2)
            cv2.putText(frame,labels_dict[label],(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
        cv2.imshow("Frame",frame)
        
        key = cv2.waitKey(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.release()

except Exception as e:
    print(e)
    cv2.destroyAllWindows()
    cap.release()
