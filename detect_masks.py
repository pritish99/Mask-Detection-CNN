from keras.models import load_model
import cv2
import numpy as np


try:
    # Loading CNN Model
    model =load_model('my_model.h5')
    faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)  #starts the webcam
    labels_dict = {0:'NO MASK',1:'MASK'}
    color_dict  = {0:(0,0,255),1:(0,255,0)}
    while(True):
        ret,frame = video_capture.read()
        #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(frame,2,5)
        #print(faces)
        for x,y,w,h in faces:
            face_img = frame[y:y+w,x:x+h]
            resized = cv2.resize(face_img,(64,64))
            normalized = resized/255
            reshaped = np.reshape(normalized,(1,64,64,3))
            result = model.predict(reshaped)
            print(result)
            result=np.round(result, 1)
            final=result[0].tolist()
            final=final.index(max(final))
            print(final)
            if round(result[0][0],1) <=0.5:
                label = 0
            else:
                label = 1
            
            cv2.rectangle(frame,(x,y),(x+210,y+210),color_dict[label],2)
            cv2.putText(frame,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
        cv2.imshow('Video',frame)
        key=cv2.waitKey(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    video_capture.release()
    
except Exception as e:
    print(e)
    cv2.destroyAllWindows()
    video_capture.release()
    
