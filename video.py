import cv2


#trained dataset
trainedDataset= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture('videos/39890-423345734.mp4')
#here we get two output so we use two variable
while True:
    success,frame=video.read()
    if success==True:
        gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = trainedDataset.detectMultiScale(gray_image, 1.5, 5)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('video',frame)
        cv2.waitKey(1)
    else:
        print("video is completed")
        break
