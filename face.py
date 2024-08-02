import cv2


#trained dataset
trainedDataset= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#in opencv all are work in greyscale


#how to read a image
img=cv2.imread('images/group.webp')


#in opencv all are work in grayscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=trainedDataset.detectMultiScale(gray,1.1,5)#Scale Factor: This parameter specifies how much the image size is reduced at each image scale. It is used to create a scale pyramid. A smaller scale factor means that the algorithm will be more thorough, but it will take longer, while a larger scale factor will speed up the process but be less accurate. Typically, a value of 1.1 to 1.5 is used.
#minneightbour is used to reduce the rectangle near to the current rectangle
print(faces)

for x, y, w, h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)



#how to show the image
cv2.imshow('Ram',img)
#cv2.imshow('gray',gray)




#to show continuosly
cv2.waitKey()