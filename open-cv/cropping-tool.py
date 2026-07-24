import cv2

import numpy as np

img=cv2.imread('image/flipkart_grid.png')

flag=False
ix,iy=-1,-1

def crop(event,x,y,flags,params):
    global flag,ix,iy
    if event==1:
        flag=True
        ix,iy=x,y
    
      
    elif event==4:
        fx,fy=x,y
        flag=False
        cv2.rectangle(img,pt1=(ix,iy),pt2=(x,y),thickness=1,color=(0,0,0))   

        # crop the image
        cropped_image=img[iy:fy,ix:fx]   

        # save the cropped image
        cv2.imwrite('image/cropped_image.png',cropped_image)
        cv2.imshow('cropped image',cropped_image)  
        cv2.waitKey(0)

cv2.namedWindow(winname='window')
cv2.setMouseCallback('window',crop)
while True:
    cv2.imshow('window',img)

    if cv2.waitKey(1) & 0xFF==ord('x'):
        break

cv2.destroyAllWindows()    