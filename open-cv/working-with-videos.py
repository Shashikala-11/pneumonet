import cv2
import numpy as np

cap=cv2.VideoCapture(0) # Open the default camera (usually the webcam)

fourcc=cv2.VideoWriter_fourcc(*'XVID') # Define the codec for video writing
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480)) # Create a VideoWriter object to save the video
while True:
    ret,frame=cap.read() # Read a frame from the camera , the ret variable indicates whether the frame was read successfully, and frame contains the image data

    img_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale
    cv2.imshow('Video',frame) # Display the frame in a window named 'Video'
    out.write(frame) # Write the frame to the output video
    # cv2.imshow('Video',img_gray) # Display the grayscale frame in a window named 'Video'

    # BGR to LUV
    '''
    img_luv=cv2.cvtColor(frame,cv2.COLOR_BGR2LUV) # Convert the frame to LUV color space
    cv2.imshow('LUV',img_luv) # Display the LUV frame in a window named 'LUV'
'''
    # BGR to RGB

    '''
    img_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # Convert the frame to RGB color space
    cv2.imshow('RGB',img_rgb) # Display the RGB frame in a window named 'RGB'
    '''
    
    print(ret) # Print the value of ret to the console
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
