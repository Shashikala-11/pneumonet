import cv2
import numpy as np

img=np.zeros((512,512,3), np.uint8)  # Create a black image of size 512x512 with 3 color channels (BGR)

# Read an image
flag,ix,iy=False,-1,-1  # Initialize variables for mouse events
def draw(event,x,y,flags,params):
    global flag,ix,iy  # Declare the variables as global to modify them inside the function
    print("Event Triggered:", event) # Print the event type to the console
    if event==1:
        flag=True  # Set the flag to True when the left mouse button is pressed
        ix,iy=x,y  # Store the initial mouse position
        # cv2.circle(img,center=(x,y),radius=10,color=(0,255,0),thickness=-1)  # Draw a green circle at the mouse click position
    elif event==0:
        if flag==True:
            cv2.rectangle(img,pt1=(ix,iy),pt2=(x,y),color=(0,255,255),thickness=-1) # Draw a yellow rectangle from the initial mouse position to the current mouse position
    elif event==4:
        flag=False  # Set the flag to False when the left mouse button is released
        cv2.rectangle(img,pt1=(ix,iy),pt2=(x,y),color=(0,255,255),thickness=-1) # Draw a yellow rectangle from the initial mouse position to the current mouse position

cv2.namedWindow(winname='Image')  # Create a window named 'Image'
cv2.setMouseCallback('Image',draw)  # Set the mouse callback function for the 'Image' window


while True:
    cv2.imshow('Image', img)  # Display the image in a window named 'Image'
    # Wait for a key press for 1 millisecond

    if cv2.waitKey(1) & 0xFF == ord('q'):  # If the 'q' key is pressed
        break  # Exit the loop and close the window

cv2.destroyAllWindows()  # Close all OpenCV windows