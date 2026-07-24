import cv2
import numpy as np

img=cv2.imread('image/flipkart_grid.png')  # Replace 'path_to_image.jpg' with the actual path to your image file

"""
print(img.shape) # Print the dimensions of the image (height, width, channels)
print(type(img)) # Print the data type of the image array

cv2.imshow('Image', img) # Display the image in a window
cv2.waitKey(0) # Wait for a key press to close the window
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
cv2.imshow('Grayscale Image', img_gray) # Display the grayscale image
print(img_gray.shape) # Print the dimensions of the grayscale image
cv2.waitKey(0) # Wait for a key press to close the window

"""
# playing with the pixel values of the image

"""
img[:,:,1]=0  # Set the green channel to 0
cv2.imshow('Image with Green Channel Set to 0', img) # Display the modified image
cv2.waitKey(0) # Wait for a key press to close the window

# applying BGR to gray image
imgBlue=img[:,:,0]  # Extract the blue channel
imgGreen=img[:,:,1]  # Extract the green channel
imgRed=img[:,:,2]  # Extract the red channel

new_img=np.hstack((imgBlue,imgGreen,imgRed)) # Stack the channels back together
cv2.imshow('Stacked Channels', new_img) # Display the stacked channels
cv2.waitKey(0) # Wait for a key press to close the window

"""
# resizing the image

"""
img_resize=cv2.resize(img, (512, 512))  # Resize the image to 300x300 pixels
cv2.imshow('Resized Image', img_resize) # Display the resized image
cv2.waitKey(0) # Wait for a key press to close the window

"""

# to resize an image of half of its original size, we can use the following code:

"""

img_half=cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))  # Resize the image to half of its original size
cv2.imshow('Half Size Image', img_half) # Display the half-size image
cv2.waitKey(0) # Wait for a key press to close the window

"""

# flipping the image

"""
flipped_vertical=cv2.flip(img, 0)  # Flip the image vertically
flipped_horizontal=cv2.flip(img, 1)  # Flip the image horizontally

cv2.imshow('Flipped Vertical', flipped_vertical) # Display the vertically flipped image
cv2.imshow('Flipped Horizontal', flipped_horizontal) # Display the horizontally flipped image
cv2.waitKey(0) # Wait for a key press to close the window

"""

# Cropping the image

"""
img_crop=img[100:500,200:600] # Crop the image to a specific region (y1:y2, x1:x2)
cv2.imshow('Cropped Image', img_crop) # Display the cropped image
cv2.waitKey(0) # Wait for a key press to close the window

# save the cropped image
cv2.imwrite('image/cropped_image.png', img_crop)  # Save the cropped image to a file

"""

# drawing shapes on the image

img=np.zeros((512,512,3),dtype=np.uint8)  # Create a black image of size 512x512 pixels

# draw a rectangle on the image
cv2.rectangle(img,pt1=(100,100),pt2=(400,400),color=(255,0,0),thickness=-1)  # Draw a green rectangle
# draw a circle on the image
cv2.circle(img,center=(100,256),radius=50,color=(0,255,0),thickness=-1)  # Draw a red circle

# draw a line on the image

cv2.line(img,pt1=(0,0),pt2=(512,512),color=(0,0,255),thickness=3)  # Draw a blue line

# check the datatype of image
print(img.dtype)  # Print the data type of the image array
# draw text on the image
cv2.putText(img,org=(100,200),fontScale=3,color=(0,255,255),thickness=2,lineType=cv2.LINE_AA,text='Hello',fontFace=cv2.FONT_ITALIC)
cv2.imshow('Black Image', img) # Display the black image
cv2.waitKey(0) # Wait for a key press to close the window