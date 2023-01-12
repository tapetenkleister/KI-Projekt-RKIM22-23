from __future__ import annotations
import cv2
import numpy as np
# Show the processing steps of the image
def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.resizeWindow(winname, 300,500)
    cv2.imshow(winname,img)

def detect_red_seal(image : np.ndarray,x_y_w_h, top_part , debug : bool):
# Converts the BGR color space of the image to the HSV color space

    x,y,w,h = x_y_w_h 
    if top_part != 0:
        cropped = image[y:y+(int(h*top_part)), x:x+w]
       
    else:     
        cropped = image[y:y+h, x:x+w]
    
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            

   # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([5,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # Erode the image
    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_5 = np.ones((5, 5), np.uint8)
    try:
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_3, iterations=2) 
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_5, iterations=2)
        contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # Get the biggest contour inside the image
        biggest_contours = max(contours, key=cv2.contourArea)
        # Create a black canvas and draw all found contours onto it
        black_canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        contour_pic = cv2.drawContours(black_canvas.copy(), biggest_contours, -1, (0, 255, 75), 4)
        # Draw the bounding box
        x_y_w_h = cv2.boundingRect(biggest_contours)
        x,y,w,h = x_y_w_h  
        bounding_boxes = cv2.rectangle(contour_pic, (x, y), (x + w, y + h), (255, 255, 255), 1)
        seal_position = y+(h/2)
        seal_position = seal_position/(cropped.shape[0])
        if debug:
                #cv2.imshow('input', image)
                showInMovedWindow('cropped', cropped,0,10)
                showInMovedWindow('mask', mask,305,10)
                showInMovedWindow('opening', opening,610,10)
                showInMovedWindow('closing', closing,920,10)
                showInMovedWindow('keypoints',bounding_boxes,1230,10)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        #seal position 2 means no seal detected
        seal_position = 2  

    
        

    
    if debug:
        #cv2.imshow('input', image)
        showInMovedWindow('cropped', cropped,0,10)
        showInMovedWindow('mask', mask,305,10)
        showInMovedWindow('opening', opening,610,10)
        showInMovedWindow('closing', closing,920,10)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return seal_position
