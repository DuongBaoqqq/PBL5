import cv2
import numpy as np

def start_camera(on_capture_image , on_exit) :
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        height, width, channels = image.shape
        
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        cv2.imshow('Captured Image', cv2.flip(image, 1))
        
        key = cv2.waitKey(1)
        if  key == ord('q'):
            break
        
        if key == ord('c'):
            if len(image) >0 and len(image[0])>0:
                img_encode = cv2.imencode('.jpg', image)[1]

                data_encode = np.array(img_encode)

                byte_encode = data_encode.tobytes()

                # Send captured image to server
                on_capture_image(byte_encode)

        
       
                            
    cv2.destroyAllWindows()
    cap.release()
    on_exit()

# changing comment