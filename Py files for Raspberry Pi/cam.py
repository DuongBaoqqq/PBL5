# Credit: Adrian Rosebrock
# https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
 
# import the necessary packages
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import numpy as np
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def start_camera(on_capture_image , on_exit):
    # Initialize the camera
    camera = PiCamera()
    
    # Set the camera resolution
    camera.resolution = (640, 480)
    
    # Set the number of frames per second
    camera.framerate = 32
    
    # Generates a 3D RGB array and stores it in rawCapture
    raw_capture = PiRGBArray(camera, size=(640, 480))
    
    # Wait a certain number of seconds to allow the camera time to warmup
    time.sleep(0.1)
    
    # Capture frames continuously from the camera
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        # Grab the raw NumPy array representing the image
        image = frame.array
        
        # Display the frame using OpenCV
        cv2.imshow("Frame", image)
        
        # Wait for keyPress for 1 millisecond
        key = cv2.waitKey(1) & 0xFF
        
        # Clear the stream in preparation for the next frame
        raw_capture.truncate(0)
        
        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        button_state = GPIO.input(12)
        if button_state = False:
            print("Nut nhan da duoc nhan")
            time.sleep(0.2)
            if len(image) >0 and len(image[0])>0:
                img_encode = cv2.imencode('.jpg', image)[1]

                data_encode = np.array(img_encode)

                byte_encode = data_encode.tobytes()

                # Send captured image to server
                on_capture_image(byte_encode)
        
    cv2.destroyAllWindows()
    on_exit()