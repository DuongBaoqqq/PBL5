import eventlet
import socketio
from tensorflow.keras.models import load_model
import json
import base64
import numpy as np
import cv2
from handlers import process_image
# from image-pro
CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
N = [i for i in range(len(CHARACTERS))]
CHAR_DICT = dict(zip(N,CHARACTERS))

sio = socketio.Server()

MODEL = load_model("cnn_model.h5")

app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})


@sio.event
def connect(sid, environ):
    print('connect ', sid)



@sio.on("data")
def on_data(sid , receive_data) :
    
    try:
        image_array = np.frombuffer(receive_data, np.uint8)
    
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        processed_image = process_image(img)

        if processed_image is None or np.array_equal(processed_image, None):
            result = "Nothing"

            sio.emit("result" , result , room=sid )
            

            print(f"ket qua la : {result}")
        else :
            # Display the image
            cv2.imwrite("Image.jpg" ,processed_image )

            resized_img = cv2.resize(processed_image, (40, 40)) / 255.0

            prediction_image= np.expand_dims(resized_img, axis=0)

            prediction = MODEL.predict(prediction_image)

            value = np.argmax(prediction)

            result = CHAR_DICT[value]

            sio.emit("result" , result , room=sid )

            print(f"ket qua la : {result}")
    except Exception as e:
        print(f"Error : {e}")
        
    pass
    

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)