import cv2
import numpy as np
cam = cv2.VideoCapture(0)
# cv2.namedWindow('Image test 1', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Image test 2', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Text test', cv2.WINDOW_NORMAL)

while True:
    ret, image = cam.read()
    cv2.imshow('Image test', image)
    image = cv2.resize(image, (960, 720))
    k = cv2.waitKey(1)
    if k == ord('q'):        
        cv2.imwrite('testimage.jpg', image)
        # cv2.imshow("Image", image)
        
        text_image = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)        
        # text_image: ảnh trắng để hiển thị chữ.
        # 'Hello': chuỗi ký tự cần hiển thị.
        # (200, 256): tọa độ của chữ trên màn hình.
        # cv2.FONT_HERSHEY_SIMPLEX: kiểu phông chữ.
        # 3: kích thước phông chữ.
        # (0, 0, 0): màu của chữ.
        # thickness=5: độ dày của chữ.        
        # cv2.putText(text_image, 'Hello', (200, 256), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), thickness=5)
        text = 'AB!'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        color = (0, 0, 255)
        thickness = 10
        cv2.putText(image, text, (0, 150), font, font_scale, color, thickness)
        cv2.imshow('Image test 1', image)
        cv2.resizeWindow('Image test 1', 1920, 1080)
        cv2.waitKey(1)
    elif k == ord('c'):
        cam.release()
        cv2.destroyAllWindows()
        cam = cv2.VideoCapture(0)

cam.release()
cv2.destroyAllWindows()