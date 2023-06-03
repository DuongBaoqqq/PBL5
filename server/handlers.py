import cv2
import mediapipe as mp
import numpy as np



def process_image(image):
    hands = mp.solutions.hands.Hands()
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Đưa hình ảnh vào model để nhận dạng các điểm trên bàn tay
    results = hands.process(image_rgb)
    height, width, channels = image.shape
    
    # Kiểm tra xem có các điểm được nhận dạng hay không
    if results.multi_hand_landmarks:
        # Vẽ các điểm trên bàn tay lên bức ảnh
        for hand_landmarks in results.multi_hand_landmarks:
            landmarksX = []
            landmarksY = []
            for landmark in hand_landmarks.landmark:
                # Tọa độ pixel của điểm nhận dạng trên bàn tay
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                # Vẽ một đường tròn tại điểm nhận dạng
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                landmarksX.append(landmark.x)
                landmarksY.append(landmark.y)
            min_X = int(min(landmarksX)*width)
            max_X = int(max(landmarksX)*width)
            min_Y = int(min(landmarksY)*height)
            max_Y = int(max(landmarksY)*height)
            # Vẽ các đường nối giữa các điểm nhận dạng
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for connection in connections:
                start_index = connection[0]
                end_index = connection[1]
                start_point = tuple(np.multiply([hand_landmarks.landmark[start_index].x, hand_landmarks.landmark[start_index].y], [image.shape[1], image.shape[0]]).astype(int))
                end_point = tuple(np.multiply([hand_landmarks.landmark[end_index].x, hand_landmarks.landmark[end_index].y], [image.shape[1], image.shape[0]]).astype(int))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        # Hiển thị bức ảnh với các điểm đã được vẽ
        imgCrop = image[min_Y-40:max_Y+40,min_X-40:max_X+40]
        
        imgCrop = cv2.flip(imgCrop , 1)
        
        print(imgCrop)
        
        if imgCrop is None or np.array_equal(imgCrop, None):
            print("IMG CROP is none")
            return None
        
        hsv = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
        
        # Đặt giới hạn màu xanh trong không gian màu HSV
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        # Tạo mặt nạ (mask) cho các màu xanh
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Áp dụng mặt nạ lên bức ảnh gốc
        result = cv2.bitwise_and(imgCrop, imgCrop, mask=mask)
        
        return result
    
    return None
    

if __name__ == '__main__':
    a = process_image(2)
    
    print(a)