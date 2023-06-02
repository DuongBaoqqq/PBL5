import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

words = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
loaded_model = load_model('./cnn_model.h5')

def NhanDien(image_new):
    image_rgb = cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)

    # Đưa hình ảnh vào model để nhận dạng các điểm trên bàn tay
    results = hands.process(image_rgb)

    # Kiểm tra xem có các điểm được nhận dạng hay không
    if results.multi_hand_landmarks:
        # Vẽ các điểm trên bàn tay lên bức ảnh
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                    # Tọa độ pixel của điểm nhận dạng trên bàn tay
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    # Vẽ một đường tròn tại điểm nhận dạng
                    cv2.circle(image_new, (x, y), 5, (0, 255, 0), -1)

            # Vẽ các đường nối giữa các điểm nhận dạng
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for connection in connections:
                start_index = connection[0]
                end_index = connection[1]
                start_point = tuple(np.multiply([hand_landmarks.landmark[start_index].x, hand_landmarks.landmark[start_index].y], [image.shape[1], image.shape[0]]).astype(int))
                end_point = tuple(np.multiply([hand_landmarks.landmark[end_index].x, hand_landmarks.landmark[end_index].y], [image.shape[1], image.shape[0]]).astype(int))
                cv2.line(image_new, start_point, end_point, (0, 255, 0), 2)
    hsv = cv2.cvtColor(image_new, cv2.COLOR_BGR2HSV)
    # Đặt giới hạn màu xanh trong không gian màu HSV
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    # Tạo mặt nạ (mask) cho các màu xanh
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Áp dụng mặt nạ lên bức ảnh gốc
    result = cv2.bitwise_and(image_new, image_new, mask=mask)
    cv2.imshow('D', result)
    
    result = cv2.resize(result, (40, 40))
    
    
    image_new=img_to_array(result) 
    image_new=image_new/255.0
    prediction_image=np.array(image_new)
    prediction_image= np.expand_dims(image_new, axis=0)

    prediction=loaded_model.predict(prediction_image)
    value=np.argmax(prediction)
    move_name=words[value]
    
    return move_name

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
# Số lượng bàn tay đã được lưu
hand_count = 0

while True:
    # Đọc frame từ video
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    
    # Chuyển đổi frame sang định dạng RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Nhận diện bàn tay trên frame
    results = hands.process(frame_rgb)
    
    # Vẽ landmark và bounding box xung quanh bàn tay (nếu có)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Tạo bounding box xung quanh bàn tay
            x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
            
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Chuyển đổi tọa độ bounding box thành pixel (nếu có tọa độ hợp lệ)
            if x_min != float('inf') and y_min != float('inf') and x_max != float('-inf') and y_max != float('-inf'):
                x_min = int(x_min * frame.shape[1]) - 50  # Giảm 10 pixel cho cạnh trái
                x_max = int(x_max * frame.shape[1]) + 50  # Tăng 10 pixel cho cạnh phải
                y_min = int(y_min * frame.shape[0]) - 50  # Giảm 10 pixel cho cạnh trên
                y_max = int(y_max * frame.shape[0]) + 50  # Tăng 10 pixel cho cạnh dưới
                
                # Đảm bảo giới hạn của bounding box không vượt quá kích thước khung hình
                x_min = max(0, x_min)
                x_max = min(frame.shape[1] - 1, x_max)
                y_min = max(0, y_min)
                y_max = min(frame.shape[0] - 1, y_max)
                
                # Vẽ bounding box trên khung hình gốc
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                
                # Cắt và lưu ảnh chứa bàn tay
                image = frame[y_min:y_max, x_min:x_max]
                
                cv2.imwrite("Data/Test_TA/M/" + str(hand_count) + ".jpg" , image)
                hand_count += 1
    print(hand_count)                     
    if hand_count > 210:
        break                     
    # Hiển thị kết quả
    cv2.imshow('Hand Detection', frame)
    
    # Thoát vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhận diện bàn tay và đóng video stream
hands.close()
cap.release()
cv2.destroyAllWindows()