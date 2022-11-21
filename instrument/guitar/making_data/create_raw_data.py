import cv2
import os
import mediapipe as mp
import tensorflow as tf


# 미디어파이프를 활용한 영상 처리 기법
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

image_x, image_y = 200, 200

# 데이터 셋을 저장하는 폴더 생성
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main(c_id):
    # 데이터 셋은 코드 당 1200장
    total_pics = 1200
    hands = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
    # 웹캠 연결
    cap = cv2.VideoCapture(0)
    create_folder("chords_" + str(c_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    # 웹 캠이 연결 되면 실행
    while cap.isOpened():
        # 웹 캠에서 전달받은 정보를 ret, img로 저장
        ret, image = cap.read()
        # 사용자 화면에 맞게 좌우 반전
        image = cv2.flip(image, 1)
        image_orig = cv2.flip(image, 1)
        # 이미지의 색상 정보를 BGR에서 RGB로 바꿈(opencv의 색상 순서는 BGR임)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # 미디어파이프를 적용한 이미지 저장
        results_hand = hands.process(image)
        image.flags.writeable = True
        # 이미지의 색상 정보를 RGB에서 BGR로 변경
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 손 랜드마크 그리기
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)
        # 미디어파이프를 적용한 이미지를 res에 저장
        res = cv2.bitwise_and(image, cv2.bitwise_not(image_orig))

        # 사진의 용량의 줄이기 위해 res이미지를 모두 grayscale로 변환환
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)


        # 이미지의 픽셀 값이 25보다 크면 255로 변환, 아니면 0으로 변환
        ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)


        contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            # if cv2.contourArea(contour) > 10000 and frames > 50:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            pic_no += 1
            cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            save_img = gray[y1:y1 + h1, x1:x1 + w1]
            # 이미지 저장
            save_img = cv2.resize(save_img, (image_x, image_y))
            # 이미지 내에 캡처 중이라는 text삽입
            cv2.putText(image, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
            # 이미지 저장
            cv2.imwrite("chords_" + str(c_id) + "_" + str(pic_no) + ".jpg", save_img)

            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 이미지에 pic_no 표시
            cv2.putText(image, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))


            keypress = cv2.waitKey(1)
            if keypress == ord('c'):
                if flag_start_capturing == False:
                    flag_start_capturing = True
                else:
                    flag_start_capturing = False
                    frames = 0
            if flag_start_capturing == True:
                frames += 1
            if pic_no == total_pics:
                break
        # 사용자의 입력 이미지
        cv2.imshow("Capturing gesture", image)
        # 미디어 파이프 적용 이미지
        cv2.imshow("Res", res)


if __name__ == '__main__':
    if tf.test.is_built_with_cuda():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('using GPU')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('using CPU instead of GPU')

    c_id = input('Enter code: ')
    main(c_id)
