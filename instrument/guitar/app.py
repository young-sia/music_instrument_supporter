from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import glob
import os.path
from music21 import *
import shutil
import os
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd


# Flask app 생성
app = Flask(__name__)


# 업로드 HTML 렌더링
@app.route('/upload')
def upload_page():
    return render_template('upload_new.html')


# 파일 업로드 처리
@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('./uploads/' + secure_filename(f.filename))
        us = environment.UserSettings()

        # 원하면 설정을 바꾸세요
        us['lilypondPath'] = 'C:/Program Files (x86)/LilyPond/usr/bin/lilypond.exe'
        us['musescoreDirectPNGPath'] = 'C:/Program Files (x86)/MuseScore 2/bin/MuseScore.exe'
        us['musicxmlPath'] = 'C:/Program Files (x86)/MuseScore 2/bin/MuseScore.exe'

        files = glob.glob("uploads/*.mid")
        for x in files:
            if not os.path.isdir(x):
                filename = os.path.splitext(x)
                try:
                    original_score = converter.parse(x).chordify()
                    conv = converter.subConverters.ConverterLilypond()
                    conv.write(original_score, fmt='lilypond', fp='score', subformats=['png'])
                except:
                    pass

        source = './score.png'
        destination = './uploads/score.png'
        shutil.move(source, destination)

        file_list = os.listdir("./uploads")

        return render_template('down_list.html', filedown_list=file_list)


# 추출된 chord 정보를 mediapipe에 전달 및 실행
@app.route('/csvplay', methods=['GET', 'POST'])
def csvplay():
    if request.method == 'POST':
        f = request.files['file']
        f.save('./' + secure_filename(f.filename))
        play_name = secure_filename(f.filename)

        a = str(play_name)
        csv_file = f'{a}'
        url_csv = f'./{csv_file}'
        # b= guitar_learner_show_image.main(csv_file)

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    model = load_model('guitar_learner.h5')

    chord_dict = {0: 'A#maj', 1: 'A#min', 2: 'Amaj', 3: 'Amin', 4: 'B#maj', 5: 'Bmaj', 6: 'Bmin', 7: 'C#maj',
                  8: 'C#min', 9: 'Cmaj', 10: 'Cmin', 11: 'D#maj', 12: 'D#min', 13: 'Dmaj', 14: 'Dmin', 15: 'Emaj',
                  16: 'Emin', 17: 'F#maj', 18: 'F#min', 19: 'Fmaj', 20: 'Fmin', 21: 'G#maj', 22: 'G#min',
                  23: 'Gmaj', 24: 'Gmin'}

    def rescale_frame(frame, percent=75):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # level 조정 필요
    def levels(level):
        if level == 1:
            return 3
        elif level == 2:
            return 2
        elif level == 3:
            return 1

    image_x, image_y = 200, 200

    cap = cv2.VideoCapture(0)

    def sub():
        hands = mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7)
        hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
        hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
        pic_no = 0
        flag_start_capturing = False
        frames = 0

        level = 3
        fps_speed = levels(level)
        fps = cap.get(cv2.CAP_PROP_FPS) * fps_speed
        print(fps)

        get_data = pd.read_csv(url_csv, na_values='NA', encoding='utf8')

        get_data = get_data.replace('C:maj', 'Cmaj')
        get_data = get_data.replace('C#:maj', 'C#maj')
        get_data = get_data.replace('D:maj', 'Dmaj')
        get_data = get_data.replace('D#:maj', 'D#maj')
        get_data = get_data.replace('E:maj', 'Emaj')
        get_data = get_data.replace('F:maj', 'Fmaj')
        get_data = get_data.replace('F#:maj', 'F#maj')
        get_data = get_data.replace('G:maj', 'Gmaj')
        get_data = get_data.replace('G#:maj', 'G#maj')
        get_data = get_data.replace('A:maj', 'Amaj')
        get_data = get_data.replace('A#:maj', 'A#maj')
        get_data = get_data.replace('B:maj', 'Bmaj')
        get_data = get_data.replace('C:min', 'Cmin')
        get_data = get_data.replace('C#:min', 'C#min')
        get_data = get_data.replace('D:min', 'Dmin')
        get_data = get_data.replace('D#:min', 'D#min')
        get_data = get_data.replace('E:min', 'Emin')
        get_data = get_data.replace('F:min', 'Fmin')
        get_data = get_data.replace('F#:min', 'F#min')
        get_data = get_data.replace('G:min', 'Gmin')
        get_data = get_data.replace('G#:min', 'G#min')
        get_data = get_data.replace('A:min', 'Amin')
        get_data = get_data.replace('A#:min', 'A#min')
        get_data = get_data.replace('B:min', 'Bmin')

        first_count = get_data['start'][0]
        get_data['time'] = get_data['end'] - get_data['start']
        transform_data = get_data.drop(columns=['start', 'end'])
        temp_data = pd.DataFrame({'chord': ['none'], 'time': [first_count]})
        refined_data = pd.concat([temp_data, transform_data], axis=0, ignore_index=True)
        refined_data['frame'] = fps * refined_data['time']
        reformed_data = refined_data.drop(columns=['time'])

        half_round = 0
        temp = 0
        ban_round_list = list()

        for row_count in range(len(reformed_data)):
            ban_new = round(reformed_data['frame'][row_count]) + temp

            diff = ban_new - reformed_data['frame'][row_count]

            ban_round_list += [ban_new]

            half_round = half_round + diff

            if half_round >= 1:
                temp = -1
                half_round = half_round - 1
            elif half_round <= -1:
                temp = 1
                half_round = half_round + 1
            else:
                temp = 0

        reformed_data['half_round'] = ban_round_list
        input_to_audio = reformed_data.drop(columns=['frame'])

        chord_list = []

        for num in range(len(input_to_audio)):
            row_count = 1
            while row_count <= input_to_audio['half_round'][num]:
                chord_list.append(input_to_audio['chord'][num])
                row_count += 1

        index = 0

        while cap.isOpened():

            ret, image = cap.read()
            image = cv2.flip(image, 1)
            image_orig = cv2.flip(image, 1)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results_hand = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_landmark_drawing_spec,
                        connection_drawing_spec=hand_connection_drawing_spec)
            res = cv2.bitwise_and(image, cv2.bitwise_not(image_orig))

            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            ret, th1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contours = sorted(contours, key=cv2.contourArea)
                contour = contours[-1]
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = gray[y1:y1 + h1, x1:x1 + w1]
                save_img = cv2.resize(save_img, (image_x, image_y))
                pred_probab, pred_class = keras_predict(model, save_img)
                print(pred_class, pred_probab)

                cv2.putText(image, str(chord_list[index]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 9)

                if chord_list[index] == chord_dict[pred_class]:
                    cv2.putText(image, str(chord_dict[pred_class]), (x1 + 50, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                (255, 0, 0), 9)
                else:
                    cv2.putText(image, str(chord_dict[pred_class]), (x1 + 50, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 0, 255), 9)

                index += 1

                keypress = cv2.waitKey(1)
                if keypress == ord('c'):
                    break
                image = rescale_frame(image, percent=75)
                resize_img = cv2.resize(image, (640, 480))
                chord_img = cv2.imread("chord_img.png")
                cv2.imshow("chord_img", chord_img)
                cv2.imshow("Img", resize_img)

        hands.close()
        cap.release()

    def keras_predict(model, image):
        processed = keras_process_image(image)
        pred_probab = model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class

    def keras_process_image(img):
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (-1, image_x, image_y, 1))
        return img

    return sub()


# csvupload 하기
@app.route('/csvupload', methods=['GET', 'POST'])
def csvupload():
    return render_template('csv_upload.html')


# 초기 화면
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5010, debug=True)
