from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# 이미지 크기
image_x, image_y = 200, 200
batch_size = 32
train_dir = "chords"


# 케라스 모델 정의
def keras_model(image_x, image_y):
    # 코드 7개 학습했으므로 읽을 폴더의 갯수 : 7개
    num_of_classes = 35
    # 순차적으로 레이어 층을 더해주는 순차 모델 사용
    model = Sequential()
    # 딥러닝 네트워크에서 노드에 입력된 값들을 비선형 함수에 통과 시킨수 다음 레이어로 전달하는 활성화 함수를
    # 가장 많이 사용되는 relu를 사용한다.
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    # 공간적 데이터에 대한 최대값 풀링 작업
    #  인풋을 두 공간에 차원에 대해 반으로 축소
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    # 과적합 방지
    model.add(Dropout(0.6))
    # 다중 클래스 분류 모델 생성
    model.add(Dense(num_of_classes, activation='softmax'))

    # 범주형 교차 엔트로피 오차 사용, 최적화 알고리즘 옵티마이저는 가장 많이 쓰이는 adam, 분류 모델 성능 평가 지표 = 정확도 사용
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 모델을 저장할 경로
    filepath = "guitar_learner.h5"
    # validation set의 loss가 가장 적을 때 저장, 모델 저장 후 화면 표시, monitor 되고 있는 값을 기준으로 가장 좋은 값으로 모델 저징
    # 분류 모델의 성능 평가 지표가 정확도이므로 클수록 좋음
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def main():
    # 적은 양의 데이터를 가지고 이미지 분류 모델 구축을 위해 실시간 이미지 증가(agumentation)
    train_datagen = ImageDataGenerator(
        # RGB영상의 계수로 구성된 원본 영상을 모델에 효과적으로 학습 시키기 위해 1/255로 스케일링하여 0-1 범위로 변환
        rescale=1. / 255,
        # 이미지를 수평, 수직으로 랜덤하게 평행 이동 시키는 범위 지정
        width_shift_range=0.2,
        height_shift_range=0.2,
        # 임의 전단 변환 범위
        shear_range=0.2,
        # 이미지 회전 범위
        rotation_range=15,
        # 임의 확대, 축소 범위
        zoom_range=0.2,
        # 이미지 좌우반전 x
        horizontal_flip=False,
        # 학습 사 데이터 일부를 나눠거 validation으로 사용할 비율
        validation_split=0.2,
        # fill_mode는 디폴트 값 사용
        fill_mode='nearest')

    # 디렉토리 설정
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        # 추후 모델에 들어갈 인풋 이미지 사이즈(200,200)
        target_size=(image_x, image_y),
        # 그레이 스케일 사용
        color_mode="grayscale",
        # 이미지 데이터 원본 소스에서 가져올 이미지 수
        batch_size=batch_size,
        # 데이터 셔플링과 변형애 사용할 선택적 난수 시드
        seed=42,
        # class_mode는 디폴트 값
        class_mode='categorical',
        # imagedatagenerator에 validation_split이 설정되어 있으므오 부분집합
        subset="training")
    # 검증
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="validation")

    model, callbacks_list = keras_model(image_x, image_y)

    # batch 단위로 생성된 데이터에 모델 피팅
    # 5번 동안 데이터에 대해 반복 수행, 각 epoch 끈테 검증 생성기로 부터 얻는 단계 숫자
    model.fit_generator(train_generator, epochs=5, validation_data=validation_generator)
    scores = model.evaluate_generator(generator=validation_generator, steps=64)
    # 모델 평가
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    # 모델 저장
    model.save('guitar_learner.h5')


if __name__ == '__main__':
    main()

