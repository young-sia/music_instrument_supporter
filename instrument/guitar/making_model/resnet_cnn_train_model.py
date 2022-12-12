from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from keras.applications.resnet_v2 import ResNet152V2

import time
import datetime

# 이미지 크기
image_x, image_y = 100, 100
batch_size = 128
epochs = 5
train_dir = "chords"


# 케라스 모델 정의
def resnet_model(image_x, image_y):
    # 코드 35개 학습했으므로 읽을 폴더의 갯수 : 35개
    num_of_classes = 35
    resnet = ResNet152V2(include_top=False, weights='imagenet', input_shape=(image_x, image_y, 3))

    # resnet을 사용해서 모델 생성
    resnet.trainable = True
    for i in resnet.layers[:528]:
        i.trainable = False

    for i in resnet.layers[525:]:
        print(i.name, i.trainable)

    x = resnet.output
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', input_dim=(200, 200, 3))(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(num_of_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=resnet.input, outputs=x)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "guitar_learner.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def main():
    # 걸린 시간 측정 시작
    start = time.time()

    print("Batch Size: ", batch_size, ", Epochs: ", epochs)
    # 적은 양의 데이터를 가지고 이미지 분류 모델 구축을 위해 실시간 이미지 증가(agumentation)
    train_datagen = ImageDataGenerator(
        # RGB영상의 계수로 구성된 원본 영상을 모델에 효과적으로 학습 시키기 위해 1/255로 스케일링하여 0-1 범위로 변환
        rescale=1. / 255,
        # 이미지를 수평, 수직으로 랜덤하게 평행 이동 시키는 범위 지정
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # 임의 전단 변환 범위
        # shear_range=0.2,
        # 이미지 회전 범위
        # rotation_range=15,
        # 임의 확대, 축소 범위
        # zoom_range=0.2,
        # 이미지 좌우반전 x
        # horizontal_flip=False,
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
        color_mode="rgb",
        # 이미지 데이터 원본 소스에서 가져올 이미지 수
        batch_size=batch_size,
        # 데이터 셔플링과 변형애 사용할 선택적 난수 시드
        seed=42,
        # class_mode는 디폴트 값
        class_mode='categorical',
        subset="training")
    # 검증
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="rgb",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="validation")

    # batch 단위로 생성된 데이터에 모델 피팅
    # 5번 동안 데이터에 대해 반복 수행, 각 epoch 끈테 검증 생성기로 부터 얻는 단계 숫자
    model, callbacks_list = resnet_model(image_x, image_y)
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    scores = model.evaluate_generator(generator=validation_generator, steps=64)
    # 모델 평가
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    # 모델 저장
    model.save('resnet_guitar_learner.h5')

    # 걸린 시간 측정
    end = time.time()
    sec = (end - start)
    time_check_list = str(datetime.timedelta(seconds=sec)).split(".")
    print(time_check_list[0])


if __name__ == '__main__':
    main()
