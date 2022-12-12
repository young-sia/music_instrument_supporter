from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import time
import datetime

import tensorflow as tf

# 이미지 크기
image_x, image_y = 200, 200
batch_size = 32
epochs = 20
train_dir = "chords"


# 케라스 모델 정의
def vgg19_cnn_model(image_x, image_y):
    # 코드 35개 학습했으므로 읽을 폴더의 갯수 : 35개
    num_of_classes = 35
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(image_x, image_y, 3))

    # x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1')(input_shape)
    # x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2')(x)
    # x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool')(x)
    #
    # x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1')(x)
    # x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool')(x)
    #
    # x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1')(x)
    # x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2')(x)
    # x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3')(x)
    # x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv4')(x)
    # x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool')(x)
    #
    # x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2')(x)
    # x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3')(x)
    # x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv4')(x)
    # x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block4_pool')(x)
    #
    # x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv3')(x)
    # x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv4')(x)
    # x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block5_pool')(x)
    #
    # x = Flatten()(x)
    # # 과적합 방지
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation = 'relu', name = 'fc1')(x)
    # x = Dense(4096, activation = 'relu', name = 'fc2')(x)
    # # 다중 클래스 분류 모델 생성
    # x = Dense(num_of_classes, activation='softmax', name = 'predicitons')(x)

    # vgg19을 사용해서 모델 생성
    vgg19.trainable = True
    for i in vgg19.layers[:528]:
        i.trainable = False

    for i in vgg19.layers[526:]:
        print(i.name, i.trainable)

    x = vgg19.output
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', input_dim=(200, 200, 3))(x)
    # x = BatchNormalization()(x)
    # x = Dense(256, activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dense(num_of_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=vgg19.input, outputs=x)

    # 범주형 교차 엔트로피 오차 사용, 최적화 알고리즘 옵티마이저는 가장 많이 쓰이는 adam, 분류 모델 성능 평가 지표 = 정확도 사용
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # 모델을 저장할 경로
    filepath = "guitar_learner.h5"
    # validation set의 loss가 가장 적을 때 저장, 모델 저장 후 화면 표시, monitor 되고 있는 값을 기준으로 가장 좋은 값으로 모델 저장
    # 분류 모델의 성능 평가 지표가 정확도이므로 클수록 좋음
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose = 1, save_best_only = True, mode = 'max')
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
    model, callbacks_list = vgg19_cnn_model(image_x, image_y)
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    scores = model.evaluate_generator(generator=validation_generator, steps=64)
    # 모델 평가
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    # 모델 저장
    model.save('vgg19_guitar_learner.h5')

    # 걸린 시간 측정
    end = time.time()
    sec = (end - start)
    time_check_list = str(datetime.timedelta(seconds=sec)).split(".")
    print(time_check_list[0])


if __name__ == '__main__':
    main()
