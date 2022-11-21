from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential, Model
from keras.layers import MaxPooling2D, Dropout, BatchNormalization, Input, Conv2D, AveragePooling2D, \
    Flatten, GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.layers.merging import concatenate

import tensorflow as tf
from keras.applications.resnet_v2 import ResNet152V2

image_x, image_y = 200, 200
batch_size = 32
train_dir = "chords"
num_of_classes = 21


def ask_what_to_do():
    x1 = input("일반 CNN을 하겠습니까?(yes/no):")
    x2 = input("Resnet을 하겠습니까?(yes/no):")
    x3 = input('GoogleNet을 하시겠습니까?(yes/no)')

    return {'cnn': x1}, {'resnet': x2}, {'googlenet': x3}


def simple_cnn_model(image_x, image_y):
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
    filepath = "guitar_learner_cnn.h5"
    # validation set의 loss가 가장 적을 때 저장, 모델 저장 후 화면 표시, monitor 되고 있는 값을 기준으로 가장 좋은 값으로 모델 저징
    # 분류 모델의 성능 평가 지표가 정확도이므로 클수록 좋음
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def resnet_cnn_model(image_x, image_y):
    # 코드 25개 학습했으므로 읽을 폴더의 갯수 : 25개
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


def inception_block_for_googlenet(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    # Input: - f1: number of filters of the 1x1 convolutional layer in the first path - f2_conv1, f2_conv3 are number
    # of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path - f3_conv1, f3_conv5 are
    # the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path - f4: number of
    # filters of the 1x1 convolutional layer in the fourth path

    # 1st path:
    path1 = Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # 2nd path
    path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu')(path2)

    # 3rd path
    path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu')(path3)

    # 4th path
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    output_layer = concatenate([path1, path2, path3, path4], axis=-1)

    return output_layer


def googlenet_cnn_model(image_x, image_y):

    input_layer = Input(shape=(image_x, image_y, 3))

    # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
    make_network = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)

    # max-pooling layer: pool_size = (3,3), strides = 2
    make_network = MaxPooling2D(pool_size=(3, 3), strides=2)(make_network)

    # convolutional layer: filters = 64, strides = 1
    make_network = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(make_network)

    # convolutional layer: filters = 192, kernel_size = (3,3)
    make_network = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(make_network)

    # max-pooling layer: pool_size = (3,3), strides = 2
    make_network = MaxPooling2D(pool_size=(3, 3), strides=2)(make_network)

    # 1st Inception block
    make_network = inception_block_for_googlenet(make_network, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16,
                                                 f3_conv5=32, f4=32)

    # 2nd Inception block
    make_network = inception_block_for_googlenet(make_network, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32,
                                                 f3_conv5=96, f4=64)

    # max-pooling layer: pool_size = (3,3), strides = 2
    make_network = MaxPooling2D(pool_size=(3, 3), strides=2)(make_network)

    # 3rd Inception block
    make_network = inception_block_for_googlenet(make_network, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16,
                                                 f3_conv5=48, f4=64)

    # Extra network 1:
    make_extra_network = AveragePooling2D(pool_size=(5, 5), strides=3)(make_network)
    make_extra_network = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(make_extra_network)
    make_extra_network = Flatten()(make_extra_network)
    make_extra_network = Dense(1024, activation='relu')(make_extra_network)
    make_extra_network = Dropout(0.7)(make_extra_network)
    make_extra_network = Dense(5, activation='softmax')(make_extra_network)

    # 4th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24,
                                                 f3_conv5=64, f4=64)

    # 5th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24,
                                                 f3_conv5=64, f4=64)

    # 6th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32,
                                                 f3_conv5=64, f4=64)

    # Extra network 2:
    make_extra_network2 = AveragePooling2D(pool_size=(5, 5), strides=3)(make_network)
    make_extra_network2 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(make_extra_network2)
    make_extra_network2 = Flatten()(make_extra_network2)
    make_extra_network2 = Dense(1024, activation='relu')(make_extra_network2)
    make_extra_network2 = Dropout(0.7)(make_extra_network2)
    make_extra_network2 = Dense(1000, activation='softmax')(make_extra_network2)

    # 7th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                                                 f3_conv5=128, f4=128)

    # max-pooling layer: pool_size = (3,3), strides = 2
    make_network = MaxPooling2D(pool_size=(3, 3), strides=2)(make_network)

    # 8th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                                                 f3_conv5=128, f4=128)

    # 9th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48,
                                                 f3_conv5=128, f4=128)

    # Global Average pooling layer
    make_network = GlobalAveragePooling2D(name='GAPL')(make_network)

    # Dropoutlayer
    make_network = Dropout(0.4)(make_network)

    # output layer
    make_network = Dense(1000, activation='softmax')(make_network)

    # model
    model = Model(input_layer, [make_network, make_extra_network, make_extra_network2], name='GoogLeNet')

    return model


def main():
    permissions = ask_what_to_do()
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
    cnn_score = dict()

    for i in permissions:
        for key in i:
            if i[key] == 'yes':
                if key == 'cnn':
                    model, callbacks_list = simple_cnn_model(image_x, image_y)
                elif key == 'resnet':
                    model, callbacks_list = resnet_cnn_model(image_x, image_y)
                elif key == 'googlenet':
                    model, callbacks_list = googlenet_cnn_model(image_x, image_y)
                else:
                    print("something went wrong.")
                    pass
                model.fit_generator(train_generator, epochs=5, validation_data=validation_generator)
                scores = model.evaluate_generator(generator=validation_generator, steps=64)
                cnn_score[key] = scores[1]
                model.save(f'{key}_guitar_learner.h5')
            else:
                pass

    # 모델 평가
    for key, value in cnn_score:
        print(f"{key} Error: %.2f%%" % (100 - value * 100))

    score_dataframe = pd.DataFrame(cnn_score)
    score_dataframe.to_csv('/cnn error score.csv')


if __name__ == '__main__':
    main()
