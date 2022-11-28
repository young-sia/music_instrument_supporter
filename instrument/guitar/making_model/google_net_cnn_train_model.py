from keras.models import Model
from keras.layers import MaxPooling2D, Dropout, Input, Conv2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merging import concatenate


image_x, image_y = 200, 200
batch_size = 64
train_dir = "chords"
num_of_classes = 21


def inception_block_for_googlenet(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
    # Input: - f1: number of filters of the 1x1 convolutional layer in the first path - f2_conv1, f2_conv3 are number
    # of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path - f3_conv1, f3_conv5 are
    # the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path - f4: number of
    # filters of the 1x1 convolutional layer in the fourth path

    # 1st path:
    path1 = Conv2D(filters=f1, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # 2nd path
    path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=f2_conv3, kernel_size=(2, 2), padding='same', activation='relu')(path2)

    # 3rd path
    path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=f3_conv5, kernel_size=(3, 3), padding='same', activation='relu')(path3)

    # 4th path
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    output_layer = concatenate([path1, path2, path3, path4], axis=-1)

    return output_layer


def googlenet_cnn_model(image_x, image_y):

    input_layer = Input(shape=(image_x, image_y, 3))

    # convolutional layer: filters = 64, kernel_size = (5,5), strides = 2
    make_network = Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu')(input_layer)

    # max-pooling layer: pool_size = (2,2), strides = (2, 2)
    make_network = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(make_network)

    # convolutional layer: filters = 64
    make_network = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(make_network)

    # convolutional layer: filters = 192, kernel_size = (5,5)
    make_network = Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation='relu')(make_network)

    # max-pooling layer: pool_size = (3,3), strides = (3, 3)
    make_network = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(make_network)

    # 1st Inception block
    make_network = inception_block_for_googlenet(make_network, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16,
                                                 f3_conv5=32, f4=32)

    # 2nd Inception block
    make_network = inception_block_for_googlenet(make_network, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32,
                                                 f3_conv5=96, f4=64)

    # max-pooling layer: pool_size = (3,3), strides = (3, 3)
    make_network = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(make_network)

    # 3rd Inception block
    make_network = inception_block_for_googlenet(make_network, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16,
                                                 f3_conv5=48, f4=64)

    # Extra network 1:
    make_extra_network = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(make_network)
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
    make_extra_network2 = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(make_network)
    make_extra_network2 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(make_extra_network2)
    make_extra_network2 = Flatten()(make_extra_network2)
    make_extra_network2 = Dense(1024, activation='relu')(make_extra_network2)
    make_extra_network2 = Dropout(0.7)(make_extra_network2)
    make_extra_network2 = Dense(1000, activation='softmax')(make_extra_network2)

    # 7th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                                                 f3_conv5=128, f4=128)

    # max-pooling layer: pool_size = (3,3), strides = (3, 3)
    make_network = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(make_network)

    # 8th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                                                 f3_conv5=128, f4=128)

    # 9th Inception block
    make_network = inception_block_for_googlenet(make_network, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48,
                                                 f3_conv5=128, f4=128)

    # Global Average pooling layer
    make_network = GlobalAveragePooling2D(name='GAPL')(make_network)

    # Dropoutlayer
    make_network = Dropout(0.6)(make_network)

    # output layer
    make_network = Dense(1024, activation='softmax')(make_network)

    # model
    model = Model(input_layer, [make_network, make_extra_network, make_extra_network2], name='GoogLeNet')

    return model


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

    model, callbacks_list = googlenet_cnn_model(image_x, image_y)
    model.fit_generator(train_generator, epochs=5, validation_data=validation_generator)
    scores = model.evaluate_generator(generator=validation_generator, steps=64)
    model.save('google_net_guitar_learner.h5')
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))


if __name__ == '__main__':
    main()
