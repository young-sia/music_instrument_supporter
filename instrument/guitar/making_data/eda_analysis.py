import warnings
from glob import glob
import numpy as np
import seaborn as sns
import PIL
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



# 데이터를 불러올 함수에 대한 정의
def load_image_for_eda(path):
    path = path
    images = []
    labels = []
    for filename in glob(path + "*"):
        for img in glob(filename + "/*.jpg"):
            an_img = PIL.Image.open(img)  # read img
            img_array = np.array(an_img)  # img to array
            images.append(img_array)  # append array to training_images
            label = filename.split('/')[3]  # get label
            labels.append(label)  # append label
    images = np.array(images)
    labels = np.array(labels)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = labels.reshape(-1, 1)
    return images, labels


# 이미지 데이터에 대한 EDA를 보인다.
def main():
    print(PIL.Image)
    warnings.filterwarnings(action='ignore')  # 경고 메세지 숨김
    training_images, training_labels = load_image_for_eda(path='./chord/train')  # path를 다음과 같이 train 폴더로 설정

    print("train 이미지 크기:", training_images.shape)
    print("train 라벨 크기:", training_labels.shape)

    plt.figure(figsize=[24, 6])
    for i in range(24):
        import random
        num = random.randint(0, training_images[0])
        plt.subplot(2, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(training_labels[num][0])
        plt.imshow(training_images[num])
    plt.show()
    print(np.unique(training_labels))

    plt.figure(figsize=[10, 5])
    sns.countplot(training_labels.ravel())
    plt.title('Distribution of training data')
    plt.xlabel('Classes')
    plt.show()

if __name__ == '__main__':
    main()