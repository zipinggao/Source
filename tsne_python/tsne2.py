
# coding='utf-8'
"""t-SNE植物幼苗分类进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn import datasets
from sklearn.manifold import TSNE
import cv2
from keras.preprocessing.image import img_to_array
import os
from matplotlib.ticker import NullFormatter

CLASS = {
    'Black-grass': 0,
    'Charlock': 1,
    'Cleavers': 2,
    'Common Chickweed': 3,
    'Common wheat': 4,
    'Fat Hen': 5,
    'Loose Silky-bent': 6,
    'Maize': 7,
    'Scentless Mayweed': 8,
    'Shepherds Purse': 9,
    'Small-flowered Cranesbill': 10,
    'Sugar beet': 11
}

def get_data():
    data = []
    label = []
    datapath = "..\\data\\train\\"
    imagepaths = sorted(list(paths.list_images(datapath)))
    # print(imagepaths)
    i = 0
    for path in imagepaths:
        if '\\ ' in str(path):
            path = path.replace('\\ ', ' ')
        image = cv2.imread(path)
        image = cv2.resize(image ,(256,256),interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        image = image.flatten()
        data.append(image)
        label.append(CLASS[path.split(os.path.sep)[3]])
    data = np.array(data)
    label = np.array(label)
    return data ,label

def main():
    data , label= get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    t1 = time()
    fig = plt.figure(figsize=(8, 8))
    # 创建了一个figure，标题为"Manifold Learning with 1000 points, 10 neighbors"
    plt.suptitle("plat seed classcifunction", fontsize=14)
    ax = fig.add_subplot(2, 1, 1)
    plt.scatter(result[:, 0], result[:, 1], c=label, cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    plt.show()

if __name__ == '__main__':
    main()
