#-*-coding:utf-8-*-
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
from imutils import paths
import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(1)

def read_xml_annotation(xml_path):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    in_file = open(xml_path ,encoding="utf8")
    tree = ET.parse(in_file)
    root = tree.getroot()
    for box in root.iter('bndbox'):
        xmin.append(int(box.find('xmin').text))
        ymin.append(int(box.find('ymin').text))
        xmax.append(int(box.find('xmax').text))
        ymax.append(int(box.find('ymax').text))
    return (xmin ,xmax,ymin,ymax)

def change_xml_annotation(xml_path ,new_xml_path,bbs_aug):
    i = 0
    in_file = open(xml_path,encoding="utf8")
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    for box  in xmlroot.iter('bndbox'):
        after = bbs_aug.bounding_boxes[i]
        box.find('xmin').text = str(int(after.x1))
        box.find('ymin').text = str(int(after.y1))
        box.find('xmax').text = str(int(after.x2))
        box.find('ymax').text = str(int(after.y2))

        print("BB %d:(%.4f, %.4f, %.4f, %.4f)" % (
            i,
            after.x1, after.y1, after.x2, after.y2)
              )
        i +=1
    tree.write(new_xml_path,encoding="utf8")


def img_aug(img ,xmin, xmax, ymin, ymax):
    length = len(xmin)
    BoundingBox = []
    for i in range(length):
        BoundingBox.append(ia.BoundingBox(x1=xmin[i], y1=ymin[i], x2=xmax[i], y2=ymax[i]))
    bbs = ia.BoundingBoxesOnImage(BoundingBox, shape=img.shape)

    # 普通的数据增强
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 水平翻转

        iaa.Crop(percent=(0, 0.1)),  # 随机裁剪

        # gaussian blur sigma在0和0.5之间
        # 但我们只是模糊了所有图像的一半
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),

        # 加强或削弱每张图像的对比度
        iaa.ContrastNormalization((0.75, 1.5)),

        # 添加高斯噪声
        # 对于50%的图像，我们对每个像素的噪音进行采样。
        # 对于其他50%的图像，我们对每个像素和信道的噪声进行采样,这可以改变像素的颜色（不仅仅是亮度）。
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

        # 让一些图像更亮一些更暗一些。对于20%的图像区域，对每一个通道进行采样
        # 最终可能会改变图像的颜色
        iaa.Multiply((0.8, 1.2), per_channel=0.2),

        # 运用仿射变换到每一张图像，缩放/缩放，转换/移动它们，旋转它们，剪切它们
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 缩放
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 转化比例
            rotate=(-25, 25),  # 旋转
            shear=(-8, 8)  # 修剪
        )],
        random_order=True  # 按随机顺序应用增强器
    )

    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    '''
    # 显示坐标
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
              )
    '''

    # 显示图像

    image_before = bbs.draw_on_image(img, thickness=3)
    image_after = bbs_aug.draw_on_image(image_aug, thickness=3, color=[0, 0, 255])
    Image.fromarray(image_before).save("11.jpg")
    Image.fromarray(image_after).save("22.jpg")
    '''
    plt.subplot(330 + 1 + 1)
    plt.imshow(image_before,cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 2)
    plt.imshow(image_after, cmap=plt.get_cmap('gray'))
    plt.show()
    '''
    return image_aug ,bbs_aug

if __name__ == "__main__":
    path = "train\\"
    imagepaths = sorted(list(paths.list_images(path)))
    i = 1
    for imagepath in imagepaths:
        img = Image.open(imagepath)
        img = np.array(img)
        xml_path = imagepath.replace('jpg','xml')
        xmin, xmax, ymin, ymax = read_xml_annotation(xml_path)
        image_aug ,bbs_aug = img_aug(img ,xmin, xmax, ymin, ymax)
        new_image_path = imagepath.replace('.jpg' , '_aug_%d.jpg'%i)
        new_xml_path = new_image_path.replace('jpg' , 'xml')
        i +=1
        Image.fromarray(image_aug).save(new_image_path)
        change_xml_annotation(xml_path, new_xml_path, bbs_aug)
