import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from matplotlib import pyplot as plt
ia.seed(1)
import cv2
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(32)],
    dtype=np.uint8
)
''''''''''''''''''''''''''''''''''''''''''''''''''
#普通的数据增强
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # 水平翻转

    iaa.Crop(percent=(0,0.1)), # 随机裁剪

    #gaussian blur sigma在0和0.5之间
    #但我们只是模糊了所有图像的一半
    iaa.Sometimes(0.5 , iaa.GaussianBlur(sigma=(0,0.5))),

    #加强或削弱每张图像的对比度
    iaa.ContrastNormalization((0.75, 1.5)),

    #添加高斯噪声
    #对于50%的图像，我们对每个像素的噪音进行采样。
    #对于其他50%的图像，我们对每个像素和信道的噪声进行采样,这可以改变像素的颜色（不仅仅是亮度）。
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

    #让一些图像更亮一些更暗一些。对于20%的图像区域，对每一个通道进行采样
    #最终可能会改变图像的颜色
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

    #运用仿射变换到每一张图像，缩放/缩放，转换/移动它们，旋转它们，剪切它们
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},   #缩放
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #转化比例
        rotate=(-25, 25), #旋转
        shear=(-8, 8) #修剪
    )] ,
    random_order = True  #按随机顺序应用增强器
)

''''''''''''''''''''''''''''''''''''''''''''''''''
#严重的图像增强
#Sometimes(0.5, ...)，在50％的情况下应用给定的增强器(增强系数)
#e.g. Sometimes(0.5, GaussianBlur(0.3)) 会模糊每一秒的图像
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq1 = iaa.Sequential([
    #对大多数图像应用以下增强器
    iaa.Fliplr(0.5), #对于所有的图像水平翻转50%
    iaa.Flipud(0.2),#对于所有的图像垂直翻转50%

    #将一些图像裁剪成其高度/宽度的0-10%
    sometimes(iaa.Crop(percent=(0, 0.1))),

    #运用仿射变换到每一张图像
    #scale:80%-120%的图像高度/宽度（每个轴独立）
    #translate: 相对于高度/宽度（每轴），从-20到+20
    #rotate: -45度到+45度
    #shear:  -16到+16度
    #order: 使用最近的邻居或双线性插值（快速）
    #mode: 使用任何可用的模式来填充新创建的像素，查看API或scikit-image，以获得可用的模式
    #cval: 如果模式是常量，那么就为新创建的像素使用一个随机亮度（例如，有时是黑色的，有时是白色的）
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-16, 16),
        order=[0, 1],
        cval=(0, 255),
        mode=ia.ALL
    )),

    #执行以下（不太重要的）增强器的0到5个
    #image. 不要执行所有的任务，因为这通常太强大了。
    iaa.SomeOf((0,5) ,
               [
                   #将一些图像转换为它们的超像素表示
                   #每个图像的20到200个超像素，但是不要用它们的平均值替换所有的超像素，只有一些（p_replace）
                   sometimes(
                       iaa.Superpixels(
                           p_replace=(0, 1.0),
                           n_segments=(20, 200)
                       )
                   ),

                   #用不同的强度来模糊每个图像
                   #高斯模糊（0和3之间的sigma）
                   #平均/均匀模糊（内核大小在2x2和7x7之间）
                   #中值模糊（内核大小在3x3和11x11之间）
                   iaa.OneOf([
                       iaa.GaussianBlur((0, 3.0)),
                       iaa.AverageBlur(k=(2, 7)),
                       iaa.MedianBlur(k=(3, 11)),
                   ]),

                   #锐化每个图像，将结果叠加在原始图像上
                   #在0（没有锐化）和1(全锐化)之间使用alpha值
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                   #和锐化一样，但对于压纹效应
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    #在一些图像中搜索所有的边或有向边
                   #这些边被标记为黑白图像，并以0到0.7的alpha值与原始图像重叠。
                   sometimes(iaa.OneOf([
                       iaa.EdgeDetect(alpha=(0, 0.7)),
                       iaa.DirectedEdgeDetect(
                           alpha=(0, 0.7), direction=(0.0, 1.0)
                       ),
                   ])),

                   #在一些图像中加入高斯噪声。
                   #在这些案例中，有50%的噪音是通过每个通道和像素随机抽取的。
                   #在其他的50%的情况下，它每一个像素（即亮度变化）一次采样。
                   iaa.AdditiveGaussianNoise(
                       loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                   ),

                    #在所有像素中随机下降1%到10%（也就是将它们设置为黑色），或者将它们放在图片上，以2%-5%的原始尺寸
                    #导致了大的掉落的矩形。
                   iaa.OneOf([
                       iaa.Dropout((0.01, 0.1), per_channel=0.5),
                       iaa.CoarseDropout(
                           (0.03, 0.15), size_percent=(0.02, 0.05),
                           per_channel=0.2
                       ),
                   ]),

                    #将每个图像的chanell与5%的概率相反
                   #这将每个像素值v设置为255-v。
                   iaa.Invert(0.05, per_channel=True), # 反转颜色通道

                   #添加一个10到每个像素值为-10
                    iaa.Add((-10, 10), per_channel=0.5),

                   #改变图像的亮度（原始值的50%-150%）
                   iaa.Multiply((0.5, 1.5), per_channel=0.5),

                   #改善或恶化图像的对比。
                   iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                   #将每个图像转换为灰度，然后覆盖
                   #将每个图像转换为灰度，然后将结果与原始的alpha值重叠。也就是说，去掉不同强度的颜色。
                   iaa.Grayscale(alpha=(0.0, 1.0)),

                   #在一些图像中，在局部移动像素（有随机的优势）。
                   sometimes(
                       iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                   ),

                   #有些图像扭曲了局部区域的强度
                   sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
               ],
               #所有上述的增强都是随机的
               random_order=True
               )
],)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
images_aug = seq1.augment_images(images)
num = range(0,9)
# for i in num:
#     plt.subplot(330 + 1 + i)
#     plt.imshow(images[i], cmap=plt.get_cmap('gray'))
#     # show the plot
# plt.show()
for i in num:
    plt.subplot(330 + 1 + i)
    plt.imshow(images_aug[i], cmap=plt.get_cmap('gray'))
    # show the plot
plt.show()
