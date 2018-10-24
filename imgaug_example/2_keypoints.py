import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
ia.seed(1)
image = ia.quokka(size=(256, 256))
keypoints = ia.KeypointsOnImage([
    ia.Keypoint(x=65, y=100),
    ia.Keypoint(x=75, y=200),
    ia.Keypoint(x=100, y=100),
    ia.Keypoint(x=200, y=80)
], shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # 改变亮度，不影响关键点
    iaa.Affine(
        rotate=10,
        scale=(0.5, 0.7)
    ) # 旋转10度，缩放到50%-70%，影响关键点
])

#我们现在可以把它应用到图像上然后再到关键点，它会导致相同的增强。
#重要：每批调用一次，否则您将始终得到与每批相同的增强！
seq_det = seq.to_deterministic()

#增加重点和图像
#因为我们只有一个图像和一个关键字的列表，
#我们使用图像和关键点将两个函数都转换为列表（批），然后0来反转它。
# 在实际实验中,你的变量很可能已经是列表了
image_aug = seq_det.augment_images([image])[0]
keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

#在增加之前/之后打印坐标（见下文）
#use after.x_int and after.y_int to get rounded integer coordinates
for i in range(len(keypoints.keypoints)):
    before = keypoints.keypoints[i]
    after = keypoints_aug.keypoints[i]
    print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
        i, before.x, before.y, after.x, after.y)
    )

#在增强之前/之后的关键点（如下所示）
image_before = keypoints.draw_on_image(image, size=7)
image_after = keypoints_aug.draw_on_image(image_aug, size=7)
plt.subplot(330 + 1 + i)
plt.imshow(image_before, cmap=plt.get_cmap('gray'))
plt.imshow(image_after, cmap=plt.get_cmap('gray'))
    # show the plot
plt.show()