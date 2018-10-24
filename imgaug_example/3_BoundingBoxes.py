import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
ia.seed(1)

image = ia.quokka(size=(256, 256))
bbs = ia.BoundingBoxesOnImage([
    ia.BoundingBox(x1=65, y1=100, x2=200, y2=150),
    ia.BoundingBox(x1=150, y1=80, x2=200, y2=130)
], shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px={"x": 40, "y": 60},
        scale=(0.5, 0.7)
    ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])

#我们现在可以把它应用到图像上，然后再到BBs上，它会导致相同的增强。
#重要：每批调用一次，否则您将永远得到对每一批都是一样的增强！
seq_det = seq.to_deterministic()

#增加BBS和图片。
#由于我们只有一个图像和一个BBs列表，我们使用图像和BBs将它们转换为函数的列表（成批），然后0来反转它。
#在一个真实的实验中，你的变量很可能已经是列表了。
image_aug = seq_det.augment_images([image])[0]
bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

# 在增加之前/之后打印坐标（见下文）
# use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )

#在增加之前/之后的BBs图片（如下所示
image_before = bbs.draw_on_image(image, thickness=2)
image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])
plt.subplot(330 + 1 + 1)
plt.imshow(image_before, cmap=plt.get_cmap('gray'))
plt.subplot(330 + 1 + 2)
plt.imshow(image_after, cmap=plt.get_cmap('gray'))
    # show the plot
plt.show()