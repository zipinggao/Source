# First row
from imgaug import augmenters as iaa
import numpy as np
import imgaug as ia
from imgaug import parameters as iap
from matplotlib import pyplot as plt
image = ia.quokka(size=(256, 256))
images = np.array(
    [ia.quokka(size=(128, 128)) for _ in range(8)],
    dtype=np.uint8
)

seq = iaa.Alpha(
    factor=(0.2, 0.8),
    first=iaa.Sharpen(1.0, lightness=2),
    second=iaa.CoarseDropout(p=0.1, size_px=8)
)

seq1 = iaa.Alpha(
    (0.0, 1.0),
    first=iaa.MedianBlur(11),
    per_channel=True
)

# Second row
seq2 = iaa.SimplexNoiseAlpha(
    first=iaa.EdgeDetect(1.0),
    per_channel=False
)

# Third row
seq3 = iaa.SimplexNoiseAlpha(
    first=iaa.EdgeDetect(1.0),
    second=iaa.ContrastNormalization((0.5, 2.0)),
    per_channel=0.5
)

# Forth row
seq4 = iaa.FrequencyNoiseAlpha(
    first=iaa.Affine(
        rotate=(-10, 10),
        translate_px={"x": (-4, 4), "y": (-4, 4)}
    ),
    second=iaa.AddToHueAndSaturation((-40, 40)),
    per_channel=0.5
)

# Fifth row
seq5 = iaa.SimplexNoiseAlpha(
    first=iaa.SimplexNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        second=iaa.ContrastNormalization((0.5, 2.0)),
        per_channel=True
    ),
    second=iaa.FrequencyNoiseAlpha(
        exponent=(-2.5, -1.0),
        first=iaa.Affine(
            rotate=(-10, 10),
            translate_px={"x": (-4, 4), "y": (-4, 4)}
        ),
        second=iaa.AddToHueAndSaturation((-40, 40)),
        per_channel=True
    ),
    per_channel=True,
    aggregation_method="max",
    sigmoid=False
)

seq6 = iaa.Alpha(
    factor=(0.2, 0.8),
    first=iaa.Affine(rotate=(-20, 20)),
    per_channel=True
)

seq7 = iaa.SimplexNoiseAlpha(
    first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
)

seq8 = iaa.SimplexNoiseAlpha(
    first=iaa.EdgeDetect(1.0),
    per_channel=True
)

seq9 = iaa.FrequencyNoiseAlpha(
    first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
)

seq10 = iaa.FrequencyNoiseAlpha(
    first=iaa.EdgeDetect(1.0),
    per_channel=True
)

seq11 = iaa.FrequencyNoiseAlpha(
    exponent=-2.0,
    first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
    size_px_max=32,
    upscale_method="linear",
    iterations=1,
    sigmoid=False
)
images_aug = seq11.augment_images(images)
print(np.array(images_aug).shape)
# images_aug = seq4.augment_images(image)
plt.subplot(330 + 1 + 1)
plt.imshow(images_aug[0]) #, cmap=plt.get_cmap('gray')
plt.subplot(330 + 1 + 2)
plt.imshow(images[0]) #, cmap=plt.get_cmap('gray')
plt.show()