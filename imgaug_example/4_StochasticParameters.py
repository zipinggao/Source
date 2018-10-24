from imgaug import augmenters as iaa
from imgaug import parameters as iap
from os import environ
'''
seq = iaa.Sequential([
    iaa.GaussianBlur(
        sigma=iap.Uniform(0.0, 1.0)
    ),
    iaa.ContrastNormalization(
        iap.Choice(
            [1.0, 1.5, 3.0],
            p=[0.5, 0.3, 0.2]
        )
    ),
    iaa.Affine(
        rotate=iap.Normal(0.0, 30),
        translate_px=iap.RandomSign(iap.Poisson(3))
    ),
    iaa.AddElementwise(
        iap.Discretize(
            (iap.Beta(0.5, 0.5) * 2 - 1.0) * 64
        )
    ),
    iaa.Multiply(
        iap.Positive(iap.Normal(0.0, 0.1)) + 1.0
    )
])
'''
params = [
    iap.Normal(0, 1),
    iap.Normal(5, 3),
    iap.Normal(iap.Choice([-3, 3]), 1),
    iap.Normal(iap.Uniform(-3, 3), 1)
]



