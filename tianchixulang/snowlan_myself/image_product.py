from keras.preprocessing.image import ImageDataGenerator ,array_to_img,img_to_array,load_img
from imutils import paths
import random
import os

'''
    ImageDataGenerator() 参数使用
    # rotation_range：整数，数据提升时图片随机转动的角度
    # width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度`
    # height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    # rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
    # shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）
    # zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    # fill_mode：‘constant’‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
    # cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
    # channel_shift_range: Float. Range for random channel shifts.
    # horizontal_flip：布尔值，进行随机水平翻转
    # vertical_flip：布尔值，进行随机竖直翻转
    # rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前
'''
def image_agment(path, image_number):
    datagen = ImageDataGenerator(
        rotation_range=20,
        rescale = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
        )

    #img = load_img("xuelang\\修印\\J01_2018.06.16 10_24_16.jpg")
    img = load_img(path)
    x = img_to_array(img)
    print(x.shape)
    x =x.reshape((1,)+x.shape)
    print(x.shape)
    i = 0

    '''
        flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix=”, save_format=’jpeg’)：
        接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据.
        # X：样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
        # y：标签
        # batch_size：整数，默认32
        # shuffle：布尔值，是否随机打乱数据，默认为True
        # save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
        # save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
        # save_format：”png”或”jpeg”之一，指定保存图片的数据格式,默认”jpeg”
        # yields:形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.
        # seed: 整数,随机数种子
    '''

    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=os.path.dirname(path), save_prefix='cat', save_format='jpg'):
        i += 1
        if i > image_number:
            break  # 否则生成器会退出循环

#image_number = 30
def image_produce(path ,image_number):
    base_path = os.listdir(path)
    for each_path in base_path:
        all_path = path+each_path
        if all_path != 'xuelang\正常':
            each_imagenumber = len(list(paths.list_images(all_path)))
            if each_imagenumber < image_number:
                multiple = int((image_number - each_imagenumber)/each_imagenumber)
                imagepaths = sorted(list(paths.list_images(all_path)))
                for each_path in imagepaths:
                    each_path = each_path.replace('\ ', ' ')
                    image_agment(each_path , multiple)

image_produce("xuelang\\" , 40)