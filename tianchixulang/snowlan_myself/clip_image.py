import os
from PIL import Image
import xml.dom.minidom
import numpy as np
from imutils import paths
from keras.preprocessing.image import img_to_array
import shutil
import math

base_path = 'xuelang11'

def clip_coo(x1,y1,x2,y2):
    w1 = np.random.randint(low= x1,high=x2,size=1)
    w2 = np.random.randint(low= x1,high=x2,size=1)
    h1 = np.random.randint(low= y1,high=y2,size=1)
    h2 = np.random.randint(low= y1,high=y2,size=1)
    if w1[0] > w2[0]:
        a = w1[0]
        w1[0] = w2[0]
        w2[0] =a
    elif w1[0] == w2[0]:
         a = np.random.randint(low=30, high=200, size=1)
         w2[0] = w1[0] + a[0]

    if h1[0] > h2[0]:
        b = h1[0]
        h1[0] = h2[0]
        h2[0] =b
    elif h1[0] == h2[0]:
         b = np.random.randint(low=30, high=200, size=1)
         h2[0] = h1[0] + b[0]

    return w1[0],w2[0],h1[0],h2[0]

def negetive_clip_image():
    index = 0
    for root, dirs, files in os.walk(base_path):
        if index != 0:
            image_path = list(paths.list_images(root))
            for each_image in image_path:
                each_image = each_image.replace('\ ', ' ')
                img = Image.open(each_image)
                label = each_image.split(os.path.sep)[1]
                if label != '正常':
                    xml_path = each_image.replace('jpg','xml')
                    xml_path = xml_path.replace('\ ',' ')
                    Domtree = xml.dom.minidom.parse(xml_path)
                    annotation = Domtree.documentElement
                    objectlist = annotation.getElementsByTagName('object')
                    print(len(objectlist))
                    i = 1
                    for objects in objectlist:
                        bndbox = objects.getElementsByTagName('bndbox')
                        for box in bndbox:
                            x1_list = box.getElementsByTagName('xmin')
                            x1 = int(x1_list[0].childNodes[0].data)
                            y1_list = box.getElementsByTagName('ymin')
                            y1 = int(y1_list[0].childNodes[0].data)
                            x2_list = box.getElementsByTagName('xmax')
                            x2 = int(x2_list[0].childNodes[0].data)
                            y2_list = box.getElementsByTagName('ymax')
                            y2 = int(y2_list[0].childNodes[0].data)
                            w = x2-x1
                            h = y2-y1
                        region = img.crop((x1,y1,x2,y2))
                        savepath = each_image.replace('xuelang11','clipimage')
                        exitdir = os.path.exists(os.path.dirname(savepath))
                        if not exitdir:
                            os.makedirs(root)
                        savepath = savepath.replace('.jpg','_'+str(i)+'.jpg')
                        i +=1
                        region.save(savepath)
        index +=1


def create_image(x1,y1,x2,y2,img,each_image,i):
    w1, w2, h1, h2 = clip_coo(x1,y1,x2,y2)
    region = img.crop((w1, h1, w2, h2))
    savepath = each_image.replace('xuelang11', 'clipimage')
    savepath = savepath.replace('.jpg', '_' + str(i) + '.jpg')
    region.save(savepath)

def positive_clip_image():
    index = 0
    for root, dirs, files in os.walk(base_path):
        if index != 0:
            image_path = list(paths.list_images(root))
            for each_image in image_path:
                each_image = each_image.replace('\ ', ' ')
                img = Image.open(each_image)
                label = each_image.split(os.path.sep)[1]
                if label == '正常':
                    i = 0
                    create_image(0,0,1279,1920,img,each_image,i)
                    i +=1
                    create_image(1280,0,2560,1920,img,each_image,i)
                    i +=1
                    create_image(0,0,2560,960,img,each_image,i)
                    i +=1
                    create_image(0,960,2560,1920,img,each_image,i)
                    i +=1
                    create_image(0,0,1280,960,img,each_image,i)
                    i +=1
                    create_image(1280,0,2560,960,img,each_image,i)
                    i +=1
                    create_image(0,960,1280,1920,img,each_image,i)
                    i +=1
                    create_image(1280,960,2560,1920,img,each_image,i)
        index += 1

def copy_true():
    path = 'tureimage'
    newpath = 'clipimage\\正常\\'
    imagepaths = sorted(list(paths.list_images(path)))
    for i in range(1100):
        print(i)
        index = np.random.randint(low= 0,high=10527,size=1)[0]
        img = imagepaths[index].replace('\ ', ' ')
        newimg = img.replace('tureimage','clipimage\\正常')
        shutil.copy(img,newimg)


def contact_image():
    path = 'clipimage'
    toImage = Image.new('RGB', (1000, 750))
    image_paths = list(paths.list_images(path))
    for image_path in image_paths:
        image_path = image_path.replace('\ ', ' ')
        fromImage = Image.open(image_path)
        sizeimage = img_to_array(fromImage)
        attribute = sizeimage.shape
        width = attribute[1]
        height = attribute[0]
        divwidth = math.ceil(1000/width)
        divheight = math.ceil(750/height)
        for i in range (divwidth):
            for j in range(divheight):
                toImage.paste(fromImage, (i*width, j*height))
        save_path = image_path.replace('clipimage','splicimage')
        exitdir = os.path.exists(os.path.dirname(save_path))
        if not exitdir:
            os.makedirs(os.path.dirname(save_path))
        toImage.save(save_path)

def twoclasses():
    path = 'xuelang'
    imagepaths = sorted(list(paths.list_images(path)))
    for each_image in imagepaths:
        each_image = each_image.replace('\ ', ' ')
        label = each_image.split(os.path.sep)[1]
        filename = each_image.split(os.path.sep)[2]
        if label == '正常':
            newimg = "twoaddimage\\complete\\"+filename
            shutil.copy(each_image, newimg)
        else:
            newimg = "twoaddimage\\incomplete\\" + filename
            shutil.copy(each_image, newimg)

# negetive_clip_image()
# positive_clip_image()
# copy_true()
# contact_image()

twoclasses()

