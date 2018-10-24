from libtiff import TIFF3D, TIFF

# dirtype = ("train", "label", "test")
dirtype = ("test-volume", "train-labels","train-volume")
def split_img():
    for t in dirtype:
        imgdir = TIFF3D.open(t + '.tif')
        imgarr = imgdir.read_image()
        for i in range(imgarr.shape[0]):
            imgname = t + '/' + str(i) + ".tif"
            img = TIFF.open(imgname , 'w')
            img.write_image(imgarr[i])

def merge_img():
    imgdir = TIFF3D.open( 'test_mask_volume_server2.tif' , 'w')
    imgarr = []
    for i in range(30):
        img = TIFF.open('train-volume/'+str(i) + '.tif')
        imgarr.append(img.read_image())
    imgdir.write_image(imgarr)


if __name__  == "__main__":
    #split_img()
    merge_img()
