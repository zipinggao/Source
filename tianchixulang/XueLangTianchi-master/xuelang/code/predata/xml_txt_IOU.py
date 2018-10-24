import xml.etree.ElementTree as ET
import os
import glob
'''
写入txt格式为

xml对应图片绝对路径   含物体个数   第一个物体对应bbox对应4个坐标       第一个物体对应bbox对应物体名称
0001.jpg               1           Xmin Ymin Xmax Ymax    破洞
'''
class Xml_Txt_IOU():
    def parse_rec(self,filename):
        """
        解析一个 PASCAL VOC xml file
        """
        tree = ET.parse(filename)
        # 存储一张图片中的所有物体
        objects = []
        # 遍历一张图中的所有物体
        for obj in tree.findall('object'):
            obj_struct = {}
            # 读取物体名称
            obj_struct['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            # 从原图左上角开始为原点，向右为x轴，向下为y轴。左上角（xmin，ymin）和右下角(xmax,ymax)
            obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                                  int(float(bbox.find('ymin').text)),
                                  int(float(bbox.find('xmax').text)),
                                  int(float(bbox.find('ymax').text))]
            objects.append(obj_struct)

        return objects


    def xml_txt_IOU(self):

        # 新建一个名为train_bobo的txt文件，准备写入数据
        root = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '/'
        txt_file = open(root+'../../data/xuelangtrainpart123.txt','w')

        # 存有jpg及xml的文件夹
        file = '../../data'


        img_list=['xuelang_round1_train_part1_20180628','xuelang_round1_train_part2_20180705','xuelang_round1_train_part3_20180709']

        for files_part in img_list:
            xml_file = os.path.join(file, files_part)
            # 拿到该文件夹下所有xml文件
            xml_files = glob.glob(os.path.join(root+xml_file, '*/*.xml'))
            xml_files.sort()

            # 遍历所有xml文件，将文件写入txt中
            for xml_dir in xml_files:
                # str.replace('a','b') 将字符串中的a替换为b
                image_name = xml_dir.split('/')[-1].replace('xml','jpg')
                # image_path为xml对应图像的绝对路径
                image_path=image_name
                # txt 写入图像名字（绝对路径）
                txt_file.write(image_path + ',')
                # results保存的是bbox物体的名字与对应4个坐标
                results = self.parse_rec(xml_dir)
                # num_obj 一张图中的物体总数
                num_obj=len(results)
                # txt写入  一张图中的物体总数
                txt_file.write(str(num_obj) + ',')
                # 遍历一张图片中的所有物体
                for result in results:
                    name = result['name']
                    bbox = result['bbox']
                    # txt写入   bbox的坐标 以及  每个物体对应的类
                    txt_file.write(
                        str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ',' + name + ',')
                # 读取完一张图片后换行
                txt_file.write('\n')
        # 关闭txt
        txt_file.close()
        print('xuelangtrainpart123.txt generate success')

xml_txt_IOU=Xml_Txt_IOU().xml_txt_IOU()