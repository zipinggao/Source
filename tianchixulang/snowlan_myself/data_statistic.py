import os
from imutils import paths

path = "xuelang\\"

def show_number(path):
    num_files = 0
    index = 0;
    data_number = []
    all_data = 0
    data_numindex = {}
    for root ,dirs, files in  os.walk(path):
        if index != 0:
            num_root =list(paths.list_images(root))
            num_files = len(num_root)
            all_data += int(num_files)
            data_numindex[os.path.basename(root)] = int(num_files)
        index +=1
    print("false:%d ，ture：%d"%(all_data -data_numindex['正常'],data_numindex['正常']))
    data_numindex['正常'] = data_numindex['正常']
    print(data_numindex)
    print(all_data)
    print(len(data_numindex))
show_number(path)
