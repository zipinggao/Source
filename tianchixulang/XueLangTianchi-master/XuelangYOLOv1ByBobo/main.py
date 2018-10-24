import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from XuelangYOLOv1ByBobo.config import opt
from XuelangYOLOv1ByBobo.data.dataset import yoloDataset
from XuelangYOLOv1ByBobo.models.net import vgg16
from XuelangYOLOv1ByBobo.utils.visualize import Visualizer
from XuelangYOLOv1ByBobo.utils.yoloLoss import yoloLoss
from XuelangYOLOv1ByBobo.utils.predictUtils import predict_result
import glob
import  os


def train():
    vis=Visualizer(opt.env)
    # 网络部分======================================================开始
    # True则返回预训练好的VGG16模型
    net = vgg16(pretrained=True)
    # 提取特征层不动
    # 修改分类层最后三层全连接层
    # 修改vgg的分类层结构
    # 修改vgg16的分类部分
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 2842),
    )
    # 初始化网络的线性层 权重及偏向
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    # 将模型加载到内存中（CPU）
    if opt.load_model_path:
        net.load_state_dict(torch.load(opt.load_model_path,map_location=lambda  storage,loc:storage))
    # 再将模型转移到GPU上
    if opt.use_gpu:
        net.cuda()

    # 将模型调整为训练模式
    net.train()
    # 网络部分======================================================结束

    # 加载数据部分====================================================开始
    # 自定义封装数据集
    train_dataset = yoloDataset(root=opt.file_root, list_file=opt.voc_2012train, train=True, transform=[transforms.ToTensor()])
    # 数据集加载器  shuffle：打乱顺序    num_workers：线程数
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    test_dataset = yoloDataset(root=opt.test_root, list_file=opt.voc_2007test, train=False, transform=[transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    # 加载数据部分====================================================结束

    #自定义的损失函数  7代表将图像分为7x7的网格   2代表一个网格预测两个框   5代表 λcoord  更重视8维的坐标预测     0.5代表没有object的bbox的confidence loss
    criterion = yoloLoss(7, 2, 5, 0.5)
    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print('训练集有 %d 张图像' % (len(train_dataset)))
    print('一个batch的大小为 %d' % (opt.batch_size))
    # 将训练过程的信息写入log文件中
    logfile = open('log/log.txt', 'w')
    # inf为正无穷大
    best_test_loss = np.inf

    for epoch in range(opt.num_epochs):
        if epoch == 1:
            opt.learning_rate = 0.0005
        if epoch == 2:
            opt.learning_rate = 0.00075
        if epoch == 3:
            opt.learning_rate = 0.001
        if epoch == 80:
            opt.learning_rate = 0.0001
        if epoch == 100:
            opt.learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.learning_rate
        # 第几次epoch  及 当前epoch的学习率
        print('\n\n当前的epoch为 %d / %d' % (epoch + 1, opt.num_epochs))
        print('当前epoch的学习率: {}'.format(opt.learning_rate))

        # 每轮epoch的总loss
        total_loss = 0.
        # 开始训练
        for i, (images, target) in enumerate(train_loader):
            images = Variable(images)
            target = Variable(target)
            if opt.use_gpu:
                images, target = images.cuda(), target.cuda()
            # 前向传播，得到预测值
            pred = net(images)
            # 计算损失  yoloLoss继承nn.Module，调用方法名即自动进行前向传播，执行forward方法
            loss = criterion(pred, target)
            total_loss += loss.data[0]
            # 优化器梯度清零
            optimizer.zero_grad()
            #loss反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            if (i + 1) % opt.print_freq == 0:
                print('在训练集上：当前epoch为 [%d/%d], Iter [%d/%d] 当前batch损失为: %.4f, 当前epoch到目前为止平均损失为: %.4f'
                      % (epoch + 1, opt.num_epochs, i + 1, len(train_loader), loss.data[0], total_loss / (i + 1)))
                # 画出训练集的平均损失
                vis.plot_train_val(loss_train=total_loss / (i + 1))
        # 保存最新的模型
        torch.save(net.state_dict(),opt.current_epoch_model_path)

        # 一次epoch验证
        validation_loss = 0.0
        # 模型调整为验证模式
        net.eval()
        # 每轮epoch之后用VOC2007测试集进行验证
        for i, (images, target) in enumerate(test_loader):
            images = Variable(images, volatile=True)
            target = Variable(target, volatile=True)
            if opt.use_gpu:
                images, target = images.cuda(), target.cuda()
            # 前向传播得到预测值
            pred = net(images)
            # loss
            loss = criterion(pred, target)
            validation_loss += loss.data[0]
        # 计算在VOC2007测试集上的平均损失
        validation_loss /= len(test_loader)
        # 画出验证集的平均损失
        vis.plot_train_val(loss_val=validation_loss)
        # 训练模型的目标是 在验证集上的loss最小
        # 保存到目前为止 在验证集上的loss最小 的模型
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('当前得到最好的验证集的平均损失为  %.5f' % best_test_loss)
            torch.save(net.state_dict(),opt.best_test_loss_model_path)
        # 将当前epoch的参数写入log文件中
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()

def predict():
    # fasle 返回 未训练的模型
    predict_model = vgg16(pretrained=False)
    # 修改网络结构
    predict_model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                #nn.Linear(4096, 4096),
                #nn.ReLU(True),
                #nn.Dropout(),
                nn.Linear(4096, 2842),
            )
    # 将模型加载到CPU
    predict_model.load_state_dict(torch.load(opt.load_model_path,map_location=lambda  storage,loc:storage))
    # 模型改为预测模式
    predict_model.eval()
    # 如果GPU可用，加载到GPU
    if opt.use_gpu:
        predict_model.cuda()
    # 测试集照片地址
    test_img_dir = opt.test_img_dir
    img_dir = glob.glob(os.path.join(test_img_dir, '*.jpg'))
    # 遍历所有bbox图片
    for dir in img_dir:
        image = cv2.imread(dir)
        image_name = dir.split('/')[-1].replace('jpg', '')
        # result中内容为  左上角坐标、右下角坐标、类别名、输入的测试图地址、预测类别的可能性
        result = predict_result(predict_model, dir)
        for left_up, right_bottom, class_name, _, prob in result:
            # 将预测框添加到测试图片中
            cv2.rectangle(image, left_up, right_bottom, (0, 255, 0), 2)
            # 预测框的左上角写入 所属类别
            cv2.putText(image, class_name, left_up, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            print(image_name+" "+str(prob))
        # 将测试结果写入
        cv2.imwrite(opt.result_img_dir+'/'+image_name+'.jpg',image)




# 主函数
if __name__ == '__main__':

    # 命令行工具
    import fire
    fire.Fire()
    # train()
    predict()



