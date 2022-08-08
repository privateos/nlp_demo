import torch

x = torch.randn((3, 4))
print('x = \n', x)
values, indices = torch.max(x, dim=1)
print('values = ', values)
print('indices = ', indices)


#2.参数解析
#1)argparse
#2)config
#3)set seed

#2.数据预处理
#1）构建词表/字表   dict={key:value} key表示词 value表示这个词的编号
#2）把train / valid / test文件转换为数字格式数据
#vocab_dict, train_dataset, dev_dataset, test_dataset

#3.模型搭建
#1)TextCNN

#4.训练模型
#1)训练的基本步骤
#2)每个一定的batch就查看验证集情况
#3)一定的正则化手段(早停:连续1000batch验证集数据没有提升，就停止训练)
#4）测试集数据弄出来