#import models.TextCNN as x
#config = x.Config('THUCNews', 'random')


#设置随机种子为一个固定值确保每次训练网络得到相同的结果


#build_dataset
#1.从训练集中构建一个字典:   key=字, value=这个字对应的编号---->build_vocab
#voc_dic={'源': 0, '珊': 1, '爱': 2, '我': 3, '是': 4}
#2.从train.txt/dev.txt/test.txt中获得数据--------->load_dataset
#train_data, dev_data, test_data =  = [
# ([3, 410, 2, 300, ], 0, 20),
# ([5, 210, 5, 200, ], 2, 32),
# ([1, 510, 70, 80, ], 6, 11),
# .....
# ]