import numpy as np

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten

from keras.layers.convolutional import Conv2D,MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

#import loader

from keras import models

from keras import layers

from keras.layers import SimpleRNN, Activation, Dense

#设置色彩通道值优先
K.set_image_data_format("channels_first")

#设置随机种子
np.random.seed(0)

#图像信息
channels = 1

height = 28

width = 28


#从MNIST数据集中读取数据和目标


(data_train,target_train),(data_test,target_test) = loader.load_mnist(train_image_path, train_label_path, test_image_path, test_label_path, normalize=False, one_hot=False)

#print(data_train)

#将训练集图像数据转换成特征  shape[0]只输出行数, shape[1]只输出列
data_train = data_train.reshape(data_train.shape[0],channels,height,width)

#将测试集图像数据转换成特征
data_test = data_test.reshape(data_test.shape[0],channels,height,width)

#将像素的强度值收缩到0和1之间
features_train = data_train.reshape(-1,28,28)/ 255
features_test = data_test.reshape(-1,28,28)/ 255

# 对目标进行one-hot编码
target_train = np_utils.to_categorical(target_train,num_classes=10)

target_test = np_utils.to_categorical(target_test,num_classes=10)

number_of_classes = target_test.shape[1]

# 启动神经网络
network = Sequential()

#添加嵌入层
#network.add(layers.Embedding(input_dim=number_of_classes,output_dim=128))

#添加一个128个神经元的长短期记忆网络
#network.add(layers.LSTM(units=128))

#添加使用sigmoid激活函数的全连接层
#network.add(layers.Dense(units=1,activation="sigmoid"))
# RNN cell
network.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, 28, 28),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=50,
    unroll=True,
))

# output layer
network.add(Dense(10))
network.add(Activation('softmax'))

# 编译神经网络
network.compile(loss = "categorical_crossentropy",
				  optimizer = "Adam",
				  metrics=["accuracy"])

# 训练神经网络
network.fit(features_train, #特征
			target_train,	#目标向量
			epochs=1,		#epoch的数量
			verbose=1,		#没有输出
			batch_size=1000,#每个批次的观察值数量
			validation_data=(features_test,target_test))#测试数据

loss, accuracy = network.evaluate(features_test, target_test, verbose=1)

print('loss:%.4f accuracy:%.4f' %(loss, accuracy))
