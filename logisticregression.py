#加载库
from sklearn.linear_model import LogisticRegression

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_mldata

from sklearn import metrics

import numpy as np
#加载数据
#import loader
mnist = fetch_mldata('MNIST original')
#print(mnist)

features, target = mnist['data'], mnist['target']
#print('x的大小为；', features.shape, '\n','x的大小为；', target.shape)


#标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

features_train,features_test,target_train,target_test = features_standardized[:60000],features_standardized[60000:],target[:60000],target[60000:]
#print(features_standardized)

#打乱数据
shuffle_index = np.random.permutation(60000)  # 随机排列一个序列，返回一个排列的序列。

features_train, target_train = features_train[shuffle_index], target_train[shuffle_index]

#创建一对多逻辑回归对象
logistic_regression = LogisticRegression(random_state=0,multi_class='ovr')

#训练模型
model = logistic_regression.fit(features_train,target_train)

#预测数据
predictions = model.predict(features_test)

#测试正确率
accuracy = metrics.accuracy_score(target_test,predictions)

print ('accuracy：%.2f%%'%(100*accuracy))
