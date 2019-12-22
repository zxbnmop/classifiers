#加载库
from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.datasets import fetch_mldata

import numpy as np

#加载数据
mnist = fetch_mldata('MNIST original')

features, target = mnist['data'], mnist['target']

#标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#分割数据集
features_train,features_test,target_train,target_test = features_standardized[:60000],features_standardized[60000:],target[:60000],target[60000:]

#打乱数据
shuffle_index = np.random.permutation(60000)  # 随机排列一个序列，返回一个排列的序列。

features_train, target_train = features_train[shuffle_index], target_train[shuffle_index]
#创建决策树分类器对象
decisiontree = DecisionTreeClassifier(random_state=0)

#训练模型
model = decisiontree.fit(features_train,target_train)

#预测数据
predictions = model.predict(features_test)

#测试正确率
accuracy = metrics.accuracy_score(target_test,predictions)

print ('accuracy：%.2f%%'%(100*accuracy))
