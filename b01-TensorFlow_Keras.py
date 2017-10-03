# -*- coding: utf-8 -*-
"""
Study TensorFlow and Keras.
"""
from sklearn import datasets

#-------------------------------
# アヤメのデータを取得
iris = datasets.load_iris()
print('----------------')
print(iris.DESCR)

#-------------------------------
# アヤメのがく、花びらのデータ
# がくの長さ - sepal length in cm
# がくの幅 - sepal width in cm
# 花びらの長さ - petal length in cm
# 花びらの幅 - petal width in cm
print('----------------')
print(iris.data)
print('LEN of iris.data :', len(iris.data))

#-------------------------------
# アヤメの種類を表す
# 0 - Iris-Setosa
# 1 - Iris-Versicolour
# 2 - Iris-Virginica
print('----------------')
print(iris.target)
print('LEN of iris.target :', len(iris.target))

#-------------------------------
# iris.dataとiris.targetの対応を維持したままシャッフルし、80%を学習データに、残りの20％をテストデータに振り分ける。
from sklearnmodel_selection import train_test_split as split
dat_train, dat_test, tgt_train, tgt_test = split(iris.data, iris.target, train_size=0.8)
print('----------------')
print(dat_train)
print('LEN of dat_train :', len(dat_train))
print(dat_test)
print('LEN of dat_test :', len(dat_test))
print(tgt_train)
print('LEN of tgt_train :', len(tgt_train))
print(tgt_test)
print('LEN of tgt_test :', len(tgt_test))

#-------------------------------
import keras
from keras.layers import Dense, Activation

# モデルを定義
model = keras.models.Swquential()
# 入力=4、隠れ層=32
model.add(Dense(units=32, input_dim=4))
# 活性化関数を設定
model.add(Activation('relu'))
# 出力=3
model.add(Dense(units=3))
# 活性化関数を設定
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#-------------------------------
# modelの学習開始
# 120個データの学習を100回繰り返す(epochs=100)。
model.fit(dat_train, tgt_train, epochs=100)

#-------------------------------
# testデータを用いてmodelを評価
score = model.evaluate(dat_test, tgt_test, batch_size=1)
print('accuracy =', score[1] )


#-------------------------------
# 新しいデータを、modelで分類
import numpy as np
nDat = np.array([[5.1, 3.5, 1.4, 0.2]])
result = model.predict(nDat)
result

# 一番確率が高いラベルを表示
print('----------------')
result.argmax()
