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
from sklearn.cross_validation import train_test_split as split

dat_train, dat_test, tgt_train, tgt_test = split(iris.data, iris.target, train_size=0.8)
print('----------------')
print(dat_train)
print('LEN of dat_train :', len(dat_train))
print('----------------')
print(dat_test)
print('LEN of dat_test :', len(dat_test))
print('----------------')
print(tgt_train)
print('LEN of tgt_train :', len(tgt_train))
print('----------------')
print(tgt_test)
print('LEN of tgt_test :', len(tgt_test))

#-------------------------------
import keras
from keras.layers import Dense, Activation

# モデルを定義　(Sequential（系列）モデルは層を積み重ねたものです．)
model = keras.models.Sequential()
# 入力=4、隠れ層=32
model.add(Dense(units=32, input_dim=4))
# 隠れ層の活性化関数を設定
# --- 活性化関数 ---
# -1 <= val <= 1
#   ソフトサイン（softsign）
#   ハイパボリックタンジェント（tanh）
# 0 <= val <= 1
#   シグモイド関数（sigmoid）
#   ハードシグモイド（hard_sigmoid） 折れ線
# 0 <= val
#   ソフトプラス（softplus）　
#   ランプ関数（relu） 折れ線
# その他
#  線形関数（linear）　係数をかけてバイアスを加える
#　　ソフトマックス（softmax） 層から出力されたすべての値に指数関数をかけ正にし）、その和を1に正規化する。確率として判断できる。
model.add(Activation('relu'))

# 出力=3
model.add(Dense(units=3))
# 出力層の活性化関数を設定
model.add(Activation('softmax'))

# 目的関数（loss）
#  平均二乗誤差（mse：差の2乗の和）
#  平均絶対誤差（msa：差の絶対値の和）
#  平均絶対誤差率（mspa：差を正解の値で割った値（誤差率）の絶対値の和）
#  対数平均二乗誤差（msle：「1を加えた値の対数」の差の2乗の和）
#  ヒンジ損失の和（hinge）
#  ヒンジ損失の二乗の和（squared_hinge）
#  2クラス分類時の交差エントロピー（binary_crossentropy）
#  Nクラス分類時の交差エントロピー（categorical_crossentropy）
#  スパースなNクラス分類交差エントロピー（sparse_categorical_crossentropy）
#  KLダイバージェンス（kld）
#  poisson
#  コサイン類似度を負にしたもの（cosine_proximity）
# 最適化手法（optimizer）
#  optimizer='sgd' : Stochastic Gradient Descent : 確率的勾配降下法
#  sgd, rsmprop, adagrad, adadelta, adam, adamax, nadam
# 評価指数
#  metrics=['accuracy'] : テストデータに対してAccuracy（正答率）を計算。
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#-------------------------------
# modelの学習開始
# 150x0.8=120個データの学習を100回繰り返す(epochs=100)。
history = model.fit(dat_train, tgt_train, epochs=100)

# 精度の履歴をプロット
import matplotlib.pyplot as plt
plt.plot(history.history['acc'],"o-",label="accuracy")
plt.plot(history.history['loss'],"o-",label="loss")
plt.title('model accuracy and loss rate')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc="best")
plt.show()

#-------------------------------
# testデータを用いてmodelを評価
score = model.evaluate(dat_test, tgt_test, batch_size=1)
print('----------------',)
print('test loss =', score[0] )
print('test accuracy =', score[1] )

#-------------------------------
# 新しいデータを、modelで分類
# がくの長さ - sepal length in cm
# がくの幅 - sepal width in cm
# 花びらの長さ - petal length in cm
# 花びらの幅 - petal width in cm
import numpy as np
nDat = np.array([[3.5, 3.5, 3.5, 3.1]])
result = model.predict(nDat)
print('----------------',)
print("result :", result)

# 一番確率が高いラベルを表示
print('----------------')
print("index of the maximum value of the result :",  result.argmax())

# モデルを可視化
from keras.utils import plot_model
plot_model(model, to_file='b01-model.png', show_shapes=True, show_layer_names=True)
