# -*- coding: utf-8 -*-
"""
Train the model
"""
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np


def model_train(data, data_size, target, classes):
    model = Sequential()
    model.add(Dense(units=64, input_dim=(data_size)))
    model.add(Activation("relu"))
    model.add(Dense(units=classes))
    model.add(Activation("softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd", metrics=["accuracy"])
    history = model.fit(data, target, epochs=60)
    
    # 精度の履歴をプロット
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['loss'],"o-",label="loss")
    plt.title('model accuracy and loss rate')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="best")
    plt.show()
    
    return model

def model_eval(model, data_test, target_test):
    score = model.evaluate(data_test, target_test)
    print("loss=", score[0])
    print("accuracy", score[1])
    
if __name__ == "__main__":
    # データを読み込み
    dataNPZ = np.load("./b02-2-CatCowEagle.npz")
    
    # データ(説明変数) とラベル(目的変数)に分ける
    data = dataNPZ["data"]
    target = dataNPZ["target"]

    # dataとtargetの行列を整形
    # dataは [3種類 x [10枚 x [75行 x [75列 x [3要素（RGB)のデータ]]]]]
    # 以下は1枚分のデータの配列
    #     1       2     ...75
    #1  [[R G B],[R G B]...[R G B]]
    #2  [[R G B],[R G B]...[R G B]]
    # ...
    #75 [[R G B],[R G B]...[R G B]]
    data_size = 75 * 75 * 3
    # dataはを 3種類 x 10枚 x [75行 x 75列 x 3要素（RGB)のデータ]に変換
    # 列に-1 を指定すると適切な値が自動的に設定される。30行、75x75x3列にデータになる。
    data = np.reshape(data, (-1, data_size))

    # targetは 3種類 x [10枚 x 1ラベルのデータ]
    # [[0 0 0 0 0 0 0 0 0 0], [1 1 1 1 1 1 1 1 1 1],[2 2 2 2 2 2 2 2 2 2 ]]
    target_size = 1
    # targetを [3種類 x 10枚 x [1ラベルのデータ]]に変換。
    # 列に-1 を指定すると適切な値が自動的に設定される。30行、1列にデータになる。
    target = np.reshape(target, (-1, target_size))
    
    # 75%をトレーニング用のデータに分割する。
    data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=0.75)

    # Cat Cow Eagleの3クラス分類
    target_classes = 3
    
    # トレーンングしたモデルを作成
    model = model_train(data_train, data_size, target_train, target_classes)
    
    # modelの評価
    model_eval(model, data_test, target_test)

    # モデルを可視化
    from keras.utils import plot_model
    plot_model(model, to_file='b02-3-model.png', show_shapes=True, show_layer_names=True)
    
    # 学習済みモデルの「重み」を保存
    model.save_weights("./b02-3-CatCowEagle.hdf5")
    

