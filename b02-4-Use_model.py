# -*- coding: utf-8 -*-
"""
Check the model
"""

from keras.models import Sequential
from keras.layers import Activation, Dense
from PIL import Image #Pythonの画像処理ライブラリ
import numpy as np

def build_model(data_size, classes, modelWeightsHDF5):
    model = Sequential()
    model.add(Dense(units=64, input_dim=(data_size)))
    model.add(Activation("relu"))
    model.add(Dense(units=classes))
    model.add(Activation("softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd", 
        metrics=["accuracy"])
    
    model.load_weights(modelWeightsHDF5)
    return model
    
def use_model(model, filePath, data_size, imgW, imgH):
    img = Image.open(filePath)
    img = img.convert("RGB")
    img = img.resize((imgW, imgH))
    data = np.asarray(img)
    data = data / 256
    data = data.reshape((-1, data_size))
    
    # modelを使って推測
    res = model.predict([data])[0]
    return res

if __name__ == "__main__":
    
    modelWeightsHDF5 = "./b02-3-CatCowEagle.hdf5"
    data_size = 75 * 75 * 3
    # modelを構築
    model = build_model(data_size, 3, modelWeightsHDF5)

    # modelを使って画像をチェック
    filePath_CAT = "./cat/32711702733.jpg"
    result = use_model(model, filePath_CAT, data_size, 75, 75)
    print("Imput img was CAT and result is:", result)
    labelIndex =  result.argmax()
    print("Most lileky label is:{0}, Accuracy is:{1}%".format( labelIndex, int(result[labelIndex]*100)))

    filePath_COW = "./cow/33772464025.jpg"
    result = use_model(model, filePath_COW, data_size, 75, 75)
    print("Imput img was COW and result is:", result)
    labelIndex =  result.argmax()
    print("Most lileky label is:{0}, Accuracy is:{1}%".format( labelIndex, int(result[labelIndex]*100)))

    filePath_EAGLE = "./eagle/15928548286.jpg"
    result = use_model(model, filePath_EAGLE, data_size, 75, 75)
    print("Imput img was EAGLE and result is:", result)
    labelIndex =  result.argmax()
    print("Most lileky label is:{0}, Accuracy is:{1}%".format( labelIndex, int(result[labelIndex]*100)))
    

    
    
    



