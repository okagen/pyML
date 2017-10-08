# -*- coding: utf-8 -*-
"""
Convert photo data to numpy data.
"""

import numpy as np
from PIL import Image #Pythonの画像処理ライブラリ
import glob, random

# 指定したフォルダ内の画像をnumpy形式のデータに変換＋ラベルのリストを生成
def glob_imgs(imgFolderPath, imgExt, imgCount, imgW, imgH, classes, label):
    
    # ファイルリストを取得し順番をシャッフル
    files = glob.glob(imgFolderPath + "/*." + imgExt)
    random.shuffle(files)
    dat = []
    lab = []
    for i, f in enumerate(files):
        if i >= imgCount: break
        img = Image.open(f)
        # RGBに変換
        img = img.convert("RGB")
        # サイズを調整
        img = img.resize((imgW, imgH))
        
        # 回転角度のリスト
        angList = [0, -5, 5]

        # 画像を回転させながらデータを生成
        for ang in angList:
            dat.append(conv_imgToNpArray(img, imgW, imgH, classes, ang))
            lab.append(label)
                
    return dat, lab

# imgをnumpy形式のデータに変換
def conv_imgToNpArray(img, imgW, imgH, classes, ang):
    if ang == 0:
        imgAng = img
    else:
        imgAng = img.rotate(ang)
    
    # imgをnumpy形式に変換
    data = np.asarray(imgAng)
    # 0<=val<=255 を 0.0<=val<1に変換
    data = data / 256
    # [列]x[行]x[色RGB]
    data = data.reshape(imgW, imgH, classes)

    return data    
    
# numpy形式のデータをデータセットとしてファイルに保存
def make_dataset(data, target, dataFilePath):
    data = np.array(data, dtype=np.float32)
    
    np.savez(dataFilePath, data=data, target=target)
    print("saved:" + dataFilePath)
        
if __name__ == '__main__':
    data = []
    target = []
    
    # 以下は、１画像分のデータ配列
    #     1       2     ...75
    #1  [[R G B],[R G B]...[R G B]]
    #2  [[R G B],[R G B]...[R G B]]
    # ...
    #75 [[R G B],[R G B]...[R G B]]
    dat_Cat, lab_Cat = glob_imgs("./cat", "jpg", 10,  75, 75, 3, 0)
    data.append(dat_Cat)
    target.append(lab_Cat)

    dat_Cow, lab_Cow = glob_imgs("./cow", "jpg", 10,  75, 75, 3, 1)
    data.append(dat_Cow)
    target.append(lab_Cow)

    dat_Eagle, lab_Eagle = glob_imgs("./eagle", "jpg", 10,  75, 75, 3, 2)
    data.append(dat_Eagle)
    target.append(lab_Eagle)

    # dataは 3種類 x [10枚 x [75行 x [75列 x [3要素（RGB)のデータ]]]]
    # targetは 3種類 x [10枚 x 1ラベルのデータ]
    # [[0 0 0 0 0 0 0 0 0 0], [1 1 1 1 1 1 1 1 1 1],[2 2 2 2 2 2 2 2 2 2 ]]
    make_dataset(data, target, "./b02-2-CatCowEagle.npz")
    
    