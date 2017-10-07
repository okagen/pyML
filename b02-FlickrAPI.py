# -*- coding: utf-8 -*-
"""
Get some pictures from Flickr.
"""
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time

# API Keyの設定
key = ""
secret = ""

# 検索キーワード
keyword = "cat"

# 検索件数＝ダウンロード数
countPict = 500

# 画像保存先のディレクトリ 無ければ作る
savedir = "./" + keyword
if not os.path.exists(savedir):
    os.mkdir(savedir)

# Flickr写真を検索 結果をJSONで取得
flickr = FlickrAPI(key, secret, format="parsed-json")
res = flickr.photos.search(
    text = keyword,
    per_page = countPict,
    media = "photos",
    sort = "relevance",
    safe_search = 1,
    extras = "url_q, license")

# 検索結果を確認
photos = res["photos"]
pprint(photos)

# ダウンロード後の待機時間
wait_time = 1

# 画像をダウンロード
try:
    for i , photo in enumerate(photos["photo"]):
        url_q = photo["url_q"]
        filepath = savedir + "/" + photo["id"] + ".jpg"
        if os.path.exists(filepath): continue

        urlretrieve(url_q, filepath)
        print(str(i + 1) + ":download=", url_q)

        time.sleep(wait_time)
except:
    import traceback
    traceback.print_exc()
    

