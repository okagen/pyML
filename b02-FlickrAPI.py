# -*- coding: utf-8 -*-
"""
Flickr
"""
from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# API Keyの設定
key = ""
secret = ""
 
# ダウンロード後の待機時間
wait_time = 1

# キーワードをチェックする
#if len(sys.argv) < 2:
#    print("python dl.py (keyword)")
#    sys.exit()

#keyword = sys.argv[1]
keyword = "cat"
savedir = "./" + keyword
if not os.path.exists(savedir):
    os.mkdir(savedir)

# Flickr APIで写真を検索
flickr = FlickrAPI(key, secret, format="parsed-json")
res = flickr.photos.search(
    text = keyword,
    per_page = 500,
    media = "photos",
    sort = "relevance",
    safe_search = 1,
    extras = "url_q, license")
pprint(res)

# 検索結果を確認
photos = res["photos"]
pprint(photos)
try:
    # 1つずつ画像をダウンロードする
    for i , photo in enumerate(photos["photo"]):
        url_q = photo["url_q"]
        filepath = savedir + "/" + photo["id"] + ".jpg"
        if os.path.exists(filepath): continue
        print(str(i + 1) + ":download=", url_q)
        urlretrieve(url_q, filepath)
        time.sleep(wait_time)
except:
    import traceback
    traceback.print_exc()
    

