# data_analysis

## 概要
東日本大震災の,各地での潮位データを用いて,宮城県中部沖での現在の観測潮位を予測する.

## フォルダの説明
### data
使用するデータを格納する.

### notebook
分析に使用したnotebookを格納する.

### scripts
データの整形に使用するファイル.

## 手順
1. 以下のコマンドを入力する.
~~~
pip3 install -r requirements.txt
~~~

2. `data/tsunami/NOWPHAS_Tsunami_data/raw`内の各テキストファイルをダウンロードする.

3. `scripts`上で以下のコマンドを順に入力する.
~~~
python3 to_csv.py
~~~
~~~
python3 df_create.py
~~~
~~~
python3 cmp_df.py
~~~

4. `notebook`内の各ファイルをダウンロードし,実行する.