import pandas as pd


# メインデータフレームの作成
main_df = pd.read_csv(
    "../../data/tsunami/NOWPHAS_Tsunami_data/cmp_csv/cut_2011TET801G.csv",
    index_col=0
)
main_df = main_df.drop(["index", "deviation", "datetime"], axis=1)

# 過去の水位データの連結
main_df_times = [f"observed_{5*i}s" for i in range(1, 13)]
for i in range(len(main_df_times)):
    main_df[main_df_times[i]] = main_df["observed"].shift(i+1)
    main_df[main_df_times[i]] = main_df["observed"] - main_df[main_df_times[i]]

# 801Gより先に津波が到達する地点の水位データを結合
cmp_filenames = ["2011TET318T.csv", "2011TET317T.csv",
                 "2011TET315T.csv", "2011TET319T.csv",
                 "2011TET802G.csv", "2011TET804G.csv",
                 "2011TET803G.csv", "2011TET806G.csv"]

pre_list = []
pre_columns = []

for filename in cmp_filenames:
    pre_df = pd.read_csv(
        "../../data/tsunami/NOWPHAS_Tsunami_data/cmp_csv/cut_"+filename,
        index_col=0
    )
    pre_list.append(pre_df["observed"])
    pre_columns.append(f"observed+{filename[7:11]}")

for i in range(len(pre_columns)):
    main_df[pre_columns[i]] = pre_list[i]

# 欠損値を含むデータの削除
main_df_full = main_df.copy().dropna()
main_df_full = main_df_full.reset_index().drop("index", axis=1)

for col in main_df_times:
    main_df_full[col] = main_df_full["observed"] - main_df_full[col]

main_df_full.to_csv(
    "../../data/tsunami/NOWPHAS_Tsunami_data/full.csv", index=False
)
