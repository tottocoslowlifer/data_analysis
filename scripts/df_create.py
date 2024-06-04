import os
from datetime import datetime as dt

import pandas as pd


def get_file_data(dir_path) -> list:
    filenames = sorted(os.listdir(dir_path))
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
    return filenames


def df_create(filename) -> pd.DataFrame:
    df = pd.read_csv(filename,
                     names=["date", "time", "observed", "astoro", "deviation"])

    df["date"] = df["date"].astype(str)
    df["time"] = df["time"].astype(str)

    for i in range(len(df)):
        df.loc[i, "time"] = df.loc[i, "time"].zfill(6)
        s = df.loc[i, "date"] + df.loc[i, "time"]
        df.loc[i, "datetime"] = dt.strptime(s, "%Y%m%d%H%M%S")
    df = df.drop(["date", "time"], axis=1)

    start = df[df["datetime"] == dt(2011, 3, 11, 15, 00, 00)].index[0]
    end = df[df["datetime"] == dt(2011, 3, 11, 17, 00, 00)].index[0]

    cut_df = df.iloc[start:end].reset_index()
    return cut_df


def null_cnt(dir_path, filenames):
    null_list = []
    max_list = []
    df_list = []

    for filename in filenames:
        cut_df = df_create(dir_path+"/csv/"+filename)
        df_list.append(cut_df)

        null_list += [[filename, cut_df.isnull().any(axis=1).sum()]]
        max_list += [
            [filename, cut_df.loc[cut_df["observed"].idxmax(), "datetime"]]
        ]

    return null_list, max_list, df_list


def main():
    filename_path = "../data/tsunami/NOWPHAS_Tsunami_data"
    csv_filenames = get_file_data(filename_path+"/csv")
    csv_filenames.remove("cmp_csv")
    csv_filenames.remove("cut_csv")

    for file in csv_filenames:
        file_df = df_create(filename_path+"/csv/"+file)
        file_df.to_csv(filename_path+"/csv/cut_csv/"+f"cut_{file}")


if __name__ == '__main__':
    main()
