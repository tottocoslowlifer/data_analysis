import os

import pandas as pd


def get_file_data(dir_path) -> list:
    filenames = sorted(os.listdir(dir_path))
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
    return filenames


def df_completion(pre_df: pd.DataFrame) -> pd.DataFrame:
    nan_index = []
    for i in range(len(pre_df)):
        if pre_df.loc[i].hasnans:
            nan_index.append(i)

    nan_full_df = pre_df.copy()
    for idx in nan_index:
        nan_full_df.loc[idx, "observed"] = float("nan")
    cmp_test_df = nan_full_df.copy()

    for i in range(len(nan_index)):
        if i == len(nan_index)-1 or nan_index[i]+1 == nan_index[i+1]:
            cmp_test_df.loc[nan_index[i], "observed"] = (
                cmp_test_df.loc[nan_index[i]-1, "observed"]
            )

        else:
            cmp_test_df.loc[nan_index[i], "observed"] = (
                cmp_test_df.loc[nan_index[i]+1, "observed"] +
                cmp_test_df.loc[nan_index[i]-1, "observed"]
            )/2

    # 合わせて潮位偏差も補完
    for i in range(len(cmp_test_df)):
        cmp_test_df.loc[i, "deviation"] = (cmp_test_df.loc[i, "observed"] -
                                           cmp_test_df.loc[i, "astoro"])

    return cmp_test_df


def main():
    cmp_filename_path = "../data/tsunami/NOWPHAS_Tsunami_data/cmp_csv"
    filename_path = "../data/tsunami/NOWPHAS_Tsunami_data/cut_csv"
    df_list = get_file_data(filename_path)

    if os.path.isdir(cmp_filename_path):
        pass
    else:
        os.mkdir(cmp_filename_path)
    
    for file in df_list:
        df = pd.read_csv(filename_path+"/"+file, index_col=0)
        df = df_completion(df)
        df.to_csv(cmp_filename_path+"/"+file)


if __name__ == '__main__':
    main()
