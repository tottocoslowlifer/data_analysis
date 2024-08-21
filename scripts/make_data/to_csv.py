import csv
import os


def get_file_data(dir_path) -> list:
    filenames = sorted(os.listdir(dir_path))
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
    return filenames


def txt_csv_converter(filename, writer):
    with open(filename, encoding="shift-jis") as f:
        file = f.readlines()
        for row in file:
            if row[0:4] != "2011":
                continue
            read = []
            read.append(row[0:8])
            read.append(row[9:15])

            if row[17:24] == "9999.99":
                read.append(None)
            else:
                read.append(float(row[17:24]))

            if row[26:33] == "9999.99":
                read.append(None)
            else:
                read.append(float(row[26:33]))

            if row[35] == "9":
                read.append(None)
            else:
                read.append(float(row[35:42]))

            writer.writerow(read)


def main():
    filename_path = "../../data/tsunami/NOWPHAS_Tsunami_data"
    filenames = get_file_data(filename_path+"/raw")
    csv_path = filename_path + "/csv"

    if os.path.isdir(csv_path):
        pass
    else:
        os.mkdir(csv_path)

    for filename in filenames:
        csv_file = csv_path + "/" + filename[:11] + ".csv"
        with open(csv_file, mode="w", newline="") as fw:
            writer = csv.writer(fw, delimiter=",")
            txt_csv_converter(filename_path+"/raw/"+filename, writer)


if __name__ == '__main__':
    main()
