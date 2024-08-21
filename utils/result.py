import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def plot_data(training_data, out_dir):
    train_y = [
        training_data[i]["Train"]["Loss"] for i in range(len(training_data))
    ]
    valid_y = [
        training_data[i]["Valid"]["Loss"] for i in range(len(training_data))
    ]

    x = [i + 1 for i in range(len(train_y))]

    plt.figure(figsize=(18, 12))
    plt.title("Loss comparison", size=15, color="red")
    plt.grid()

    plt.plot(x, train_y, label="Train")
    plt.plot(x, valid_y, label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    plt.savefig(f"{out_dir}/Loss_curve.png")


# 真の値と予測値との差の可視化
def torch_compare_preds(y_eval, y_pred, out_dir):
    plt.figure(figsize=(10, 5))
    plt.grid()

    plt.plot(y_eval, label="correct")
    plt.plot(y_pred, label="prediction")

    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    plt.savefig(f"{out_dir}/corr_pred_diff.png")


# 結果の出力
def result_print(model, std, df, X, logger, mode):
    y_pred = model(X).detach().numpy()
    y_pred = std.inverse_transform(y_pred)
    y = std.inverse_transform(df["observed"].to_frame())

    logger.info(f"{mode} MSE: {np.mean((y_pred - y) ** 2)}")
    logger.info(f"r^2 {mode} data: {r2_score(y_pred, y)}")

    return logger
