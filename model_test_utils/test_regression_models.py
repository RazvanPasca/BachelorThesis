import numpy as np
from matplotlib import pyplot as plt


def test_regression_models(model, nr_samples, params):
    labels = ["Condition 0", "Condition 1", "Condition 2"]

    gen = params.dataset.validation_sample_generator(nr_samples, return_address=True)
    X_train, Y_train, addresses = next(gen)
    plot_regression_act_pred(X_train, Y_train, addresses, labels, model, "Train")

    gen = params.dataset.train_sample_generator(nr_samples, return_address=True)
    X_val, Y_val, addresses = next(gen)
    plot_regression_act_pred(X_val, Y_val, addresses, labels, model, "Test")


def plot_regression_act_pred(X_val, Y_val, addresses, labels, model, source):
    Y_pred = model.predict(X_val)
    abs_dif = np.abs(Y_pred - Y_val)

    for condition in range(3):
        Y_val_list = []
        Y_pred_list = []
        for i, address in enumerate(addresses):
            if address.condition == condition:
                Y_val_list.append(Y_val[i])
                Y_pred_list.append(Y_pred[i])

        plt.scatter(Y_val_list, Y_pred_list, label=labels[condition])
    plt.legend()
    plt.title("{} prediction MAE:{:.4}, prediction error std:{:.4}".format(source, np.mean(abs_dif), np.std(abs_dif)),
              fontsize=15)
    plt.xlabel("Actual values", fontsize=15)
    plt.ylabel("Predicted values", fontsize=15)
    plt.show()
