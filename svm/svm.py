import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from datasets.DATASET_PATHS import GABI_CAT_DATASET_PATH as CAT_DATASET_PATH
from datasets.DATASET_PATHS import GABI_MOUSEACH_DATASET_PATH as MOUSEACH_DATASET_PATH
from datasets.DATASET_PATHS import GABI_MOUSE_DATASET_PATH as MOUSE_DATASET_PATH
from datasets.LFPDataset import LFPDataset


def load_cat_dataset(path, label='condition'):
    dataset = LFPDataset(path, normalization='Zsc')

    x = []
    y = []
    for condition_id in range(0, dataset.number_of_conditions):
        for i in range(0, dataset.channels.shape[1] // dataset.trial_length):
            if dataset.stimulus_conditions[i] == condition_id + 1:
                trial = dataset.channels[:,
                        (i * dataset.trial_length):((i * dataset.trial_length) + dataset.trial_length)]
                if label == 'condition':
                    x.append(trial.reshape(-1))
                    y.append(condition_id)
                elif label == 'channel':
                    for i, c in enumerate(trial):
                        x.append(c)
                        y.append(i)
                else:
                    return None
    return x, y


def load_mouse_dataset(path, label='orientation', ignore_channels=False):
    dataset = LFPDataset(path, normalization='Zsc')
    x = []
    y = []
    for stimulus_condition in dataset.stimulus_conditions:
        index = int(stimulus_condition['Trial']) - 1
        events = [{'timestamp': dataset.event_timestamps[4 * index + i],
                   'code': dataset.event_codes[4 * index + i]} for i in range(4)]
        trial = dataset.channels[:, events[1]['timestamp']:(events[1]['timestamp'] + 2672)][:-1]
        # removed heart rate

        if label == 'orientation':
            if ignore_channels:
                for i, c in enumerate(trial):
                    x.append(c)
                    y.append((int(stimulus_condition['Condition number']) - 1) // 3)
            else:
                x.append(trial.reshape(-1))
                y.append((int(stimulus_condition['Condition number']) - 1) // 3)
        elif label == 'contrast':
            if ignore_channels:
                for i, c in enumerate(trial):
                    x.append(c)
                    y.append((int(stimulus_condition['Condition number']) - 1) % 3)
            else:
                x.append(trial.reshape(-1))
                y.append((int(stimulus_condition['Condition number']) - 1) % 3)
        elif label == 'condition':
            if ignore_channels:
                for i, c in enumerate(trial):
                    x.append(c)
                    y.append((int(stimulus_condition['Condition number']) - 1))
            else:
                x.append(trial.reshape(-1))
                y.append((int(stimulus_condition['Condition number']) - 1))
        elif label == 'channel':
            for i, c in enumerate(trial):
                x.append(c)
                y.append(i)
        else:
            return None
    return x, y


def compute_confusion_matrix(svc, x, y):
    y_pred = svc.predict(x)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))


if __name__ == '__main__':
    # x, y = load_cat_dataset(CAT_DATASET_PATH, label='condition')
    x, y = load_mouse_dataset(MOUSEACH_DATASET_PATH, label='contrast', ignore_channels=True)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    for i in range(0, 1):
        print("C={}".format(10 ** i))
        svc = SVC(kernel='linear', C=10 ** i)

        svc.fit(X_train, y_train)

        print("TRAIN CONFUSION MATRIX:")
        compute_confusion_matrix(svc, X_train, y_train)

        if not len(y_test) is 0:
            print("TEST CONFUSION MATRIX:")
            compute_confusion_matrix(svc, X_test, y_test)
