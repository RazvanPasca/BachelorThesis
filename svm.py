import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from datasets.DATASET_PATHS import GABI_CAT_DATASET_PATH as CAT_DATASET_PATH

from datasets.LFPDataset import LFPDataset

dataset = LFPDataset(CAT_DATASET_PATH, normalization='Zsc')

cond_trial_channel = []
for condition_id in range(0, dataset.number_of_conditions):
    condition = []
    for i in range(0, dataset.channels.shape[1] // dataset.trial_length):
        if dataset.stimulus_conditions[i] == condition_id + 1:
            trial = dataset.channels[:, (i * dataset.trial_length):((i * dataset.trial_length) + dataset.trial_length)]
            condition.append(trial)
    cond_trial_channel.append(np.array(condition))

cond_trial_channel = np.array(cond_trial_channel, dtype=np.float32)

cond_trial_channel = cond_trial_channel.reshape((3, 20, -1))
x = []
y = []
for i, movie in enumerate(cond_trial_channel):
    x.append(cond_trial_channel[i])
    y.append([i for _ in range(20)])

x = np.array(x).reshape((60, -1))
y = np.array(y).reshape(60)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)

svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

print("TEST CONF MATRIX:")
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("TRAIN CONF MATRIX:")
y_pred = svclassifier.predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(classification_report(y_train, y_pred))