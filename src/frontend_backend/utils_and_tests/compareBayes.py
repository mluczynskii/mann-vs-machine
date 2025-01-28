import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from src.models.tokenization import tokenize  # Assuming the tokenize function is defined
from src.models.exception import NotTrainedError
from src.models.naive_bayes import NaiveBayesModel


def evaluate_model(model, X_train, X_test, y_train, y_test):
    X_train_tokenized = model.preprocess_set(X_train)
    X_test_tokenized = model.preprocess_set(X_test)

    model.train(X_train_tokenized, y_train)

    y_pred = model.classify(X_test_tokenized)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    return accuracy, precision, recall, f1, specificity, confusion_matrix(y_test, y_pred)


df1 = pd.read_excel("src/frontend_backend/dataset.xlsx")
df2 = pd.read_excel("src/frontend_backend/datasetBalanced.xlsx")

X1 = df1.iloc[:, 0].values  # Tweets
y1 = df1.iloc[:, 1].values  # Labels

X2 = df2.iloc[:, 0].values
y2 = df2.iloc[:, 1].values

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

nb_model = NaiveBayesModel()

metrics1 = evaluate_model(nb_model, X_train1, X_test1, y_train1, y_test1)
metrics2 = evaluate_model(nb_model, X_train2, X_test2, y_train2, y_test2)

labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
dataset1_metrics = [metrics1[0], metrics1[1], metrics1[2], metrics1[3], metrics1[4]]
dataset2_metrics = [metrics2[0], metrics2[1], metrics2[2], metrics2[3], metrics2[4]]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, dataset1_metrics, width, label='Unbalanced data')
rects2 = ax.bar(x + width/2, dataset2_metrics, width, label='Balanced data')


ax.set_title('Naive Bayes Performance Depending On Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

for rect in rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')


plt.tight_layout()
plt.show()


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')


    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()


plot_confusion_matrix(metrics1[5], 'Confusion Matrix - Unbalanced data')


plot_confusion_matrix(metrics2[5], 'Confusion Matrix - Balanced data')
