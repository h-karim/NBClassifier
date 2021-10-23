import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.naive_bayes import GaussianNB as gaussian
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
import warnings
warnings.filterwarnings(
    'ignore', category=ConvergenceWarning)

df = pd.read_csv('drug200.csv')
print(df)
bp = pd.Categorical(df['BP'], ordered=True, categories=[
                    'LOW', 'NORMAL', 'HIGH'])
ch = pd.Categorical(df['Cholesterol'], ordered=True,
                    categories=['LOW', 'NORMAL', 'HIGH'])
sex = pd.get_dummies(df['Sex'])
print(sex)
df['BP'] = bp.codes
df['Cholesterol'] = ch.codes
del(df['Sex'])
df = df.join(sex)[['Age', 'M', 'F', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']]
print(df)
X = df
y = df.pop('Drug')
X_train, X_test, y_train, y_test = ms.train_test_split(X, y)
print(X_train, y_train)
print(X_test, y_test)
dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': [
    10, 100], 'min_samples_split': [2, 7, 15]}
f1 = make_scorer(f1_score, average='weighted')

mlp_params = {'activation': ['logistic', 'tanh', 'relu',
                             'identity'], 'hidden_layer_sizes': [(30, 50), (10, 10, 10)], 'solver': ['adam', 'sgd']}
top_mlp = GridSearchCV(estimator=MLPClassifier(),
                       param_grid=mlp_params, scoring=f1)
top_dt = GridSearchCV(estimator=DecisionTreeClassifier(),
                      param_grid=dt_params, scoring=f1)

base_mlp = MLPClassifier(hidden_layer_sizes=(
    100), activation='logistic', solver='sgd')
models = [('NB', gaussian()), ('Base-DT', DecisionTreeClassifier()), ('Top-DT', top_dt),
          ('PER', Perceptron()), ('Base-MLP', base_mlp), ('Top-MLP', top_mlp)]
sep = '--------------------------------\n'
labels = y.unique()
FILE = 'drugs-performance.txt'
with open(FILE, 'a') as f:
    for model in models:
        f.write(f'a) {model[0]}:\n')
        model[1].fit(X_train.values, y_train.values)
        if model[0] == 'Top-DT' or model[0] == 'Top-MLP':
            hyperparams = model[1].best_params_
            f.write(f'best hyperparameters: {hyperparams}\n')
        y_pred = model[1].predict(X_test.values)
        report = classification_report(
            y_test, y_pred, output_dict=True, target_names=labels, zero_division=0)
        confusion = confusion_matrix(y_test, y_pred)
        f.write('b) Confusion matrix: \n')
        f.write(f'{np.array2string(confusion)}\n')
        for drug in labels:
            precision = report[drug]['precision']
            recall = report[drug]['recall']
            f1 = report[drug]['f1-score']
            f.write(f'Class {drug}:\n')
            f.write(
                f'\tc) precision: {precision}\n\trecall: {recall}\n\tF1-measure: {f1}\n')
            f.write(sep)
        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        f.write(
            f'Accuracy: {accuracy} macro-average F1: {macro_f1} weighted-average F1: {weighted_f1}\n')
        f.write(sep)

accuracies = {}
macro_f1_total = {}
weighted_f1_total = {}

for i in range(0, 10):
    models = [('NB', gaussian()), ('Base-DT', DecisionTreeClassifier()), ('Top-DT', clone(top_dt)),
              ('PER', Perceptron()), ('Base-MLP', clone(base_mlp)), ('Top-MLP', clone(top_mlp))]
    for model in models:
        model[1].fit(X_train.values, y_train.values)
        y_pred = model[1].predict(X_test.values)
        report = classification_report(
            y_test, y_pred, output_dict=True, target_names=labels, zero_division=0)
        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        if model[0] not in accuracies:
            accuracies[model[0]] = []
            macro_f1_total[model[0]] = []
            weighted_f1_total[model[0]] = []

        accuracies[model[0]].append(accuracy)
        macro_f1_total[model[0]].append(macro_f1)
        weighted_f1_total[model[0]].append(weighted_f1)
print(accuracies, macro_f1_total, weighted_f1_total)
with open(FILE, 'a') as f:
    f.write('Step 8 stats:\n')
    for model in accuracies:
        accuracies[model]
        acc_avg = np.average(accuracies[model])
        m_f1_avg = np.average(macro_f1_total[model])
        w_f1_avg = np.average(weighted_f1_total[model])
        acc_std = np.std(accuracies[model])
        m_f1_std = np.std(macro_f1_total[model])
        w_f1_std = np.std(weighted_f1_total[model])
        f.write(
            f'\t{model}: \n\t\t\t\t\taccuracy\t macro-F1 avg\t weighted F1\n')
        f.write(f'avg\t\t\t\t{acc_avg}\t {m_f1_avg}\t {w_f1_avg}\n')
        f.write(f'std dev.\t{acc_std}\t {m_f1_std}\t {w_f1_std}\n')
        f.write('----\n')
    f.write(sep)
