from math import e as E
from math import log
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from sklearn import model_selection as ms
from sklearn.naive_bayes import MultinomialNB as nbc
import matplotlib.pyplot as plt
import numpy as np

container = './bbc/'
dataset = datasets.load_files(container, encoding='latin1')
labels = dataset['target_names']
X, y = np.array(dataset['data']).T, np.array(
    dataset['target'], dtype=np.int16).T
M = np.column_stack((X, y))
vectorizer = CountVectorizer()
vocab = vectorizer.fit_transform(dataset['data'])
X_train, X_test, y_train, y_test = ms.train_test_split(
    vocab, y, test_size=.2, random_state=None)
nb_model = nbc()
nb_model2 = nbc()
nb_model4 = nbc(alpha=0.9)
nb_model3 = nbc(alpha=0.0001)
vocab_size = vocab.shape[1]
vocabarr = vocab.toarray()
total_words = vocabarr.sum(axis=0)
total_words_num = len(total_words)
freq1_total = len(total_words[total_words == 1])
freq1_per = freq1_total/vocab_size*100
#precision = report[type]["precision"]
#recall = report[type]["recall"]
#f1 = report[type]["f1-score"]
# plt.show()
models = [("MultinomialNB default values, try 1", nb_model), ("MultinomialNB default values, try 2",
                                                              nb_model2), ("MultinomialNB smoothing 0.0001", nb_model3), ("MultinomialNB smoothing 0.9", nb_model4)]
FILE = "bbc-performance.txt"
#predic_prob = nb_model2.predict_log_proba(X)
sep = "-----------\n"
with open(FILE, "a") as f:
    for model in models:
        model[1].fit(X_train, y_train.ravel())
        y_pred = model[1].predict(X_test)
        confusion = sk.metrics.confusion_matrix(y_test, y_pred)
        report = sk.metrics.classification_report(
            y_test, y_pred, output_dict=True, target_names=labels)
        accuracy = report['accuracy']
        mac_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        word2 = vectorizer.vocabulary_["sinners"]
        word1 = vectorizer.vocabulary_["killers"]
        w1_prob = total_words[word1]/total_words_num
        w2_prob = total_words[word2]/total_words_num
        w1_log_prob = log(w1_prob)
        w2_log_prob = log(w2_prob)
        f.write(model[0]+":\n")
        f.write("b) confusion_matrix: \n")
        f.writelines(np.array2string(confusion))
        f.write(
            f"\nd) accuracy: {accuracy} macro avg F1: {mac_f1}\tweighted avg F1: {weighted_f1}\n")
        f.write(f'f) vocab size: {vocab_size}\n')
        f.write(f'h) word tokens entire corpus: {total_words.sum()}\n')
        f.write(f'j) words with frequency 1: {freq1_total} ({freq1_per}%)\n')
        f.write(
            f'k) favourite words: sinner:{w1_log_prob}, killer:{w2_log_prob}\n')
        for i, type in enumerate(labels):
            class_data = M[M[:, 1] == str(i)][:, 0]
            precision = report[type]["precision"]
            recall = report[type]["recall"]
            f1 = report[type]["f1-score"]
            f.write(f'\tclass {type}:\n')
            f.write(
                f'\tc) precision: {precision}, recall: {recall}, f1-score: {f1}\n')
            prior = [E**log for log in model[1].class_log_prior_]
            f.write(f'\te) prior probability: {prior[i]}\n')
            type_vocab = vectorizer.transform(class_data).toarray().sum(axis=0)
            num_tokens = len(type_vocab)
            f.write(f'\tg) word tokens num: {num_tokens}\n')
            freq0_num = num_tokens - len(type_vocab[type_vocab != 0])
            freq0_per = freq0_num/num_tokens * 100
            f.write(
                f'\ti) words with frequency 0: {freq0_num} ({freq0_per}%)\n')
            f.write('\t'+sep)
        f.write(sep)
