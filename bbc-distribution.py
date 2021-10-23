import numpy as np
import matplotlib.pyplot as plt
import os


def count_files(d):
    return len(os.listdir(d))


if __name__ == '__main__':
    DIR = ['business', 'entertainment', 'politics', 'sport', 'tech']
    PATH = './bbc/'
    counts = {}
    for d in DIR:
        counts[d] = count_files(PATH+d)
    print(counts)
    X = np.arange(1, len(counts)+1)
    labels = list(counts.keys())
    height = list(counts.values())
    title = 'Distribution of the BBC dataset'
    plt.title(title)
    fig = plt.figure()
    plt.bar(X, height=height, tick_label=labels)
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Distribution of document types")
    plt.savefig('bbc-distribution.pdf')
