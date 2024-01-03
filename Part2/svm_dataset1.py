import pickle
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))

def plot_decision_boundaries(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)



# SVM configurations
configs = [
    {'C': 1, 'kernel': 'linear'},
    {'C': 1, 'kernel': 'rbf'},
    {'C': 100, 'kernel': 'linear'},
    {'C': 100, 'kernel': 'rbf'}
]

plt.figure(figsize=(12, 10))

for i, config in enumerate(configs, 1):
    # train an SVM model for each config
    clf = svm.SVC(C=config['C'], kernel=config['kernel'])
    clf.fit(dataset,labels)

    # plot the boundaries
    plt.subplot(2, 2, i)
    plot_title = f"SVM with C={config['C']} and kernel='{config['kernel']}'"
    plot_decision_boundaries(dataset, labels, clf, plot_title)

plt.tight_layout()
plt.savefig('svm_dataset1.png')
plt.show()