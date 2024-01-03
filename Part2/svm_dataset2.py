import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_error = np.std(data, ddof=1) / np.sqrt(len(data))  # ddof=1 for sample standard deviation
    ci = 1.96 * std_error  # 95% CI for normal distribution
    return mean, mean - ci, mean + ci




dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))


param_grid = {
    'C': [1, 10],
    'kernel': ['linear', 'rbf']
}

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

grid_search = GridSearchCV(SVC(), param_grid, cv=cv_strategy, scoring='accuracy', return_train_score=False)


all_results = []

for iteration in range(5):
    X_scaled = StandardScaler().fit_transform(dataset)
    grid_search.fit(X_scaled, labels)

    results = grid_search.cv_results_
    for params, mean_test_score in zip(results['params'], results['mean_test_score']):
        all_results.append({
            'iteration': iteration + 1,
            'parameters': params,
            'mean_test_score': mean_test_score
        })



print("Cross-Validation Results for all Parameter Combinations")
print("Iteration\t\t\t\tParameters\t\t\t\tMean Test Score (Accuracy)")

for result in all_results:
    if result['parameters']['kernel'] == 'rbf':
        print(f"\t{result['iteration']}\t\t\t{result['parameters']}\t\t\t\t{result['mean_test_score']:.4f}")
    else:
        print(f"\t{result['iteration']}\t\t\t{result['parameters']}\t\t\t{result['mean_test_score']:.4f}")

