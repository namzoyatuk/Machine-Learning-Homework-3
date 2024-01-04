import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data))
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

# cross-validation for 5 times
for iteration in range(5):
    # preprocess the data
    X_scaled = StandardScaler().fit_transform(dataset)
    grid_search.fit(X_scaled, labels)

    results = grid_search.cv_results_
    for params, mean_test_score in zip(results['params'], results['mean_test_score']):
        all_results.append({
            'iteration': iteration + 1,
            'parameters': params,
            'mean_test_score': mean_test_score
        })



# printing the results
print("Cross-Validation Results for all Parameter Combinations")
print("Iteration\t\t\t\tParameters\t\t\t\tMean Test Score (Accuracy)")

for result in all_results:
    if result['parameters']['kernel'] == 'rbf':
        print(f"\t{result['iteration']}\t\t\t{result['parameters']}\t\t\t\t{result['mean_test_score']:.4f}")
    else:
        print(f"\t{result['iteration']}\t\t\t{result['parameters']}\t\t\t{result['mean_test_score']:.4f}")

summary_results = {}
for result in all_results:
    params_str = str(result['parameters'])
    if params_str not in summary_results:
        summary_results[params_str] = []
    summary_results[params_str].append(result['mean_test_score'])

# print summary results with 95% CI
print("\nSummary of Results with 95% Confidence Interval:")
print("Parameters\t\t\tMean Test Score (Accuracy)\t95% Confidence Interval")
for params_str, scores in summary_results.items():
    mean_score, lower_ci, upper_ci = calculate_confidence_interval(scores)
    print(f"{params_str}\t{mean_score:.4f}\t\t({lower_ci:.4f}, {upper_ci:.4f})")