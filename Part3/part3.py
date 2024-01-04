import numpy as np
from DataLoader import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, make_scorer


def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data))
    ci = 1.96 * std_error  # 95% CI for normal distribution
    return mean, mean - ci, mean + ci


data_path = "../data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

"""
Implementing the nested cross validation
"""

models = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

params = [
    {'n_neighbors': [3, 5, 11], 'metric': ['euclidean', 'manhattan']},
    {'C': [1, 10], 'kernel': ['linear', 'rbf']},
    {'max_depth': [5, 10, 20], 'min_samples_leaf': [1, 2, 4]},
    {'n_estimators': [10, 100], 'max_depth': [5, 10]}
]

# Cross-validation strategies
outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

# Scoring function
scoring = make_scorer(accuracy_score)

# Initialize variables to keep track of results
results = {model_name: [] for model_name in ['KNN', 'SVM', 'DecisionTree', 'RandomForest']}
best_params = {model_name: [] for model_name in ['KNN', 'SVM', 'DecisionTree', 'RandomForest']}
f1_scores = {model_name: [] for model_name in ['KNN', 'SVM', 'DecisionTree', 'RandomForest']}

model_names = ['KNN', 'SVM', 'DecisionTree', 'RandomForest']

normalized = MinMaxScaler(feature_range=(-1, 1)).fit_transform(dataset)

for train_idx, test_idx in outer_cv.split(normalized, labels):  # Outer loop
    X_train, X_test = normalized[train_idx], normalized[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    for model, param, name in zip(models, params, model_names):  # Inner loop
        inner_clf = GridSearchCV(estimator=model, param_grid=param, cv=inner_cv, scoring=scoring)
        inner_clf.fit(X_train, y_train)

        best_model = inner_clf.best_estimator_
        y_pred = best_model.predict(X_test)

        # Store the results
        results[name].append(inner_clf.cv_results_['mean_test_score'])
        best_params[name].append(best_model.score(X_test, y_test))
        f1_scores[name].append(f1_score(y_test, y_pred))

# Display the results
for model_name in model_names:
    print(f'\t{model_name}\t')

    for idx, configuration in enumerate(params[model_names.index(model_name)]):
        config_results = [result[idx] for result in results[model_name]]
        mean_acc, ci_low_acc, ci_high_acc = calculate_confidence_interval(config_results)

        print(f'Configuration {idx + 1}: {configuration}')
        print(f'Mean Accuracy: {mean_acc:.3f}, Confidence Interval: ({ci_low_acc}, {ci_high_acc})\n')

    overall_mean_acc, overall_ci_low_acc, overall_ci_high_acc = calculate_confidence_interval(best_params[model_name])
    overall_mean_f1, overall_ci_low_f1, overall_ci_high_f1 = calculate_confidence_interval(f1_scores[model_name])

    print(
        f'Overall {model_name} Accuracy: {overall_mean_acc:.3f}, Accuracy Confidence Interval: ({overall_ci_low_acc}, {overall_ci_high_acc}), \n'
        f'F1 Score: {overall_mean_f1:.3f}, F1 Score Confidence Interval: ({overall_ci_low_f1}, {overall_ci_high_f1})\n')