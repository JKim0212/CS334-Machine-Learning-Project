import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

#importing data
X_train = pd.read_csv("xTrain.csv")
X_test = pd.read_csv("xTest.csv")
y_train = pd.read_csv("yTrain.csv")
y_test = pd.read_csv("yTest.csv")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
y_train = y_train.to_numpy().flatten()
y_val = y_val.to_numpy().flatten()

# KNN
from sklearn.neighbors import KNeighborsClassifier
param_grid = {'n_neighbors': range(1,251),
              'metric': ['euclidean', 'manhattan']}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

y_val_pred = grid_search.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

y_test_pred = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# #dTC
# from sklearn.tree import DecisionTreeClassifier
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 5, 10, 15, 20],
#     'min_samples_leaf': [1, 2, 4]
# }

# dt_classifier = DecisionTreeClassifier()

# grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)

# y_val_pred = grid_search.predict(X_val)
# val_accuracy = accuracy_score(y_val, y_val_pred)
# print("Validation Accuracy:", val_accuracy)

# y_test_pred = grid_search.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print("Test Accuracy:", test_accuracy)

# #Neural Network
# # from sklearn.neural_network import MLPClassifier
# # param_grid = {
# #     'hidden_layer_sizes': [(5,), (10,), (15,),(5, 5),(10, 5), (10, 10),(10, 5, 5)],
# #     'activation': ['logistic', 'tanh', 'relu'],
# #     'alpha': [0.0001, 0.001, 0.01],
# #     'learning_rate': ['constant', 'invscaling', 'adaptive']
# # }

# # mlp_classifier = MLPClassifier(max_iter=1000)  # Will need actual data to test
# # grid_search = GridSearchCV(mlp_classifier, param_grid, cv=5, scoring='accuracy')
# # grid_search.fit(X_train, y_train)

# # best_params = grid_search.best_params_
# # print("Best Hyperparameters:", best_params)

# # y_val_pred = grid_search.predict(X_val)
# # val_accuracy = accuracy_score(y_val, y_val_pred)
# # print("Validation Accuracy:", val_accuracy)

# # y_test_pred = grid_search.predict(X_test)
# # test_accuracy = accuracy_score(y_test, y_test_pred)
# # print("Test Accuracy:", test_accuracy)

# # Random Forest 
# from sklearn.ensemble import RandomForestClassifier
# param_grid = {
#     'n_estimators': [5, 10, 15, 20],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# rf_classifier = RandomForestClassifier(random_state=42)

# grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)

# y_val_pred = grid_search.predict(X_val)
# val_accuracy = accuracy_score(y_val, y_val_pred)
# print("Validation Accuracy:", val_accuracy)

# y_test_pred = grid_search.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print("Test Accuracy:", test_accuracy)