import pandas as pd
import numpy as np
import argparse
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import RandomForestClassifier as rf
def to_array(file):
  return pd.read_csv(file).to_numpy()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("xTrain",
                        help="filename of the updated training data")
  parser.add_argument("yTrain",
                        help="filename of the updated training data")
  parser.add_argument("xTest",
                        help="filename of the updated training data")
  parser.add_argument("yTest",
                        help="filename of the updated training data")
  args = parser.parse_args()
  xTrain = to_array(args.xTrain)
  yTrain = to_array(args.yTrain).flatten()
  xTest = to_array(args.xTest)
  yTest= to_array(args.yTest).flatten()

  '''
  Model Building
  '''
  knn_model = knn(metric="manhattan", n_neighbors=9)
  knn_model.fit(xTrain,yTrain)

  dt_model = dt(criterion='gini', max_depth=10, min_samples_leaf=2)
  dt_model.fit(xTrain,yTrain)

  rf_model = rf(max_depth=10,max_features='sqrt',min_samples_leaf=4,n_estimators=20)
  rf_model.fit(xTrain, yTrain)

  '''
  Model Accuracy
  '''

  
  knn_pred = knn_model.predict(xTest)
  print(f"KNN accuracy : {accuracy_score(yTest, knn_pred)}")
 
  
  dt_pred = dt_model.predict(xTest)
  print(f"dt accuracy : {accuracy_score(yTest, dt_pred)}")

  
  rf_pred = rf_model.predict(xTest)
  print(f"rf accuracy : {accuracy_score(yTest, rf_pred)}")
 

  '''
  Macro average AUC
  '''
  knn_pred_proba = knn_model.predict_proba(xTest)
  knn_auc = roc_auc_score(yTest, knn_pred_proba,average="macro", multi_class=("ovr"))
  print(f"KNN One-vs-rest AUC:{knn_auc}")

  dt_pred_proba = dt_model.predict_proba(xTest)
  dt_auc = roc_auc_score(yTest, dt_pred_proba,average="macro", multi_class=("ovr"))
  print(f"Decision Tree One-vs-rest AUC:{dt_auc}")

  rf_pred_proba = rf_model.predict_proba(xTest)
  rf_auc = roc_auc_score(yTest, rf_pred_proba,average="macro", multi_class=("ovr"))
  print(f"Random Forest One-vs-rest AUC:{rf_auc}")
  
if __name__ == "__main__":
  main()