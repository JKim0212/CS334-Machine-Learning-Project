import pandas as pd
import argparse
import pdb
import numpy as np
import sklearn.preprocessing as prepross
import sklearn.model_selection as ms
import seaborn as sb
import matplotlib.pyplot as plt

def feature_selection(data):
  corr = data.corr(method="pearson")
  sb.set()
  sb.heatmap(corr, cmap="binary")
  plt.show()
  row = corr.iloc[0,:]
  sel_feat = []

  for i in range(1,len(row)):
     if np.absolute(row[i]) >= 0.1:
        sel_feat.append(row.index[i])
  return sel_feat
  



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset",
                        type=str,
                        help="dataset")
  args = parser.parse_args()
  data = pd.read_csv(args.dataset)
  features = feature_selection(data)
  y_data = data["Estimated owners"]
  x_data = data[features]
  
  scaler = prepross.StandardScaler(with_mean=True, with_std=True)
  x_data = scaler.fit_transform(x_data)
  x_data = pd.DataFrame(x_data, columns=features)
  xTrain, xTest, yTrain, yTest = ms.train_test_split(x_data, y_data, test_size=0.33, stratify=y_data)
  xTrain.to_csv("xTrain.csv", index=None)
  xTest.to_csv("xTest.csv", index=None)
  yTrain.to_csv("yTrain.csv", index=None)
  yTest.to_csv("yTest.csv",index=None)
  
if __name__ == "__main__":
    main()