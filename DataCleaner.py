import pandas as pd
import pdb


# data = pd.read_csv("games_modified.csv")
# data = data.drop(["Developers", "Publishers"], axis=1)
# data = data[data.columns[0:18]]
# data = data.dropna(axis=0)
# data.loc[(data["Estimated owners"] == "0"), "Estimated owners"] = "0 - 20000"
# data.loc[(data["Estimated owners"] == "0 - 0"), "Estimated owners"] = "0 - 20000"
# data.to_csv("dataset.csv", index=None)


# one-hot encoding for estimated owners
# Estimated Owners
# 0-20000: 0
# 20000-50000: 1
# 50000-100000: 2
# 100000-200000: 3
# 200000-500000: 4
# 500000-1000000: 5
# 1000000-2000000: 6
# 2000000-5000000: 7
# 5000000-10000000: 8
# 10000000-20000000: 9 
# 20000000-50000000: 10
# 50000000-100000000: 11
# 10000000-200000000: 12

# data = pd.read_csv("dataset.csv")
# data.loc[(data["Estimated owners"] == "0 - 20000"), "Estimated owners"] = 0
# data.loc[(data["Estimated owners"] == "20000 - 50000"), "Estimated owners"] = 1
# data.loc[(data["Estimated owners"] == "50000 - 100000"), "Estimated owners"] = 2
# data.loc[(data["Estimated owners"] == "100000 - 200000"), "Estimated owners"] = 3
# data.loc[(data["Estimated owners"] == "200000 - 500000"), "Estimated owners"] = 4
# data.loc[(data["Estimated owners"] == "500000 - 1000000"), "Estimated owners"] = 5
# data.loc[(data["Estimated owners"] == "1000000 - 2000000"), "Estimated owners"] = 6
# data.loc[(data["Estimated owners"] == "2000000 - 5000000"), "Estimated owners"] = 7
# data.loc[(data["Estimated owners"] == "5000000 - 10000000"), "Estimated owners"] = 8
# data.loc[(data["Estimated owners"] == "10000000 - 20000000"), "Estimated owners"] = 9
# data.loc[(data["Estimated owners"] == "20000000 - 50000000"), "Estimated owners"] = 10
# data.loc[(data["Estimated owners"] == "50000000 - 100000000"), "Estimated owners"] = 11
# data.loc[(data["Estimated owners"] == "100000000 - 200000000"), "Estimated owners"] = 12
# data.to_csv("mod_dataset.csv", index=None)

# # Supported Languages
# # the number of supported languages
# # converting supported languages to numbers

# data = pd.read_csv("mod_dataset.csv")
# for index in range(data.shape[0]):
#   data.loc[index, "Supported languages"] = len(data.loc[index, "Supported languages"].split(","))
# data.to_csv("mod_dataset.csv", index=None)

# # Full Audio
# # number of supported audio languages
# # converting supported audio to numbers

# data = pd.read_csv("mod_dataset.csv")
# for index in range(data.shape[0]):
#   data.loc[index, "Full audio languages"] = len(data.loc[index, "Full audio languages"].split(",")) if data.loc[index, "Full audio languages"] != "[]" else 0
# data.to_csv("mod_dataset.csv", index=None)

# # Windows, Mac, Linux
# # whether the game supports the respective OS; 1 is support, 0 if not

# data = pd.read_csv("mod_dataset.csv")
# for index in range(data.shape[0]):
#   data.loc[index, "Windows"] = 1 if data.loc[index, "Windows"] == "TRUE" else 0
#   data.loc[index, "Mac"] = 1 if data.loc[index, "Mac"] == "TRUE" else 0
#   data.loc[index, "Linux"] = 1 if data.loc[index, "Linux"] == "TRUE" else 0
# data.to_csv("mod_dataset.csv", index=None)

# # Multiplayer
# # whether the game supports multiplayer or not
# # 1 if supported, 0 if not
# data = pd.read_csv("mod_dataset.csv")
# data = data.rename(columns={"Categories":"Multiplayer"})

# for index in range(data.shape[0]):
#   data.loc[index, "Multiplayer"] = 1 if "Multi-player" in data.loc[index, "Multiplayer"] else 0
# data.to_csv("mod_dataset.csv", index=None)

# # Genre
# # The number of genres the game belongs to
# # number greater or equal to 1
# data = pd.read_csv("mod_dataset.csv")

# for index in range(data.shape[0]):
#   data.loc[index, "Genres"] = len(data.loc[index, "Genres"].split(","))
# data.to_csv("mod_dataset.csv", index=None)

# # Price
# # Price of the game

# # DLC count
# # number of DLCs the game has

indices = []
data = pd.read_csv("mod_dataset.csv")
for i in range(data.shape[0]):

  try:
    int(data.loc[i, "Metacritic score"])
  except:
    indices.append(i)     
data = data.drop(indices, axis=0)

data.to_csv("mod1_dataset.csv", index=None)






