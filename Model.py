import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open("./data.pickle", "rb"))

data_list = data_dict["data"]
labels = data_dict["labels"]

max_length = 84
uniform_data = [
    (
        item + [0] * (max_length - len(item))
        if len(item) < max_length
        else item[:max_length]
    )
    for item in data_list
]
data = np.asarray(uniform_data)
print(f"Shape of the data array: {data.shape}")


x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)


model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print("{}% of samples were classified correctly !".format(score * 100))

with open("model.p", "wb") as f:
    pickle.dump({"model": model}, f)
