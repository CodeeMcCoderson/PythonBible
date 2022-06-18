import pickle
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
# Split out features and target
X = data.data
Y = data.target
# split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
# the random_state variable produces the same split of our dataset everytime, the number 30 is irrelevant
model = SVC(kernel='linear', C=3)
model.fit(X_train, Y_train)
# we then save our model to a specified file
with open('model.pickle', 'wb') as file: pickle.dump(model, file)
# we can open the model in any script and use
with open('model.pickle', 'rb') as file: model = pickle.load(file)
#model.predict([...])

# NOW LETS TRAIN MODEL TO BE MOST ACURATE

best_accuracy = 0

for x in range(2500):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    model = SVC(kernel='linear', C=3)
    model.fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print('Best accuracy: ', accuracy)
        with open('model.pickle', 'wb') as file:
            pickle.dump(model, file)

print('end')
