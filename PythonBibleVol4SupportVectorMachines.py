from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# load dataset from the scikit learn module
data = load_breast_cancer()
# Split out features and target
X = data.data
Y = data.target
# split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
# the random_state variable produces the same split of our dataset everytime, the number 30 is irrelevant
model = SVC(kernel='linear', C=3)
model.fit(X_train, Y_train)
# we choose a linear kernal (new dimension to increase complexity), we are allowing 3 misclassification (outliers)
accuracy = model.score(X_test, Y_test)
print(accuracy)
# compare against K-Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
# this model uses the same random data from earlier since we captured the specific iteration of the split
knn_accuracy = knn.score(X_test, Y_test)
print(knn_accuracy)
