from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
# load data
digits = load_digits()
# we have to scale down our data and standardize it
data = scale(digits.data)
# see size of data
print(digits.data.shape)
# 1797 rows 64 features
# visualize the data
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
plt.show()
quit()
# train the model (clustering is unsupervised learning so it trains without knowing the answers)
clf = KMeans(n_clusters=10, init='random',n_init=10)
clf.fit(data)
# we pick 10 as the number of clusters because we are dealing with digits 0 - 9
# we randomly place the centroids
# the init defines how many time the function will run to find the best clusters
# we can try and predict what cluster an input belongs to
#clf.predict([ENTER DATA HERE])
