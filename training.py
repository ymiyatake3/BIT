# Reference:
# https://www.codingame.com/playgrounds/37409/handwritten-digit-recognition-using-scikit-learn


from mnist import MNIST
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

print("Loading dataset...")
mndata = MNIST("./data/")
images, labels = mndata.load_training()

clf = LinearSVC()

# Train on the first 10000 images:
train_x = images[:10000]
train_y = labels[:10000]

print("Train model")
clf.fit(train_x, train_y)

joblib.dump(clf, "digits_cls.pkl", compress=3)

print("Done!")

'''
# Test on the next 1000 images:
test_x = images[10000:11000]
expected = labels[10000:11000].tolist()

print("Compute predictions")
predicted = clf.predict(test_x)

print("Accuracy: ", accuracy_score(expected, predicted))
'''
