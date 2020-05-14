import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


def show_some_digits(images, targets, sample_size=24, title_text='Digit {}' ):
    '''
    Visualize random digits in a grid plot
    images - array of flatten gidigs [:,784]
    targets - final labels
    '''
    nsamples=sample_size
    rand_idx = np.random.choice(images.shape[0], nsamples)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))


    img = plt.figure(1, figsize=(10, 8), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples/6.0), 6, index + 1)
        plt.axis('off')
        #each image is flat, we have to reshape to 2D array 28x28-784
        plt.imshow(image.reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(title_text.format(label))
    
    

def PCA(X, K):

    N,M = X.shape   
    # Center the data (subtract mean column values)
    Xc = X - np.ones((N,1))*X.mean(0)
    # PCA by computing SVD of Y
    U,S,V = np.linalg.svd(Xc,full_matrices=False)
    V = V.T
    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum() 
    # Project data onto principal component space
    Z = Xc @ V

    return Z[:,range(K)]

mnist = fetch_openml('mnist_784', version=1)
images = mnist.data[:20000,:]
targets = mnist.target[:20000]

X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.3)
classNames = np.unique(y_train)
#n_classes = len(classNames)

X_train_pca = PCA(X_train, 20)
X_test_pca = PCA(X_test, 20)

#clf=LinearSVC(penalty='l2', loss='squared_hinge', random_state=0, max_iter=10e4)
#clf = sk.svm.SVC()
param_grid = {'C': np.logspace(start = -15, stop = 1000, base = 1.02),
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf'), param_grid = param_grid)

clf.fit(X_train_pca,y_train)
print(clf.best_estimator_)
y_predict = clf.predict(X_test_pca)

print("training score:", clf.score(X_train_pca, y_train))
print("test score:", clf.score(X_test_pca , y_test))

print(classification_report(y_test, y_predict, target_names=classNames))
print(confusion_matrix(y_test, y_predict, labels=classNames))

show_some_digits(X_test, y_predict, 12)

# now you can save it to a file
joblib.dump(clf, 'Classifier.pkl') 

# and later you can load it
#clf = joblib.load('Classifier.pkl')

plt.show()

    
