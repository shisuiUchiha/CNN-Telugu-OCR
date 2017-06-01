# paper-implementation, zoneing
# keras-characters
import numpy as np
import glob
from skimage.morphology import square,dilation
from skimage.transform import rescale
from skimage import io
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
from sklearn import svm, linear_model, neighbors
from sklearn.naive_bayes import GaussianNB

X = []
y =[]
# Restrict maximum width/height to 128 due to memory constraints
count = 0
max_size = 64
char_dir = '../data/64characters/'
for imagefile in glob.glob(char_dir + '*.tiff'):
	im  = io.imread(imagefile)
	im = dilation(np.invert(im),square(3))
	features =[0]*64
	# Image consists of 64 8x8 zones. Features are sum of intensities in zone 
	for i in xrange(8):
		for j in xrange(8):
			features[i+8*j] = np.sum(im[8*i:8 + 8*i,8*j:8+8*j])
	image_name = imagefile[-11:-8]
	y.append(int(image_name))
	X.append(features)
	# count+=1
	# if count >12000:
	# 	break 		

print "done reading"
t = int(round(len(X)*0.8))		
X = np.array(X)
y = np.array(y)
print X.shape, y.shape

Z = np.c_[X,y]
# Randomly Shuffle data
np.random.shuffle(Z)
X = Z[:,:-1]
y = Z[:,-1]

X_train = X[0:t]
y_train = y[0:t]

X_test = X[t:]
y_test = y[t:]
# Naive Bayes Classifier
model=GaussianNB()
model.fit(X_train,y_train)
print "Score NB:", clf.score(X_test,y_test)
# 1 Nearest Neighbor Classifier
clf = neighbors.KNeighborsClassifier(1)
clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
print "Score KNN:", clf.score(X_test,y_test)
# SVM classifier
clf  = svm.SVC()
clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
print "Score SVM:", clf.score(X_test,y_test)