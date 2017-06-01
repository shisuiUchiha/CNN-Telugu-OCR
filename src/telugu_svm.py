import numpy as np
from PIL import Image, ImageFilter
import glob
from skimage.morphology import square,dilation
from skimage.transform import rescale
from skimage import io
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
from sklearn import svm


X = []
y =[]
# Restrict maximum width/height to 128 due to memory constraints
size = 32
max_size = 64
char_dir = '../data/64characters/'
for imagefile in glob.glob(char_dir + '*.tiff'):
	im = rescale(dilation(np.invert(io.imread(imagefile)),square(3)),float(size)/max_size)
	image_name = imagefile[-11:-8]
	y.append(int(image_name))
	X.append(im.flatten()) 		

print "done reading"
t = int(round(len(X)*0.8))		
X = np.array(X)
y = np.array(y)

X_train = X[0:t]
y_train = y[0:t]

X_test = X[t:]
y_test = y[t:]


# model=svm.SVC(C=1.0,kernel='linear',gamma=0.1)
model = svm.SVC()
model.fit(X_train,y_train)
print "Score:", model.score(X_test,y_test)

