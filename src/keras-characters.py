# keras-characters
import numpy as np
from PIL import Image, ImageFilter
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from skimage.morphology import square,dilation
from skimage.transform import rescale
from skimage import io
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
K.set_image_dim_ordering('th')


X = []
y =[]
# Restrict maximum width/height to 128 due to memory constraints
max_size = 64
size = 32
char_dir = '../data/64characters/'
for imagefile in glob.glob(char_dir + '*.tiff'):
	im = rescale(dilation(dilation(np.invert(io.imread(imagefile)),square(3)),square(3)),float(size)/max_size)
	image_name = imagefile[-11:-8]
	y.append(int(image_name))
	X.append(im.flatten()) 		

t = int(round(len(X)*0.8))		
X = np.array(X)
y = np.array(y)
Z = np.c_[X,y]
# Randomly Shuffle DAta
np.random.shuffle(Z)
X = Z[:,:-1]
y = Z[:,-1]
X = X.reshape(X.shape[0], 1, size, size).astype('float32')

X_train = X[0:t]
y_train = y[0:t]

X_test = X[t:]
y_test = y[t:]
# Convert to vectors, One hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
model = Sequential()
model.add(Conv2D(1, (7, 7), padding='SAME', input_shape=(1,size,size), activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), padding='SAME',  activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='SAME',  activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding='SAME',  activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))	
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Training with 25% Crossvalidation Split
model.fit(X_train, y_train, validation_split=0.25, epochs=32, batch_size=128)
model.save('teluguchar4.h5')
# Calculate Score for test data
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))