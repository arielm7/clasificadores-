import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
%matplotlib inline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data() #Returns a tuple of Numpy arrays.

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))   #Adding layers at the model.
model.add(MaxPooling2D(pool_size=(2, 2)))          #Max pooling operation for temporal data
model.add(Dropout(0.25))						   #Ignore a fraction of inputs during training rate, avoid the overfitting: 0.25 Fraction to drop	
model.add(Flatten())							   #Flatten the input
model.add(Dense(128, activation='relu'))		   #Implement the activation function.
model.add(Dropout(0.5))							   #Ignore a fraction of inputs during training rate, avoid the overfitting: 0.25 Fraction to drop
model.add(Dense(num_classes, activation='softmax'))#Implement the activation function.

#Configure its learning process
model.compile(loss=keras.losses.categorical_crossentropy,#String (name of optimizer) or optimizer instance
              optimizer=keras.optimizers.Adadelta(),	 #String (name of objective function) or objective function
              metrics=['accuracy'])						 # List of metrics to be evaluated by the model during training and testing				         

#Trains the model for a given number of epochs (iterations on a dataset).
model.fit(x_train, y_train,         			  #Numpy array of training data , Numpy array of target (label) data
          batch_size=batch_size,  				  #Number of samples per gradient update
          epochs=epochs,          				  #Number of epochs to train the model 
          verbose=1,              				  #Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
          validation_data=(x_test, y_test)) 	  #Tuple on which to evaluate the loss and any model metrics at the end of each epoch.
score = model.evaluate(x_test, y_test, verbose=0) #Computes the loss based on the input
print('\n\nTest loss:', score[0])				  #Output for evaluate: loss
print('Test accuracy:', score[1])				  #Output for evaluate: accuracy

scores = model.evaluate(x_test,y_test, verbose=0)

