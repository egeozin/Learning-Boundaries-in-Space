import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
#from matplotlib.pyplot import imshow
from utils import *

########################
# Import the layer types needed
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
########################


##TODO

# ADD INIT AND UPDATE FUNCTIONS
# HANDLE DATA FROM SIMULATION
# WRITE OBJECTIVE FUNCTION
# LOOK AT OUTPUT ACTIVATION 

class CNNTraining:
# Load the dataset
	def __init__(self,X,out_layers):
	 	# Define the model
		self.model = Sequential()
		print ("X Model: " , X.shape)
		self.model.add(Convolution2D(nb_filter=32,nb_row=3,nb_col=3,dim_ordering='tf',input_shape=(X.shape[0], X.shape[1], X.shape[2])))
		self.model.add(Activation("relu"))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Convolution2D(nb_filter=64,nb_row=3,nb_col=3))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Flatten())
		self.model.add(Dense(output_dim=128))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(output_dim=out_layers))
		self.model.add(Activation("sigmoid"))
		self.model.compile(loss='mean_squared_logarithmic_error', optimizer = SGD(lr=1, momentum=0.9, nesterov=True) )


		## Fit the model (10% of training data used as validation set)
		# model.fit(X_train, y_train, nb_epoch=2, batch_size=32,validation_split=0.1, show_accuracy=True)

		## Evaluate the model on test data
		# objective_score = model.evaluate(X_test, y_test, show_accuracy=True, batch_size=32)
		# print objective_score

	def update_model_offline(self,X,Y):
		history=model.fit(X, Y, nb_epoch=2, batch_size=32,validation_split=0.1, show_accuracy=True)
		return loss


	def update_model_online(self,X, Y):
		X_Train=np.expand_dims(X,axis=0)
		Y_Train=np.expand_dims(Y,axis=0)
		#loss=self.model.train_on_batch(X_Train,Y_Train)
		loss = self.model.fit(X_Train, Y_Train, nb_epoch=1, batch_size=1, validation_split=0.0, show_accuracy=True)
		# history=self.model.fit(X_Train, Y_Train, nb_epoch=1, batch_size=1,validation_split=0.0, verbose=1, show_accuracy=True)		
		return loss

	def update_model_online_lr(self,X, Y, counter):
		X_Train=np.expand_dims(X,axis=0)
		Y_Train=np.expand_dims(Y,axis=0)
		#loss=self.model.train_on_batch(X_Train,Y_Train)
		loss = self.model.fit(X_Train, Y_Train, nb_epoch=1, batch_size=1, validation_split=0.0, show_accuracy=True)
		lr = self.model.optimizer.lr.get_value()
		self.model.optimizer.lr.set_value(np.asarray((1.0/counter), dtype=np.float32 ))
		print("Learning_parameter: ", self.model.optimizer.lr.get_value())
		# history=self.model.fit(X_Train, Y_Train, nb_epoch=1, batch_size=1,validation_split=0.0, verbose=1, show_accuracy=True)		
		return loss

	def load_weights(self,file):
		self.model.load_weights(file)

	def predict_model(self,X):
		X_Pred=np.expand_dims(X,axis=0)
		return self.model.predict_on_batch(X_Pred)

	#TODO IMPORT SENSOR DATA
	#X_train, y_train, X_test, y_test = getMNISTData()

	# We need to rehape the data back into a 1x28x28 image
	# X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
	# X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

	# ## Categorize the labels
	# y_train = np_utils.to_categorical(y_train, num_classes)
	# y_test = np_utils.to_categorical(y_test, num_classes)

	