import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def initialize_parameters(n_x=2,n_h=3,n_y=2):

	np.random.seed(1)
	W1 = np.random.randn(n_h,n_x)*0.01
	b1 =np.zeros((n_h,1))
	W2 =np.random.randn(n_h,n_y)*0.01
	b2=np.zeros((n_h,1))

	assert(W1.shape == (n_h,n_x))
	assert(b1.shape == (n_h,1))
	assert(W2.shape == (n_h,n_y))
	assert(b2.shape == (n_h,1))

	parameters = {	"W1":W1,
					"b1":b1,
					"W2":W2,
					"b2":b2}

	print(W1)
	print(b1)
	print(W2)
	print(b2)
	return parameters


def initialize_parameters_deep(layer_dims):
	np.random.seed(3)
	parameters = {}
	L = len(layer_dims)

	for l in range(1,L):
		print(str(layer_dims[l])+" " +str(layer_dims[l-1]))
		parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters['b'+str(l)] = np.zeros((layer_dims[l],1))

		#print(parameters['W'+str(l)])

		assert(parameters['W'+str(l)].shape == (layer_dims[l],layer_dims[l-1]))
		assert(parameters['b'+str(l)].shape == (layer_dims[l],1))

	return parameters

def relu(Z , linear_cache):
	#print(type(Z))
	activation_cache = Z
	A =np.maximum(0.0 , Z)
	return A , activation_cache

def softmax(Z , linear_cache):
	activation_cache = Z
	
	#print(softmax)
	return (np.exp(Z)/np.sum(np.exp(Z), axis=0)) , activation_cache
	#using tensorfow methods
	#return tf.exp(Z)/tf.reduce_sum(tf.exp(Z)) , activation_cache



def linear_forward(A_prev,W,B):
	print("W = " +str(W.shape)+ "W.T = " +str(W.T.shape)+ " A " +str(A_prev.shape))
	Z = np.dot(W,A_prev) + B 
	#print(str(Z.shape) + " = "+str(W.shape[0]) + " X "+str(A_prev.shape[1]))
	assert(Z.shape == (W.shape[0],(A_prev).shape[1]))
	cache = (A_prev,W,B)
	return Z , cache

def linear_activation_forward(A_prev, W , b , activation):
	if activation == "softmax":
		Z , linear_cache = linear_forward( A_prev , W ,b)
		A , activation_cache = softmax(Z , linear_cache)

	elif activation == "relu":
		Z, linear_cache= linear_forward(A_prev ,W , b)
		A, activation_cache = relu(Z , linear_cache)

	#assert(A.shape == W.shape[0],A.shape[1])
	cache = (linear_cache , activation_cache)

	return A , cache

def L_model_forward(X, parameters):
	caches = []
	#print(parameters)
	A = X
	L = len(parameters) // 2
	for l in range(1,L):
		A_prev = A
		#print(l)
		#print(A_prev.shape)
		
		A , cache = linear_activation_forward(A_prev , parameters["W"+str(l)].T , parameters["b"+str(l)] , activation = "relu")
		print(str ( A.shape)+"  = "+  str(parameters["W"+str(l)].shape) + " "+str(A_prev.shape)+ " "+str(parameters["b"+str(l)].shape ))
		caches.append(cache)
		
	print("--------------------------------------------------"+str(l)+"----------------------")
	AL , cache = linear_activation_forward(A , parameters["W"+str(L)] , parameters["b"+str(L)] , activation = "softmax")
	caches.append(cache)

	#print("AL "+ str(AL.shape))
	#assert(AL.shape == (, X.shape[1]))
	return AL, caches

	
def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = -(np.sum(Y*log(AL) , (1-Y) * log(1-AL)))/m
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	return cost
	
def linear_backward(dZ , cache ):
	A_prev , W ,b = cache
	m = A_prev.shape[1]
	dW = np.dot( dZ , A_prev.T)/m
	db = np.sum(dZ , axis=1 , keepdims = True)/m
	dA_pre = np.dot(W.T,dZ)

	assert (dA_pre.shape==A.shape)
	assert	(dW.shape == W.shape)
	assert	(db.shape==b.shape)
	return dA_pre , dW , db

def relu_backward(dA , activation_cache):
	#todo
	Z = activation_cache
	return (1.*(Z > 0))*dA

def softmax_backward(dA , activation_cache):
	return None

def linear_activation_backward(dA , cache , activation):
	linear_cache , activation_cache = cache
	
	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_pre , dW , db = linear_backward(dZ , cache)

	elif activation == "softmax":
		dZ = softmax_backward(dA , activation_cache)
		dA_pre , dW , db = linear_backward(dZ , cache)	

	return dA_pre, dW , db	

def main():
	fashion_mnist = keras.datasets.fashion_mnist

	
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	print(type(train_images))
	batch = 63
	#get the 1st batch of images from 60,000 images 
	batch1_images=np.zeros((64,28,28))
	for i in range(batch):
		batch1_images[i] = train_images[i]
	plt.figure(figsize=(10,10))
	# flatten the images to required dimensions
	batch_ = np.reshape(batch1_images , (64,784))
	batch_ = batch_.transpose()
	#print(batch_.shape)
	#plt.subplot(5,5,i+1)
	plt.grid(False)
	plt.imshow(batch1_images[0])
	#plt.imshow(train_images[0])
	#plt.show()
	#initialize_parameters(3,2,1)

	parameters = initialize_parameters_deep([784,300,10])
#-------------------to test linear forward------------------
	# A =  batch_
	# W = parameters["W1"]
	# b = parameters["b1"]
	# Z , linear_cache = linear_forward(A,W,b)
	# print("Z = "+str(Z))
	# print(Z.shape)
#-----------------------------------------------------------
#----------------to test linear_activation_forward----------
	A_prev = batch_
	W = parameters["W1"]
	b = parameters["b1"]
	# A , linear_activation_cache = linear_activation_forward(A_prev ,W ,b , activation = "relu")
	# print("with reLU : A =" + str(A))
	# A,linear_activation_cache = linear_activation_forward(A_prev , W,b,activation = "softmax")
	# print("with softmax: A = " + str(A))
	AL , cache = L_model_forward(A_prev , parameters)
	#print(AL)
if __name__ == "__main__":
	main() 
