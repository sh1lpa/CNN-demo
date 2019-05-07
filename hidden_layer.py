import numpy as np
import matplotlib 


def initialize_parameters(n_x=2,n_h=3,n_y=2):

	np.random.seed(1)
	W1 = np.random.randn(n_h,n_x)*0.01
	b1 =np.zeros((n_h,1))
	W2 =np.random.randn(n_y,n_h)*0.01
	b2=np.zeros((n_y,1))

	assert(W1.shape == (n_h,n_x))
	assert(b1.shape == (n_h,1))
	assert(W2.shape == (n_y,n_h))
	assert(b2.shape == (n_y,1))

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
		parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters['b'+str(l)] = np.zeros((layer_dims[l],1))

		print(parameters['W'+str(l)])

		assert(parameters['W'+str(l)].shape == (layer_dims[l],layer_dims[l-1]))
		assert(parameters['b'+str(l)].shape == (layer_dims[l],1))

	return parameters


def linear_forward(A,W,B):
	Z = np.dot(W,A)+B
	assert(Z.shape == (W.shape[0],A.shape[1]))
	cache(A,W,B)
	return Z , cache

def linear_activation_forward(A_pre , W , b , activation):
	if activation == "sigmoid"
		Z , linear_cache = linear_forward( A_pre , W ,b)
		A , activation_cache = sigmoid(Z)

	elif activation == "relu":
		Z,linear_forward(A_pre ,W , b)
		A, activation_cache = relu(Z)

	assert(A.shape == W.shape[0],A.shape[1])
	cache = (linear_cache , activation_cache)

	return A , cache

def L_model_forward(X, parameters):
	caches = [] 
	A = X
	L = len(parameters)

	for i in range(1,L)
		A_prev = A
		A , cache = linear_activation_forward(A_prev , parameters["W"+str(l)] , parameters["b"+str(l)] , activation = "relu")
		caches.append(cache)
	AL , cache = linear_activation_forward(A_prev , parameters["W"+str(l)] , parameters["b"+str(l)] , activation = "sigmoid")
	caches.append(cache)


	assert(AL.shape == (1, X.shape[1]))
	return AL, caches

	
def compute_cost(AL, Y)
	m = Y.shape[1]
	cost = -(np.sum(Y*log(AL) , (1-Y) * log(1-AL)))/m
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	return cost
	


def main():
	#initialize_parameters(3,2,1)
	#initialize_parameters_deep([3,5,3,4,3])
#-------------------to test linear forward------------------
	#A,W,b = linear_forward_test_case()
	#Z , linear_cache = linear_forward(A,W,b)
	#print("Z = "+str(Z))
#-----------------------------------------------------------
#----------------to test linear_activation_forward----------
	A_pre ,W,b =  linear_activation_forward_test_case()
	#A , linear_activation_cache = linear_activation_forward(A_pre ,W ,b , activation = "sigmoid")
	#print("with sigmoid : A =" + str(A))
	#A,linear_activation_cache = linear_activation_forward(A_prev , W,b,activation = "relu")
	#print("with reLU: A = " + str(A))
	
if __name__ == "__main__":
	main()		
