import numpy as np
import tensorflow as tf
from tensorflow import keras

def zero_pad(X , pad ):
  npad = ((0, 0), (2, 2), (2, 2))
  b = np.pad(X, pad_width=npad, mode='constant', constant_values=0)

  return b #np.pad(X , (, (0,0), (0,0)) , 'constant' , constant_values=(0,0))

def conv(_X , _Y):
  sum = 0
  result = [[0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]]  
  # 
  # iterate through rows of X  
  for i in range(len(_X)):  
     for j in range(len(_Y[0])):  
         for k in range(len(_Y)):  
             result[i][j] += _X[i][k] * _Y[k][j]
             sum = sum + result[i][j]  
  return sum

if __name__ == '__main__':

   
   
  _Y = [[10,11],[17,18]]  

  fashion_mnist = keras.datasets.fashion_mnist
    
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  X = train_images
  height, width = X[0].shape[:2]
  count = X[0]
  print(X[0][0][0])
  print(count)
  #print(height , width)
  for j in range (width-4):
    for i in range (height-4):
      print(i , j)
      _X = [
            [ X[0][i][j]   , X[0][i+1][j]   , X[0][i+2][j]   , X[0][i+3][j]   , X[0][i+4][j]   ]
            [ X[0][i][j+1] , X[0][i+1][j+1] , X[0][i+2][j+1] , X[0][i+3][j+1] , X[0][i+4][j+1] ]
            [ X[0][i][j+2] , X[0][i+1][j+2] , X[0][i+2][j+2] , X[0][i+3][j+2] , X[0][i+4][j+2] ]
            [ X[0][i][j+3] , X[0][i+1][j+3] , X[0][i+2][j+3] , X[0][i+3][j+3] , X[0][i+4][j+3] ]
            [ X[0][i][j+4] , X[0][i+1][j+4] , X[0][i+2][j+4] , X[0][i+3][j+4] , X[0][i+4][j+4] ] 
           ]
  print(_X)
  X_pad  = zero_pad(train_images,0)
  #print(X_pad[3][2])