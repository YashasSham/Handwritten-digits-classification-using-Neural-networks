
# coding: utf-8

# In[2]:

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import numpy.ma as ma
import time
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


# In[4]:

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1.0/(1.0+np.exp(-1.0*z))


# In[5]:

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('C:/Users/kanne/Desktop/ML/Assignment2/Assignment2/basecode/mnist_all')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    n_feature=train_data.shape[1]
    std_dev = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (std_dev[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]
   

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label,index


# In[ ]:




# In[6]:

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    
#   adding bias to the training data
    train_data_bias = np.ones((train_data.shape[0], train_data.shape[1]+1))
    train_data_bias[:, :-1] = train_data
    

    a = np.dot(train_data_bias, np.transpose(w1))
    
    z = sigmoid(a)
    
    bias_z = np.append(z,np.ones([z.shape[0],1]),1)
    
    b = np.dot(bias_z, np.transpose(w2))
    o = sigmoid(b)
    
    y=np.zeros((train_label.shape[0],10))    
    for i in range(train_label.shape[0]):
        y[i][np.int(train_label[i])] = 1 
        
    #error = (-(y * np.log(o)) + ((y-1) * (np.log(1- o))))

    error = -( np.multiply(y,(np.log(o))) + np.multiply((1-y),(np.log(1-o))) )
    errorFunc= (np.sum(error))/train_data_bias.shape[0]

    obj_grad = np.array([])
    delta = o - y
    init_grad_w2 = np.dot(np.transpose(bias_z),delta)
    back1 = np.dot(delta, w2)  
    pro = np.multiply((1-bias_z),(bias_z))
    pro1 = np.multiply(pro,back1)
    init_grad_w1 = np.dot(train_data_bias.T,pro1)
    init_grad_w1 = init_grad_w1[:, 0:n_hidden]
    
   
    
    grad_w1 = (np.transpose(init_grad_w1) + (lambdaval *w1))/ train_data_bias.shape[0]
    grad_w2 = (np.transpose(init_grad_w2) + (lambdaval *w2))/ train_data_bias.shape[0]
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val= errorFunc + ((lambdaval/(2 * train_data_bias.shape[0]))*(np.sum(np.square(w1)) + np.sum(np.square(w2))))

    return (obj_val, obj_grad)


# In[7]:

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    images = data.shape[0]
    input_bias = np.ones((data.shape[0],data.shape[1]+1))
    input_bias[:, :-1] = data
   
    a = np.dot(input_bias, np.transpose(w1))
    z = sigmoid(a)
    bias_z = np.append(z,np.ones([z.shape[0],1]),1)
    b = np.dot(bias_z, np.transpose(w2))
    o = sigmoid(b)

    
    for i in range (images):
        labelIndex=np.argmax(o[i])
        labels=np.append(labels,labelIndex)


    return labels


# In[24]:

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label,index = preprocess()



#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden =50

lambda_values=[0,10,20,30,40,50,60]
Training_accuracy=[]
Validation_accuracy=[]
Test_accuracy=[]

for lambdaval in lambda_values:

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)


    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    #lambdaval = 10

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)
    obj = [index,n_hidden, w1, w2, lambdaval]
    # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
    pickle.dump(obj, open('params.pickle', 'wb'))

    # find the accuracy on Training Dataset

    Training_accuracy.append(100 * np.mean((predicted_label == train_label).astype(float)))
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # # find the accuracy on Validation Dataset
    Validation_accuracy.append(100 * np.mean((predicted_label == validation_label).astype(float)))
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # # find the accuracy on Test Dataset
    Test_accuracy.append(100 * np.mean((predicted_label == test_label).astype(float)))
    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
    
# PLotting the accuracy
plt.plot(lambda_values,Training_accuracy)
plt.plot(lambda_values,Validation_accuracy)
plt.plot(lambda_values,Test_accuracy)
plt.legend(('Training_accuracy','Validation_accuracy','Test_accuracy'))


# In[8]:

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label,index = preprocess()
total_time=[]
Training_accuracy=[]
Validation_accuracy=[]
Test_accuracy=[]

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden_values =[4,8,12,16,20]

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices

# set the regularization hyper-parameter
lambdaval = 10

for n_hidden in n_hidden_values:
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)


    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    start_time=time.time()
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)
    obj = [index,n_hidden, w1, w2, lambdaval]
    # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
    pickle.dump(obj, open('params.pickle', 'wb'))

    # find the accuracy on Training Dataset

    Training_accuracy.append(100 * np.mean((predicted_label == train_label).astype(float)))
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # # find the accuracy on Validation Dataset
    Validation_accuracy.append(100 * np.mean((predicted_label == validation_label).astype(float)))
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # # find the accuracy on Test Dataset
    Test_accuracy.append(100 * np.mean((predicted_label == test_label).astype(float)))
    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
    
    end_time=time.time()
    total_time.append(end_time-start_time)

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(n_hidden_values,total_time)
ax1.set_title('Training time vs hidden units')

ax2.plot(n_hidden_values,Training_accuracy)
ax2.plot(n_hidden_values,Validation_accuracy)
ax2.plot(n_hidden_values,Test_accuracy)
ax2.set_title('Accuracy vs hidden units')
ax2.legend(('Training_accuracy','Validation_accuracy','Test_accuracy'))
print(total_time)


# In[ ]:



