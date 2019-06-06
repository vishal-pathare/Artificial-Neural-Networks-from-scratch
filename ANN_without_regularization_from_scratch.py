import numpy as np

#Class to be imported by user to develop neural network

class ANN:

    '''USER SHOULD ONLY CALL : 
        1.init_parameters
        2.train
        3.test
        4.predict
    '''

    def init_parameters(self, dims_list):
        
        #Returns a dictionary of parameters(weights and biases)
        #eg. parameters["W1"] means weights corresponding to layer 1

        parameters = {}
        L = len(dims_list)
        for i in range(1, L):
            parameters["W" + str(i)] = np.random.randn(dims_list[i], dims_list[i - 1]) * 0.01
            parameters["b" + str(i)] = np.random.randn(dims_list[i], 1) * 0.01
        return parameters
    
    def linear_forward(self, A_prev, W, b):

        #Calculates and returns Z

        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    def activation_forward(self, A_prev, W, b, activation):

        #Calculates activation of layer

        Z, cache = self.linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A, Z = self.sigmoid(Z)
        else:
            A, Z = self.relu(Z)
        caches = (cache, Z)
        return A, caches

    def L_layers_forward(self, X, parameters):

        #Input : input layer activations and initialized parameter dictionary
        #Output : Output layer activations and list of caches corresponding to each layer 

        caches = []
        L = len(parameters) // 2
        A = X
        for i in range(1, L):
            A_prev = A
            A, cache = self.activation_forward(A_prev, parameters["W" + str(i)], parameters["b" + str(i)], "relu")
            caches.append(cache)
        AL, cache = self.activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
        caches.append(cache)
        return AL, caches
        
    def cost(self, AL, Y):

        #Calculates cost using cross entropy loss

        n = Y.shape[1]
        cost = -np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL))) / n
        cost = np.squeeze(cost)
        return cost

    def linear_back(self, dZ, cache):

        '''
        Description : Calculates derivatives of cost function
        Input : dZ, dA, dW, db (dZ given because activation function may vary depending on layer position)
        Variables:
            dA_prev = derivative of cost wrt activation of previous layer
            dW = derivative of cost wrt weights of current layer
            db = derivative of cost wrt biases of current layer'''

        A_prev, W, _ = cache
        n = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / n
        db = np.sum(dZ, axis = 1, keepdims = True) / n
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def activation_back(self, dA, cache, activation):

        #Returns derivatives of cost function wrt previous layer activation, weights of current layer, 
        #biases of current layer

        linear_cache, Z = cache
        if activation == "relu":
            dZ = dA * self.relu_back(Z)
        else:
            dZ = dA * self.sigmoid_back(Z)
        dA_prev, dW, db = self.linear_back(dZ, linear_cache)
        return dA_prev, dW, db

    def L_layer_back(self, AL, Y, caches):

        '''
        Description : performs back propagation throughout the neural network
        Input : activations of output layer as calculated by forward propagation, true activations, caches list
        Output : dictionary of gradients which can be used to update parameters'''

        grads = {}
        Y = Y.reshape(AL.shape)
        L = len(caches)
        #Calculate gradient of output layer
        dAL = -np.divide(Y, AL) + np.divide((1 - Y), (1 - AL))
        dA_prev, dW, db = self.activation_back(dAL, caches[L - 1], "sigmoid")
        grads["dA" + str(L - 1)] = dA_prev
        grads["dW" + str(L)] = dW
        grads["db" + str(L)] = db
        for i in reversed(range(1, L)):
            grads["dA" + str(i - 1)], grads["dW" + str(i)], grads["db" + str(i)] = self.activation_back(grads["dA" + str(i)], caches[i - 1], "relu")
        return grads

    def update(self, grads, parameters, learning_rate):
        
        #Updates parameters according to corresponding gradients and learning rate

        L = len(parameters) // 2
        for i in range(1, L):
            parameters["W" + str(i)] -= learning_rate * grads["dW" + str(i)]
            parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]
        return parameters

    def train(self, X, Y, parameters, learning_rate = 0.075, no_of_iterations = 1000, print_cost = False):

        #Trains the neural network according to training set provided by user and prints cost after every 50 iterations

        for i in range(no_of_iterations):
            AL, caches = self.L_layers_forward(X, parameters)
            if i % 50 == 0:
                cost1 = self.cost(AL, Y)
            grads = self.L_layer_back(AL, Y, caches)
            parameters = self.update(grads, parameters, learning_rate)
            if print_cost == True and i % 50 == 0:
                print("Cost of {}th iteration is {}".format(i, cost1))
            print (i)
        print ("Final cost is : ", self.cost(AL, Y))
        return parameters

    def predict(self, X, parameters):

        #Returns the predicted activations of output layer
        
        AL, _ = self.L_layers_forward(X, parameters)
        return AL

    def test(self, X, Y, parameters):

        #Tests the calculated activations against the true activations of test set and return the accuracy of predictions

        AL, _ = self.L_layers_forward(X, parameters)
        correct = 0
        AL = AL.T
        Y = Y.T
        for i in range(AL.shape[0]):
            if np.argmax(AL[i]) == np.argmax(Y[i]):
                correct += 1
        print ("Accuracy of the set is {}".format((correct / AL.shape[0]) * 100))

    '''Helper functions'''

    def relu(self, Z):
        return np.maximum(0, Z), Z

    def sigmoid(self, Z):
        sig = 1 / (1 + np.exp(-Z))
        return sig, Z

    def relu_back(self, Z):
        return np.greater(Z, 0).astype(int)

    def sigmoid_back(self, Z):
        sig, _ = self.sigmoid(Z)
        return sig * (1 - sig)