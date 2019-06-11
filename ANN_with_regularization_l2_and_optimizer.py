import numpy as np
import math

#Class to be imported by user to develop neural network

class ANN:

    '''USER SHOULD ONLY CALL : 
        1.init_parameters
        2.train
        3.test
        4.predict
    '''

    def __init__(self, no_of_layers):
        self.no_of_layers = 0
    
    def init_parameters(self, dims_list):
        
        #Returns an initialized dictionary of parameters(weights and biases)
        #eg. parameters["W1"] means weights corresponding to layer 1

        parameters = {}
        L = len(dims_list)
        self.no_of_layers = L
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

        '''
        Input : input layer activations and initialized parameter dictionary
        Output : Output layer activations and list of caches corresponding to each layer'''

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
        
    def cost(self, AL, Y, parameters):

        #Calculates cost using cross entropy loss

        n = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / n
        L2_sum = 0
        for i in range(1, self.no_of_layers):
            L2_sum += np.sum(np.square(parameters["W" + str(i)]))
        L2_cost = (0.1 / n) * L2_sum
        cost += L2_cost
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
        dW = np.dot(dZ, A_prev.T) / n + (0.1 / n) * W
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
        dAL = -np.divide(Y, (AL + 1e-8)) + np.divide((1 - Y), (1 - AL + 1e-8))
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

    def adam_initializer(self, parameters):
        L = len(parameters) // 2
        v = {}
        s = {}
        for i in range(1, L + 1):
            v["dW" + str(i)] = np.zeros(parameters["W" + str(i)].shape)
            v["db" + str(i)] = np.zeros(parameters["b" + str(i)].shape)
            s["dW" + str(i)] = np.zeros(parameters["W" + str(i)].shape)
            s["db" + str(i)] = np.zeros(parameters["b" + str(i)].shape)
        return v, s

    def adam_optimizer_update(self, v, s, grads, learning_rate, parameters, t = 2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        L = len(parameters) // 2
        s_corrected = {}
        v_corrected = {}

        for i in range(1, L + 1):
            v["dW" + str(i)] = beta1 * v["dW" + str(i)] + (1 - beta1) * grads["dW" + str(i)]
            v["db" + str(i)] = beta1 * v["db" + str(i)] + (1 - beta1) * grads["db" + str(i)]

            v_corrected["dW" + str(i)] = v["dW" + str(i)] / (1 - beta1 ** 2)
            v_corrected["db" + str(i)] = v["db" + str(i)] / (1 - beta1 ** 2)

            s["dW" + str(i)] = beta2 * s["dW" + str(i)] + (1 - beta2) * (grads["dW" + str(i)] ** 2)
            s["db" + str(i)] = beta2 * s["db" + str(i)] + (1 - beta2) * (grads["db" + str(i)] ** 2)

            s_corrected["dW" + str(i)] = s["dW" + str(i)] / (1 - beta2 ** 2)
            s_corrected["db" + str(i)] = s["db" + str(i)] / (1 - beta2 ** 2)

            parameters["W" + str(i)] -= learning_rate * (v_corrected["dW" + str(i)]) / (np.sqrt(s_corrected["dW" + str(i)]) + epsilon)
            parameters["b" + str(i)] -= learning_rate * (v_corrected["db" + str(i)]) / (np.sqrt(s_corrected["db" + str(i)]) + epsilon)

        return parameters


    def train(self, X, Y, parameters, learning_rate = 0.075, no_of_iterations = 1000, print_cost = False, batch_size = 64, optimizer = "gradient_descent_optimizer"):

        #Trains the neural network according to training set provided by user and prints cost after every 50 iterations
        # optimizer can be gradient descent of adam
        
        if optimizer == "adam_optimizer":
            v, s = self.adam_initializer(parameters)
        for i in range(no_of_iterations):
            mini_batches = self.random_minibatches(X, Y, batch_size)
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch
                AL, caches = self.L_layers_forward(mini_batch_X, parameters)
                if i % 50 == 0:
                    cost1 = self.cost(AL, mini_batch_Y, parameters)
                grads = self.L_layer_back(AL, mini_batch_Y, caches)
                if optimizer == "gradient_descent_optimizer":
                    parameters = self.update(grads, parameters, learning_rate)
                else:
                    parameters = self.adam_optimizer_update(v, s, t = 2, grads = grads, parameters = parameters, learning_rate = learning_rate)
            if print_cost == True and i % 50 == 0:
                print("Cost of {}th iteration is {}".format(i, cost1))
            print (i)
        print ("Final cost is : ", cost1)
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
        np.clip(Z, -500, 500)
        sig = 1.0 / (1.0 + np.exp(-Z))
        return sig, Z

    def relu_back(self, Z):
        return np.greater(Z, 0).astype(int)

    def sigmoid_back(self, Z):
        sig, _ = self.sigmoid(Z)
        return sig * (1 - sig)

    def random_minibatches(self, X, Y, mini_batch_size):
        mini_batches = []
        n = X.shape[1]
        permutations = list(np.random.permutation(n))
        no_of_complete_batches = n // mini_batch_size
        X_shuffle = X[:, permutations]
        Y_shuffle = Y[:, permutations]
        for i in range(no_of_complete_batches):
            mini_batch_X = X_shuffle[:, mini_batch_size * i : mini_batch_size * (i + 1)]
            mini_batch_Y = Y_shuffle[:, mini_batch_size * i : mini_batch_size * (i + 1)]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        if n % mini_batch_size != 0:
            mini_batch_X = X_shuffle[:, mini_batch_size * no_of_complete_batches : n]
            mini_batch_Y = Y_shuffle[:, mini_batch_size * no_of_complete_batches : n]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        return mini_batches
