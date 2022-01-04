import numpy as np
import matplotlib.pyplot as plt


class MyLogisticRegression:
    def __init__(self):
        pass
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    def forward_and_backward_propagation(self,w,b,x_train,y_train):
        # forward propagation
        z = np.dot(w.T,x_train) + b
        y_head = self.sigmoid(z)
        loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
        cost = (np.sum(loss))/x_train.shape[1]
        # backward propagation
        derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 
        derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
        gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
        return cost,gradients

    def update(self,w, b, x_train, y_train, learning_rate,number_of_iterarion):
        cost_list = []
        cost_list2 = []
        index = []

        for i in range(number_of_iterarion):
            cost,gradients = self.forward_and_backward_propagation(w,b,x_train,y_train)
            cost_list.append(cost)

            w = w - learning_rate * gradients["derivative_weight"]
            b = b - learning_rate * gradients["derivative_bias"]
            if i % 10 == 0:
                cost_list2.append(cost)
                index.append(i)
                print ("Cost after %i th iteration : %f" %(i, cost))

        parameters = {"weight": w,"bias": b}
        self.plot_cost(index,cost_list2)
        return parameters, gradients, cost_list

    def plot_cost(self,index,cost_list):
        plt.plot(index,cost_list)
        plt.xticks(index,rotation='vertical')
        plt.xlabel("Number of Iterarion")
        plt.ylabel("Cost")
        plt.show()
        
    
    def predict(self,w,b,x_test):
        z = self.sigmoid(np.dot(w.T,x_test)+b)
        Y_prediction = np.zeros((1,x_test.shape[1]))
        for i in range(z.shape[1]):
            if z[0,i]<= 0.5:
                Y_prediction[0,i] = 0
            else:
                Y_prediction[0,i] = 1
        return Y_prediction

    def fit(self, x_train, y_train, learning_rate , num_iterations):
 
        dimension =  x_train.shape[0]
        w = np.full((dimension,1),0.01)
        b = 0.0
        parameters, gradients, cost_list = self.update(w, b, x_train, y_train, learning_rate,num_iterations)
        return parameters, gradients, cost_list 

