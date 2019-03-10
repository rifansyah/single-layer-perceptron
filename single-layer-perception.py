import numpy as np
import math
from math import exp
import matplotlib.pyplot as plt
import pandas as pd

class SLP(object):

    def __init__(self, iris_data, epoch, k, alpha):
        self.iris = iris_data
        self.epoch = epoch
        self.k = k
        self.alpha = alpha
        self.weight = [0.5, 0.5, 0.5, 0.5]
        self.bias = 0.5
        self.accuracy_training_list = []
        self.errors_training_list = []
        self.accuracy_test_list = []
        self.errors_test_list = []
        self.correct_train_data = 0
        self.correct_test_data = 0

    def get_result(self, iris_data):
        temp = 0
        for i in range(len(iris_data)):
            temp += iris_data[i] * self.weight[i]
        temp += self.bias
        return temp

    def get_activation(self, result):
        return 1 / (1 + exp(-result))
    
    def get_prediction(self, activation):
        if(activation >= 0.5):
            return 1
        else:
            return 0
    
    def get_error(self, iris_type, activation):
        return pow((iris_type - activation), 2)
    
    def update_weight_bias(self, iris_data, activation):
        for i in range(len(iris_data) - 2):
            d_weight = 2 * iris_data[i] * (iris_data[5] - activation) * (1 - activation) * activation
            self.weight[i] = self.weight[i] + (self.alpha * d_weight)
        
        d_bias = 2 * (iris_data[5] - activation) * (1 - activation) * activation
        self.bias = self.bias + (self.alpha * d_bias)

    def k_fold(self):
        fold = []
        for i in range(self.k):
            fold.append(self.iris[20*i:(20*i)+20])

        error_train_temp = 0
        error_test_temp = 0
        accuracy_train_temp = 0
        accuracy_test_temp = 0
        train_data = 0
        test_data = 0

        for i in range(self.k):
            data_test = fold[i]
            data_train = []
            for j in [x for x in range(self.k) if x != i]:
                data_train.extend(fold[j])
            
            # train data
            for data in data_train:
                result = self.get_result(data[:4])
                activation = self.get_activation(result)
                prediction = self.get_prediction(activation)
                error = self.get_error(data[5], activation)
                self.update_weight_bias(data, activation)

                if(prediction == data[5]):
                    accuracy_train_temp += 1
                
                error_train_temp += error
                train_data += 1


            # test data
            for data in data_test:
                result = self.get_result(data[:4])
                activation = self.get_activation(result)
                prediction = self.get_prediction(activation)
                error = self.get_error(data[5], activation)

                if(prediction == data[5]):
                    accuracy_test_temp += 1

                error_test_temp += error
                test_data += 1
        
        self.accuracy_training_list.append(accuracy_train_temp / train_data)
        self.errors_training_list.append(error_train_temp / train_data)
        self.accuracy_test_list.append(accuracy_test_temp / test_data)
        self.errors_test_list.append(error_test_temp / test_data)

    def train(self):
        # random data
        np.random.shuffle(self.iris)

        for i in range(self.epoch):
            self.k_fold()

    def plot(self):
        plt.figure(1)
        plt.plot(np.arange(self.epoch), self.errors_training_list, color = 'green', label = 'training')
        plt.plot(np.arange(self.epoch), self.errors_test_list, color = 'red', label = 'test')
        plt.xlabel('Epoch')
        plt.ylabel('Average errors @epoch')
        plt.title('Error')      
        plt.legend()

        plt.figure(2)
        plt.plot(self.accuracy_training_list, color = 'green', label = 'training')
        plt.plot(self.accuracy_test_list, color = 'red', label = 'test')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracies @epoch')
        plt.legend()
        plt.show()

            
def main():
    iris_data = pd.read_csv('iris.csv').values

    slp = SLP(iris_data, epoch = 5, k = 5, alpha = 0.1)
    slp.train()
    slp.plot()

if __name__ == "__main__":
    main()