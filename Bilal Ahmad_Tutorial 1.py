#Bilal AHMAAD 577552
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
class Perceptron (object):
    def __init__(self,eta= 0.01, n_inter=10):
        self.eta=eta 
        self.n_inter= n_inter

    def weight_init(self,X):
        return np.dot (X, self.w_[1:]) + self.w_[0]
    
    def predict (self,x):
        return np.where(self.weighted_sum(x)>= 0.0, 1, -1) 
    
    def fit (self,x,y): 
        self.w_ = np.zeros (1+ x.shape [1])
        
        self. errors_ = []

        
        print("weights:", self.w_)
        
        for _ in range (self.n_inter):
            error = 0

       
        for xi,y in zip (x,y):
            y_pred = self. pred(xi)
            
            update = self.eta * (y -y_pred)

            self. w_[1:] = self.w_[1:] + update * xi  

            print("Updated Weights:", self.weight_[1:]) 

            self.w_[0] = self.w_[0] + update 

            error += int (update != 0.0)

            self.errors_.append(error)

       
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

df = shuffle(df)

df.head()

x=df.iloc[:,0:4].values

y=df.iloc[:,4].values

print(x[0:5])

print(y[0:5])

print("splitting data into train and test sets")
from sklearn.model_selection import train_test_split 

train_data, test_data, train_labels, test_labels = train_test_split (x,y,test_size=0.25)

train_labels = np.where (train_labels == 'Iris-setosa', 1, -1)

test_labels = np.where (test_labels == 'Iris-setosa', 1, -1)

print ('train data:' , train_data [0:2])

print ('train labels:' , train_labels [0:2])

print ('test data:' , test_data [0:2])

print ('test labels:' , test_labels [0:2])
print(train_labels.shape)

from sklearn.linear_model import Perceptron 

perceptron2=Perceptron(eta0=0.1, max_iter=10)

perceptron2.fit(train_data,train_labels) 

test_preds = perceptron2.predict(test_data)

print(test_preds)

from sklearn.metrics import accuracy_score 

y_preds = perceptron2.predict(test_data)

accuracy = accuracy_score (y_preds,test_labels)

print ('Accuracey:', round (accuracy, 2) * 100 ,"%")





import numpy as np

class perceptron (object):

    def __init__(self,eta= 0.01, n_inter=10):
        
        self.eta=eta
        self.n_inter= n_inter

    def weight_init(self,X):
            return np.dot (X, self.w_[1:]) + self.w_[0]
            
    def predict (self,x):
            return np.where(self.weighted_sum(x)>= 1, 0 )
    
##########################
#TASK2 here I added sigmoid
# # ##########################   
            
    def sigmoid(self,X):
        return ( 1 / 1 + np.exp( - ( np.dot(X, self.w_[1:]) + self.w_[0]  ) ) )
                
    def fit (self,x, y):
                
        self.w_ = np.zeros (1+ x.shape [1])
        self. errors_ = []

        print("weights:", self.w_)

        for _ in range (self.n_inter):
            error = 0
            for xi,y in zip (x,y):
                y_pred = self. pred(xi)

                update = self.eta * (y-y_pred) 

                self. w_[1:] = self.w_[1:] + update * xi 

                print("Updated Weights:", self.weight_[1:]) 
                self.w_[0] = self.w_[0] + update
                error += int (update != 0.0)

                self.errors_.append(error)
                            
                             
##########################
#TASK3
# # ##########################                            
import pandas as pd
    
from sklearn.utils import shuffle   
import numpy as np
df = pd.read_csv ('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
                    
df = shuffle (df)
df.head ()
x=df.iloc[:,0:4].values
y=df.iloc[:,4].values

print(x[0:5])
print(y[0:5])
    
from sklearn.model_selection import train_test_split 

train_data, test_data, train_labels, test_labels = train_test_split (x,y,test_size=0.25)

train_labels = np.where (train_labels == 'Iris-setosa', 1, 0)

test_labels = np.where (test_labels == 'Iris-setosa', 1, 0)

print ('train data:' , train_data [0:2])

print ('train labels:' , train_labels [0:2])

print ('test data:' , test_data [0:2])

print ('test labels:' , test_labels [0:2])
print(train_labels.shape)

from sklearn.linear_model import Perceptron 

perceptron2=Perceptron(eta0=0.1, max_iter=10)

perceptron2.fit(train_data,train_labels) 

test_preds = perceptron2.predict(test_data)

print(test_preds)

from sklearn.metrics import accuracy_score 

y_preds = perceptron2.predict(test_data)

accuracy = accuracy_score (y_preds,test_labels)

print ('Accuracey:', round (accuracy, 2) * 100 ,"%")
print("Task 3 is done")



##########################
#TASK1
# # ##########################   
manual=input("Enter 4 features of iris flower (sepal length, sepal width, petal length, petal width) separated by commas:" )

man_Arrauy= np.array(list(map(float, manual.split(",")))).reshape(1, -1)
man_prediction = perceptron2.predict(man_Arrauy)
print("The predicted class for the input features is:", "Iris-setosa" if man_prediction[0] == 1 else "Not Iris-setosa") 




         
