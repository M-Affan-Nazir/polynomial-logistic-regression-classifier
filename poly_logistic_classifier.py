from sklearn.preprocessing import PolynomialFeatures
import random
import math
import numpy as np
import os



class Polynomial_Classifier:

     def __init__(self, order, MinMaxObject = None):
          self.order = order
          self.weightInitialized = False
          self.minmax = MinMaxObject
          self.modelName = ""
     
     def name(self, name):
        self.modelName = name

     def _getFeatureVector(self,data = []):
        data = [data]
        poly = PolynomialFeatures(self.order)
        features = poly.fit_transform(data)
        return features[0]

     def _convertToFeatureMatrix(self,x):
        matrix = []
        for i in x:
            matrix.append(self._getFeatureVector(i))
        return matrix
     
     def train(self, trainingData=(), validationData=(), epochs=0, batchSize=1, stepSize=0.01):
        xTrain =  np.array(self._convertToFeatureMatrix(trainingData[0]))
        xVal = np.array(self._convertToFeatureMatrix(validationData[0]))
        yTrain = np.array(trainingData[1])
        yVal = np.array(validationData[1])
        self.batchSize = batchSize
        self.features = len(xTrain[0])
        if(self.weightInitialized == False):
            self.w = np.random.rand(self.features)*0.01
            self.weightInitialized = True
        for i in range(epochs):
            indices = list(range(len(xVal)))
            random.shuffle(indices)
            xVal_shuffled = [xVal[i] for i in indices]
            yVal_shuffled = [yVal[i] for i in indices]
            for j in range(0,len(xVal_shuffled),batchSize):
                batchX = xVal_shuffled[j:j+batchSize]
                batchY = yVal_shuffled[j:j+batchSize]
                self._epoch(batchX,batchY, stepSize)
            loss = self._cost(xTrain,yTrain)
            valLoss = self._valLoss(xVal,yVal)
            print("Epoch " + str(i+1) + ": | loss = " + str(self._round(loss,3)) + " validation loss = " + str(self._round(valLoss,3)))

     def _epoch(self, x, y, stepSize):
        g = self._grad(x,y)
        self.w = self.w - stepSize * g

     def _grad(self,x,y):
        p = self._predict(x)
        diff = p - y
        g = np.dot(diff.T,x)
        g = g / len(p)
        return g

     def _predict(self, x):
        return self._sigmoid(np.dot(x,self.w))
     
     def _cost(self,x,y):
        p = self._predict(x)
        #Categorial cross entropy
        loss = 0
        for i in range(len(p)):
            loss += -y[i]*math.log(p[i]) - (1 - y[i])*math.log(1-p[i])   #p is a vector of prediction. We need the average cost; so we sum all predictions, (and coresponding y)
        loss = loss/len(p)
        return loss

     def _valLoss(self, x,y):
        return self._cost(x,y)
     
     def _sigmoid(self, x):
        sig = []
        for i in range(len(x)):
            sig.append(1 / (1 + math.exp(-x[i])))
        return np.array(sig)
     
     def _round(self, num, sig_figs):
        if num == 0:
            return 0
        else:
            return round(num, sig_figs - int(np.floor(np.log10(abs(num)))) - 1)
    
     def save(self, path=None):
        if self.modelName != "":
            if path is None:
                path = os.getcwd()
            if not os.path.isdir(path):
                print(f"The directory {path} does not exist.")
                return
            if self.minmax != None:
                min = self.minmax.data_min_ #An array, since num_weights > 1
                max = self.minmax.data_max_

            file_path = os.path.join(path, self.modelName + ".prf")

            w_str = str(self.w.tolist())
            try:
                with open(file_path, 'w') as f:
                    f.write(w_str)
            except IOError as e:
                print(f"An error occurred: {e}")
        else:
            print()
            raise Exception("Save Failed: Model name required in order to export weights")