import numpy as np

class Normalizer:
    def __init__(self,Ytrain,mean=1,width=1):
        self.Ymin,self.Ymax = np.min(Ytrain,axis=0), np.max(Ytrain,axis=0)
        self.Ywidth = self.Ymax - self.Ymin
        self.mean = mean
        self.width = width
        
    def whiten(self,Ytrain):
        return (Ytrain - (self.Ymax+self.Ymin)/2.) * self.width/self.Ywidth
    
    def color(self,Ypred):
        return Ypred * (self.Ywidth/self.width) + (self.Ymax+self.Ymin)/2.
