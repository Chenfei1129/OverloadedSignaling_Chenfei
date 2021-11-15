#def calculateSignalCost(signal, cost = 10):
 #   return (len(signal.split()) *cost)

class CalculateSignalCost:
    def __init__(self, cost):
        
        self.cost = cost
        
    def __call__(self, signal):
        
        return (len(signal.split()) *self.cost)
    
