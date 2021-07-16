"""In this script we are aiming to build a simple house price prediction problem
    - Based on 2 features no of bedrooms and  total sq footage of the house
    - mean price of houses excluding bedroom costs are $50k - cost of paint, wiring etc
    - each bedroom costs an additional $50k
    - we also have a multiplicative factor of $100 for the sq footage"""

import numpy as np
from tensorflow import keras

class PricingDataset:
    def __init__(self,m=10):
        self.bedrooms = np.random.randint(1,5,m).astype(np.float64) # maximum number of bedrooms normal houses have is 5 so 1-5
        self.total_sq_footage = np.random.randint(500,5000,m).astype(np.float64) # houses range from 500 to 5000 sq feet depending on  how spacious they are
        self.X = np.column_stack([self.bedrooms,self.total_sq_footage]) # X for trining
        self.y = 50000*self.bedrooms + 50*self.total_sq_footage + 50000 + np.random.rand(m)*10

    def __str__(self):
        return f"""Dataset X:\n{self.X[:5,:]}\nY:\n{self.y[:5]}\n\n total Length = {X.shape[0]}"""
    def __getitem__(self, item):
        return self.X[item,:],self.y[item]

#pd1 = PricingDataset()
#print(pd1[0])

## Creating input dataset with 100 Samples

pd2 = PricingDataset(300)

### This problem can be solved by using the help of a similar model used in Simple_linear_regression.py - only change is we will
#    need two neurons in the input dimension instead of 1

### simple single layer neural netowork architecture
model = keras.Sequential([keras.layers.Dense(units=1,activation='linear',input_dim=2)])  # Single layer simple neural network to predict house prices

### Adding loss and activation  to the model

model.compile(optimizer=keras.optimizers.Adam(learning_rate=50),loss='mean_squared_error') ### learning rate has been set to a high value intentionally

### Fitting the model based on input data for 1200 epochs
model.fit(pd2.X,pd2.y,epochs=1200,batch_size= 128)

## Getting model Parameters
print(model.get_layer(index=0).trainable_weights)  # the trainable weights should be close to the values we have already coded

### Predicting the model
print(model.predict([np.array([3,1800]).reshape(1,2)]))   ### we can see that the model takes a long time to converge - we have to run it for close to 50000 epochs


### Learnigns
### this ca be solved by 2 methods
 ## increasing the learning rate or by normalizing the input variables
    ## Incresing the learning rate here is a bad idea - because if you trian the model for more then 1500 epochs you can observe the loss value increasing
    ## the best approach to avoid this would be to use a small learning rate and normalizing the input features and the target

