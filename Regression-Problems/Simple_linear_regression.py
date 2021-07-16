from tensorflow import keras ## building model using tensorflow
import numpy as np # used to genrate random data for the problem

class Dataset:
    def __init__(self,m=10):
        self.xs = np.random.randint(0,25,m).astype(np.float32)
        self.ys = (2*self.xs + 5 +(np.random.rand(m)*0.1)).astype(np.float32)
    def __str__(self):
        return f"X {str(self.xs)} \ny {str(self.ys)}"

    def __getitem__(self, item):
        return self.xs[item],self.ys[item]

TrainData = Dataset()
#print(TrainData)

## Building a simple neural network
## Sequential is used for building the architecture
## We are trying to approximate a simple linear function
    # with one dependant variable so we have used 1 unit as the output
    # the input dimensions would be 1 - no of features, the number of examples provided need not be mentioned here
    # it automatically initializes the bias term because the use_bias term is true by default

### Defining the model
model = keras.Sequential([keras.layers.Dense(units=1,input_dim=1,activation='linear')])

### Defining the optimizer and the loss function -> needs to be done during compilation

model.compile(optimizer='adam',loss='mean_squared_error')

### Training the model | or fitting the model on training data
## epochs by default is set to 1
## epoch - one full run over the training data
model.fit(TrainData.xs,TrainData.ys,epochs=1200,batch_size=2)


print(model.get_layer(index=0).trainable_weights)
print(model.predict([10])) ## the output of the prediction should be close to 25

#### Learnings
## Since the number of input samples are low
# The Model takes about 1200 epochs to converge with a batch size of 2
# increasing the batch size to 10