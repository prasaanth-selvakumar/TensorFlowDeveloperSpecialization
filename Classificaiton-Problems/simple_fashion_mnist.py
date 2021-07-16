### First classification on the specialization
### Using a Dense neural netwok to predict classes in fashion MNIST dataset

from tensorflow import keras
import matplotlib.pyplot as plt

fm = keras.datasets.fashion_mnist ### using the inbuit datasets class to fetch fashion mnist data

## This contains 70k labelled images of 28x28 pixels;

(train_data,train_lables),(test_data,test_labels) = fm.load_data()  ## Out of the 70k 50k can be used for training and 20k can be used for testingg

#### Displaying a sample version of the image
plt.imshow(train_data[0])
print("Train label",train_lables[0]) # Checking if the labels are 1 hot encoded or 0 -9
# Neural network work well with normalized data. So let's divide all values in training and test data by 255
train_data = train_data/255.0  # Divides the entire array by 255
test_data = test_data/255.0

### Creating a simple neural network that contains one hidden layer to predict classes

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28,28)))  ## changeing the input dimensions to suit the dense layer
model.add(keras.layers.Dense(128,activation='relu')) ## using a hidden layer with 128 hidden usints to predict the data
model.add(keras.layers.Dense(10,activation='softmax')) ## Softmax has been used here to select 1 of 10 classes

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])  # Sparse categorical cross entropy is used
# when the labels are 0 -9 instead of being one hot encoded

model.fit(train_data,train_lables,epochs = 8,batch_size = 128)

print(model.evaluate(test_data,test_labels,batch_size=64))

