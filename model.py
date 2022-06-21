from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ModelMaker():

  def __init__(self, hidden_layers=1, hidden_nodes=4):
    """
    This function will prepare a Sequential model
    according to input configuration

    hidden_layers -> Int : number of hidden layers in network
    hidden_layers -> Int : number of nodes in each hidden layer
    return -> None
    """
    self.hidden_layers = hidden_layers
    self.hidden_nodes = hidden_nodes
    
    self.model = None
    self.X = None
    self.y = None

    print(f"Number of hidden layers: {self.hidden_layers}")
    print(f"Number of nodes in each hidden layer: {self.hidden_nodes}")

    model = Sequential()
    
    # input layer
    model.add(Dense(12, input_shape=(2,), activation='relu'))
    
    # hidden layer(s)
    for l in range(0, self.hidden_layers):
      model.add(Dense(self.hidden_nodes, activation='relu'))
    
    # output layer -- binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the keras model
    model.compile(loss='binary_crossentropy', 
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # set the instance to trained model
    self.model = model
    print(self.model.summary())

  
  def prepare_data(self, dataset):
    """
    This function will extract features and labels.
    Additionaly it will split the training and validation data
    
    dataset -> pd.DataFrame : dataset
    return -> train and val tuple : (X_train, X_val, y_train, y_val)
    """
    print("Preparing dataset...")
    self.X = dataset.iloc[:,0:2]
    self.y = dataset.iloc[:,2]

    print(f"Features are:{self.X.columns}")

    X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.30)#random_state=42)
    return X_train, X_val, y_train, y_val

  def train_model(self, train_X, train_y, epochs=10, batch_size=10):
    """
    This function will train the model based on input
    epochs and batch_size. Additionaly it will plot the loss and
    accuracy curves of the model

    train_X -> pd.DataFrame : training features
    train_y -> pd.Series : training labels
    epoch -> Int : number of samples processed before the model is updated
    batch_size -> Int : number of complete passes through the training dataset

    return -> None
    """
    print("Training model...")
    history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    for k,v in history.history.items():
      print(k,v)
      print(f"Avg. {k}", np.mean(v))
        
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
  
  def predict(self, val_X, val_y):
    """
    This function will evaluate the model based on validation data
    and print 5 predictions for display

    val_X -> pd.DataFrame : validation features
    val_y -> pd.Series : validation labels

    return -> None
    """
    print("Running predictions...")
    predictions = (self.model.predict(val_X) > 0.5).astype(int)

    # summarize the first 5 cases
    print("Feature Set => Predicted Value (Actual Value)")
    for i in range(5):
      print('%s => %d (expected %d)' % (val_X.iloc[i].values, predictions[i], val_y.iloc[i]))

# Program starts from here
URL = "https://raw.githubusercontent.com/devAmoghS/DL-Assignment-1/main/data2_0.75_6.csv"
df = pd.read_csv(URL, names=["X1", "X2", "class"], header=None)    

# intialising model
modm = ModelMaker(2, 12)

# preparing dataset
X_train, X_val, y_train, y_val = modm.prepare_data(df)

# training model
modm.train_model(X_train, y_train, epochs=10000, batch_size=10)

# running predictions
modm.predict(X_val, y_val)
