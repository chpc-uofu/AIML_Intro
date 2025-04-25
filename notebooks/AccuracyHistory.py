import tensorflow
from tensorflow import keras
class AccuracyHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
     self.acc = []
     self.loss = [] 

  def on_epoch_end(self, batch, logs={}):
     print (logs)
     self.acc.append(logs.get('sparse_categorical_accuracy'))  
     self.loss.append(logs.get('loss'))
     #possible values are acc (accuracy with training data) and the more important val_acc(accuracy with test data)
     #For regression you can use val_mean_absolute_error

