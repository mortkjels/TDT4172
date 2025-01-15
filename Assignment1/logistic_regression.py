import numpy as np


class LogisticRegression():

  def __init__(self, learning_rate=0.001, epochs=400):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.weights = None
    self.bias = None
    self.losses = []
    self.train_accuracy = []

  def fit(self, X, y):
        

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            lin_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(lin_model)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            accuracy = np.mean(pred_to_class == y)
            self.train_accuracy.append(accuracy)

            

            if epoch % 25 == 0:
                print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")
            
        
        
    
  def predict(self, X):
      """
      Generates predictions
      
      Note: should be called after .fit()
      
      Args:
          X (array<m,n>): a matrix of floats with 
              m rows (#samples) and n columns (#features)
          
      Returns:
          A length m array of floats
      """
      lin_model = np.dot(X, self.weights) + self.bias
      y_pred = self.sigmoid(lin_model)
      return [1 if _y > 0.5 else 0 for _y in y_pred]
  


  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def compute_gradients(self, X, y, y_pred):
        samples = X.shape[0]
        error = y_pred - y
        grad_w = (1/samples) * np.dot(X.T, error)
        grad_b = (1/samples) * np.sum(error)
        return grad_w, grad_b
  
  def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

  def compute_loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
  


  def predict_proba(self, X):
        lin_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(lin_model)
        return y_pred
  
  