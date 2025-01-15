import numpy as np
class LinearRegression:
    
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learningrate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses = []

    
    def compute_gradients(self, X, y, y_pred):
        m = X.shape[0]
        error = y_pred - y
        grad_w = (2/m) * np.dot(X.T, error)
        grad_b = (2/m) * np.sum(error)
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learningrate * grad_w
        self.bias -= self.learningrate * grad_b

    def compute_loss(self, y, y_pred):
        return np.mean((y_pred - y) ** 2)
    


        
    def fit(self, X, y):
        """
#         Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # Ensure X is a 2D array

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            
        

    
    def predict(self, X):
        """
#         Generates predictions
        
#         Note: should be called after .fit()
        
#         Args:
#             X (array<m,n>): a matrix of floats with 
#                 m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        lin_model = np.dot(X.reshape(-1, 1), self.weights) + self.bias
        return lin_model

