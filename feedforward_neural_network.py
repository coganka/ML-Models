import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predictions = self.sigmoid(self.output_layer_input)

        return self.predictions

    def backward(self, inputs, targets, learning_rate):
        output_error = targets - self.predictions
        output_delta = output_error * self.sigmoid_derivative(self.predictions)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        self.weights_hidden_output += learning_rate * self.hidden_layer_output.T.dot(output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weights_input_hidden += learning_rate * inputs.T.dot(hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward(inputs)
            self.backward(inputs, targets, learning_rate)

            mse = np.mean((predictions - targets) ** 2)
            print(f"Epoch {epoch+1}/{epochs}, Mean Squared Error: {mse}")


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

neural_network = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
neural_network.train(inputs, targets, epochs=5000, learning_rate=0.1)

new_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = neural_network.forward(new_data)

print("Predictions after training:")
print(predictions)
