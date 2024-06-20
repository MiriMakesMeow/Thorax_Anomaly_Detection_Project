from NeuralNetwork.network import NetworkMatrix

def train(network, X_train, y_train, epochs, learning_rate):
    for epoch in range(epochs):
        total_loss = 0
        for X, y in zip(X_train, y_train):
            output = network.query(X) # forward-pass
            loss = network.train(targets=y, learning_rate=learning_rate) # backpropagation
            total_loss += loss
        average_loss = total_loss / len(X_train)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Average Loss: {average_loss}')
    print(f'Trained Network')
    return output
