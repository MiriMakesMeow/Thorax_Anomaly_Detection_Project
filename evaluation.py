import numpy as np

def evaluate(network, X_val, y_val):
    correct_predictions = 0
    total_examples = len(X_val)

    for i in range(total_examples):
        X = np.expand_dims(X_val[i], axis=0)
        y_true = y_val[i]
        prediction = network.query(X)

        if prediction > 0.4:
            predicted_class = 1
        else:
            predicted_class = 0

        is_correct = predicted_class == y_true
        correct_predictions += int(is_correct)

        print(f'Predicted: {prediction}, Truth: {y_true}')

    accuracy = correct_predictions / len(X_val)
    
    print(f'Accuracy: {accuracy}')
    