import numpy as np

def shuffle_and_split(images, labels, valid_index):
    # generate indices array
    indices = np.arange(valid_index)

    # shuffling indices 
    np.random.shuffle(indices)

    # slice data 80% training, 10% valid and test
    train_split = int(0.8 * valid_index)
    val_split = int(0.9 * valid_index)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    # extract data
    X_train, y_train = images[train_indices], labels[train_indices]
    X_val, y_val = images[val_indices], labels[val_indices]
    X_test, y_test = images[test_indices], labels[test_indices]

    print(f'Shuffled and Split Images')
    return X_train, y_train, X_val, y_val, X_test, y_test
