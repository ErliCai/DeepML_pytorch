import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
    i = 0
    result = []
    while len(X) > i:
        X_cut = X[i:i+batch_size]
        if y is not None:
            y_cut = y[i:i+batch_size]
            result.append([X_cut,y_cut])
        else:
            result.append(X_cut)

        i += batch_size

    return result