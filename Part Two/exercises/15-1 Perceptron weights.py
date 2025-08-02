import numpy as np

# w0 - 2w1 = 0 
# w0 + 7w2 = 0 

# w0 = 2w1
# w0 = -7w2
# 2w1 = -7w2
# w1 = -7, w2 = 2 
# w0 = 2 * w1 = -14

w1 = -7
w2 = 2
w0 = 2 * w1

weights = np.array([w0, w1, w2])
print("Weights [w0, w1, w2]:", weights)
