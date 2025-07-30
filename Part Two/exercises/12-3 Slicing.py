import numpy as np
import matplotlib.pyplot as plt
arr = [[]]
j = -1
for i in range(1,55):
    if(i % 5 == 0):
        arr.append([])
        j += 1

    arr[j].append(i)
arr.remove(arr[-1])
print(arr)
mat = np.array(arr)
print(mat)
   