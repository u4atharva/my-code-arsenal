import numpy as np

def PrintZ(input1):
    if int(input1) < 3:
        return 'Invalid Input! Input should be greater than 2.'
    else:
        result = np.eye(input1, dtype=np.int)
        result[[0,-1]] = 1
        return result[::-1].flatten().tolist()

print(PrintZ(6))



