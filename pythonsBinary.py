import numpy as np


def lenthyBin(number,length):
    binaryNumber=bin(number)[2:]
    if length>=len(binaryNumber):
        NoOfzeros=length-len(binaryNumber)
        lengthadjustedBinaryNumber="0"*NoOfzeros+binaryNumber
    else:
        return binaryNumber
    return lengthadjustedBinaryNumber

def to_one_hot(data_point_index, vocab_size):
    print(vocab_size)
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    
    return temp



print(lenthyBin(5,2))
print(to_one_hot(5,10)[5])