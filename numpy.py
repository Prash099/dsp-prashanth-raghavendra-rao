import numpy as np

arr1 = np.arange(4, dtype = np.float_).reshape(2, 2)

print('First array:')
print(arr1)

print('\nSecond array:')
arr2 = np.array([12, 12])
print(arr2)

print('\nAdding the two arrays:')
print(np.add(arr1, arr2))

print('\nSubtracting the two arrays:')
print(np.subtract(arr1, arr2))

print('\nMultiplying the two arrays:')
print(np.multiply(arr1, arr2))

print('\nDividing the two arrays:')
print(np.divide(arr1, arr2))
