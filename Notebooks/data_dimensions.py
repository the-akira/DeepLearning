import numpy as np 

# Scalars (0D Tensors)
x = np.array(12)
print(f'x = {x}, dimensões = {x.ndim}\n')

# Vetores (1D Tensors)
y = np.array([12, 3, 6, 14])
print(f'y = {y}, dimensões = {y.ndim}\n')

# Matrizes (2D Tensors)
k = np.array([[5, 78, 2, 34, 0], 
              [6, 79, 3, 35, 1], 
              [7, 80, 4, 36, 2]])
print(f'k = {k}, dimensões = {k.ndim}\n')

# Tensors (3D)
z = np.array([[[5, 33, 2, 34, 0],
               [6, 32, 3, 35, 1],
               [19, 22, 7, 32, 3]],
              [[24, 59, 5, 29, 1],
               [20, 18, 4, 28, 2],
               [21, 22, 2, 28, 1]],
              [[22, 28, 3, 28, 2],
               [17, 12, 8, 28, 3],
               [3, 29, 9, 28, 0]]])
print(f'z = {z}, dimensões = {z.ndim}')