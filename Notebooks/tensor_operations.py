import numpy as np

def relu(x):
    assert len(x.shape) == 2 # x é um Tensor NumPy 2D

    x = x.copy() # Evitar sobrescrever o Tensor de input
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

x = np.array([[0.5,0.0,-3.0],[2.3,5.9,-1.3]])
print("Função ReLU")
print(relu(x))
print("Função ReLU")
print(np.maximum(x,0.0))

def add(x, y):
    assert len(x.shape) == 2 # x e y são Tensors NumPy 2D
    assert x.shape == y.shape

    x = x.copy() # Evitar sobrescrever o Tensor de input
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
        	x[i, j] += y[i, j]
    return x 

a = np.array([[1,2,3],[6,7,8]])
b = np.array([[4,5,9],[1,3,3]])
print("a + b")
c = add(a,b)
print(c)

print("a + b")
d = a + b
print(d)

def add_matrix_and_vector(x, y):
    assert len(x.shape) == 2 # x é um Tensor NumPy 2D
    assert len(y.shape) == 1 # y é um Vetor NumPy
    assert x.shape[1] == y.shape[0]

    x = x.copy() # Evitar sobrescrever o Tensor de input
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x 

print("M + N")
m = np.array([[8,4,9],[4,7,2]])
n = np.array([1,2,3])
print(add_matrix_and_vector(m,n))

# X é um tensor aleatório com shape (64, 3, 32, 10)
X = np.random.random((64, 3, 32, 10))
# Y é um tensor aleatório com shape (32, 10) 
Y = np.random.random((32, 10))
# O output Z é um tensor com shape (32, 10)
Z = np.maximum(X, Y)

print("Produto escalar m . n (Matriz . Vetor)")
print(np.dot(m,n))

def vector_dot(x, y):
    assert len(x.shape) == 1 # x é um vetor NumPy
    assert len(y.shape) == 1 # y é um vetor NumPy
    assert x.shape[0] == y.shape[0]

    z = 0 
    for i in range(x.shape[0]):
    	z += x[i] * y[i]
    return z

print("Produto escalar p . q (Vetor . Vetor)")
p = np.array([1,2,3,4,5])
q = np.array([3,4,7,8,9])
print(vector_dot(p,q))

def matrix_vector_dot(x, y):
    assert len(x.shape) == 2 # x é uma matriz NumPy
    assert len(y.shape) == 1 # y é um vetor Numpy
    assert x.shape[1] == y.shape[0] # A dimensão 1 de x deve ser a mesma da dimensão 0 de y

    z = np.zeros(x.shape[0]) # Essa operação retorna um vetor de 0's com a mesma shape de y
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
        	z[i] += x[i, j] * y[j]
    return z

print("Produto escalar M . N (Matriz . Vetor)")
M = np.array([[8,4,9],[4,7,2]])
N = np.array([1,2,3])
print(matrix_vector_dot(M,N))

def matriz_vetor_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = vector_dot(x[i,:], y)
    return z

print("Produto escalar M . N (Matriz . Vetor)")
M = np.array([[8,4,9],[4,7,2]])
N = np.array([1,2,3])
print(matriz_vetor_dot(M,N))

def matrix_dot(x, y):
    assert len(x.shape) == 2 # x é uma matriz NumPy
    assert len(y.shape) == 2 # y é uma matriz NumPy
    assert x.shape[1] == y.shape[0] # A dimensão 1 de x deve ser a mesma dimensão 0 de y

    z = np.zeros((x.shape[0], y.shape[1])) # Essa operação retorna uma matriz de 0's com uma shape específica
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = vector_dot(row_x, column_y)
    return z

print('Produto escalar A . B (Matriz . Matriz)')
A = np.array([[1,2,3],[6,7,8],[4,2,1]])
B = np.array([[4,5,7],[5,2,1],[9,4,3]])
print(matrix_dot(A,B))