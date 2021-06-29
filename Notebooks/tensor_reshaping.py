import numpy as np 

# Criando uma Matriz de 3 linhas e 2 colunas
x = np.array([[0.3, 1.3],[2.0, 3.0], [4.1, 7.2]])
print(x)
print(x.shape)
print(x.reshape((6,1)))
print(x.reshape((2,3)))
print(x.flatten())

# Criando uma Matriz de apenas zeros de shape (300,20)
z = np.zeros((300,20))
# O método tranpose inverte as linhas e colunas
z = np.transpose(z)
print(z.shape)