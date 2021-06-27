from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# Carregando os dados e separando-os em conjuntos de treinamento e teste
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Imprimindo informações sobre os dados de treinamento
print(f'Shape das imagens de treinamento: {train_images.shape}')
print(f'Dimensões das imagens de treinamento: {train_images.ndim}')
print(f'Tipo de dados das imagens de treinamento: {train_images.dtype}')
print(f'Quantidade de labels de treinamento: {len(train_labels)}')
print(f'Labels de treinamento [0-9]: {train_labels}\n')

# Imprimindo informações sobre os dados de teste
print(f'Shape das imagens de teste: {test_images.shape}')
print(f'Quantidade de labels de teste: {len(test_images)}')
print(f'Labels de teste [0-9]: {test_labels}')

# Apresentando um dígito
dígito = train_images[4]
plt.imshow(dígito, cmap=plt.cm.binary)
plt.show()

# Construindo a Rede Neural (Modelo de Classificação de Dígitos)
rede_neural = Sequential()
rede_neural.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
rede_neural.add(Dense(10, activation='softmax'))
rede_neural.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Preparando os dados de imagens para alimentá-los à rede neural
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Preparando as labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Treinando o modelo
rede_neural.fit(train_images, train_labels, epochs=5, batch_size=128)

# Checando a perfomance do modelo
test_loss, test_acc = rede_neural.evaluate(test_images, test_labels)
print(f'Accuracy de teste: {test_acc*100:.2f}%')

# Realizando uma previsão
idx = np.random.randint(0,40)
número = test_images[idx]
print(f'Valor correto: {np.argmax(test_labels[idx], axis=-1)}')
previsão = np.argmax(rede_neural.predict(número.reshape(1, 28 * 28)), axis=-1)
print(f'Valor previsto: {previsão[0]}')