# Estrutura Matemática das Redes Neurais Artificiais

Compreender o **Deep Learning** requer familiaridade com muitos conceitos matemáticos simples: [tensores](https://en.wikipedia.org/wiki/Tensor), operações de tensores, diferenciação, [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) e assim por diante.

## Neurônio Artificial

Um **neurônio artificial** é uma função matemática concebida como um modelo de neurônios biológicos. Neurônios artificiais são unidades elementares em uma **rede neural artificial**. O neurônio artificial recebe uma ou mais **entradas** (representando potenciais pós-sinápticos excitatórios e potenciais pós-sinápticos inibitórios em dendritos neurais) e os soma para produzir uma **saída** (ou ativação, representando o potencial de ação de um neurônio que é transmitido ao longo de seu axônio). Normalmente, cada entrada é *weighted* separadamente e a soma é passada por uma função não-linear conhecida como **função de ativação**.

A ilustração a seguir apresenta o neurônio artificial e seus diferentes componentes:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/Neuron.png)

Como podemos observar, um neurônio basicamente pega **entradas**, faz algumas cálculos com elas e produz uma **saída**.

A seguir temos um exemplo de um neurônio com apenas duas entradas:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/NeuronZoom.png)

Três etapas estão ocorrendo nessa ilustração:

1. **Vermelho**: Cada **entrada** é multiplicada por um *weight*:

```
x1 -> x1 * w1
x2 -> x2 * x2
```

2. **Verde**: Cada **entrada weighted** é adicionada com um **bias** **b**:

```
(x1 * w1) + (x2 * w2) + b
```

3. **Amarelo**: Finalmente, a soma é passada para uma **função de ativação**:

```
y = f(x1 * w1 + x2 * w2 + b)
```

A função de ativação é usada para transformar uma entrada ilimitada em uma saída que tem uma forma agradável e previsível. Uma função de ativação comumente usada é a [função sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function):

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/Sigmoid.png)

A função sigmoid emite apenas números no intervalo **(0,1)**. Podemos pensar nisso como compactar **(-∞,+∞)** para **(0,1)** - grandes números negativos tornam-se ~0 e grandes números positivos tornam-se ~1.

Outra função de ativação muito comum é a retificadora, definida como a parte positiva de seu argumento:

```
f(x) = max(0,x)
```

A imagem a seguir apresenta o comportamento da função retificadora:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/ReLU.png)

Onde **x** é a entrada para um neurônio, esperamos que qualquer valor positivo seja retornado inalterado, enquanto um valor de entrada de **0** ou um valor negativo seja retornado como o valor **0**. Esta função de ativação foi introduzida pela primeira vez em uma rede dinâmica por **Hahnloser**, em um artigo de 2000 na **Nature**. Com fortes motivações biológicas e justificativas matemáticas, foi demonstrado pela primeira vez em 2011 para permitir um melhor treinamento de **deeper networks**.

### Um Simples Exemplo

Suponha que temos um neurônio de duas entradas que usa a função de ativação sigmoid e tem os seguintes parâmetros:

```
w = [0,1]
b = 4
```

`w = [0,1]` é apenas uma forma de escrever `w1 = 0, w2 = 1` em forma vetorial. Agora, vamos dar ao neurônio uma entrada de `x = [2, 3]`. Usaremos o [produto escalar](https://simple.wikipedia.org/wiki/Dot_product) para escrever os procedimentos de forma mais concisa:

```
(w * x) + b = ((w1 * x1) + (w2 * x2)) + b
(w * x) + b = 0 * 2 + 1 * 3 + 4
(w * x) + b = 7

y = f(w * x + b) = f(7) = 0.999
``` 

O neurônio produz a saída **0.999** fornecidos os inputs `x1 = 2, x2 = 3`. Esse processo de transmitir entradas para obter uma saída é conhecido como **feedforward**.

### Codificando um Neurônio

Para implementar o neurônio, vamos utilizar [NumPy](https://www.numpy.org/), uma biblioteca de computação numérica popular e poderosa para Python, que irá nos ajudar a fazer matemática:

```python
import numpy as np

def sigmoid(x):
    # função de ativação sigmoid: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Produto escalar de inputs e weights e adiciona o bias
        # E então usa a função de ativação sigmoid
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

    def __str__(self):
        return f'Weights: {self.weights} | Bias: {self.bias}'

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)
print(n)

x = np.array([2, 3])                         
print(f'Neuron input = {x}')                 # x1 = 2, x2 = 3
print(f'Neuron output = {n.feedforward(x)}') # 0.9990889488055994
```

Como podemos observar, estamos reproduzindo o mesmo exemplo que fizemos a mão anteriormente, mas dessa vez com Python, o resultado é o mesmo!

## Construindo uma Rede Neural Artificial

Vejamos agora um exemplo concreto de uma rede neural que usa as bibliotecas Python [Tensorflow](https://www.tensorflow.org/) e [Keras](https://keras.io/) para aprender a classificar dígitos manuscritos.

O problema que estamos tentando resolver aqui é classificar imagens em tons de cinza de dígitos escritos à mão (**28×28 pixels**) em suas 10 categorias (**0** a **9**). Usaremos o conjunto de dados [MNIST](http://yann.lecun.com/exdb/mnist/), um clássico na comunidade de machine learning, que existe há quase tanto tempo quanto o próprio campo e tem sido intensamente estudado. É um conjunto de 60.000 imagens de treinamento, além de 10.000 imagens de teste, reunidas pelo National Institute of Standards and Technology (o NIST no MNIST) na década de 1980. Podemos pensar em "resolver" o MNIST como o "Hello World" do Deep Learning - é o que fazemos para verificar se nossos algoritmos estão funcionando conforme o esperado.

A imagem a seguir apresenta exemplos de alguns dígitos encontrados no conjunto de dados MNIST:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/MNISTDigits.png)

**Observação**: No machine learning, uma categoria em um problema de classificação é chamada de **classe**. Os pontos de dados são chamados de **amostras**. A classe associada a uma amostra específica é chamada de rótulo (**label**).

O conjunto de dados MNIST vem pré-carregado no Tensorflow/Keras, na forma de um conjunto de quatro [arrays Numpy](https://numpy.org/doc/stable/reference/generated/numpy.array.html).

```python
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

**train_images** e **train_labels** formam o conjunto de treinamento, os dados com os quais o modelo aprenderá. O desempenho do modelo será então testado no conjunto de teste, **test_images** e **test_labels**.

As imagens são codificadas como arrays Numpy e os rótulos são um array de dígitos, variando de 0 a 9. As imagens e rótulos têm uma correspondência um a um.

Vejamos então os dados de treinamento:

```python
print(train_images.shape) # (60000, 28, 28)
print(len(train_labels)) # 60000
print(train_labels) # array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
```

E também temos os dados de teste:

```python
print(test_images.shape) # (10000, 28, 28)
print(len(test_labels)) # 10000
print(test_labels) # array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
```

O fluxo de trabalho será o seguinte: primeiro, alimentaremos a rede neural com os dados de treinamento, **train_images** e **train_labels**. A rede aprenderá a associar imagens e rótulos. Por fim, pediremos à rede que produza previsões para **test_images** e verificaremos se essas previsões correspondem aos rótulos de **test_labels**.

Vamos então construir a rede neural:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

rede_neural = Sequential()
rede_neural.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
rede_neural.add(Dense(10, activation='softmax'))
```

O bloco de construção principal das redes neurais é a camada (*layer*), um módulo de processamento de dados que você pode considerar um filtro de dados. Alguns dados entram e saem de uma forma mais útil. Especificamente, as camadas extraem **representações** dos dados alimentados nelas - com sorte, representações que são mais significativas para o problema em questão. A maior parte do deep learning consiste em encadear camadas simples que implementarão uma forma de **destilação progressiva de dados**. Um modelo de deep learning é como uma peneira para processamento de dados, feita de uma sucessão de filtros de dados cada vez mais refinados - as camadas (*layers*).

Neste caso, nossa rede consiste em uma sequência de duas camadas densas (*Dense layers*), que são camadas neurais densamente conectadas (também chamadas de *fully connected*). A segunda (e última) camada é uma camada **[softmax](https://en.wikipedia.org/wiki/Softmax_function)** de 10 vias, o que significa que ela retornará um array de 10 pontuações de probabilidade (somando no total 1). Cada pontuação será a probabilidade de que a imagem do dígito atual pertença a uma de nossas classes de 10 dígitos.

Para preparar a rede para o treinamento, precisamos escolher mais três elementos, como parte da etapa de compilação:

- Uma **função Loss**: como a rede será capaz de medir seu desempenho nos dados de treinamento e, portanto, como será capaz de se orientar na direção certa.
- Um **otimizador**: o mecanismo pelo qual a rede se atualiza com base nos dados que vê e em sua função Loss.
- **Métricas** a serem monitoradas durante o treinamento e teste - aqui, vamos nos preocupar apenas com a [accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy) (a fração das imagens que foram classificadas corretamente).

Usaremos o seguinte comando para compilar o nosso modelo:

```python
rede_neural.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
```

Antes do treinamento, iremos pré-processar os dados remodelando-os na forma que a rede espera e dimensionando-os de modo que todos os valores estejam no intervalo **[0, 1]**. Anteriormente, nossas imagens de treinamento, por exemplo, eram armazenadas em um array de forma **(60000, 28, 28)** do tipo **uint8** com valores no intervalo **[0, 255]**. Nós o transformamos em um array **float32** de forma **(60000, 28 * 28)** com valores entre **0** e **1**.

Preparando os dados de imagens para alimentá-los à rede neural:

```python
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
```

Também precisamos codificar categoricamente os rótulos:

```python
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

Agora já estamos prontos para treinar a rede neural, o que em Keras é feito por meio de uma chamada para o método **fit**, ou seja, ajustamos o modelo aos dados de treinamento:

```python
rede_neural.fit(train_images, train_labels, epochs=5, batch_size=128)
```

Duas quantidades são exibidas durante o treinamento: a **Loss** da rede nos dados de treinamento e a **Accuracy** da rede nos dados de treinamento. Rapidamente alcançamos uma accuracy de **0.989** (**98.9%**) nos dados de treinamento. Agora vamos verificar se o modelo tem um bom desempenho no conjunto de teste também:

```python
test_loss, test_acc = rede_neural.evaluate(test_images, test_labels)
print(f'Accuracy de teste: {test_acc*100:.2f}%') # 97.85%
```

A accuracy do conjunto de teste acabou sendo **97.85%**, o que é um pouco menor do que a accuracy do conjunto de treinamento. Essa lacuna entre a accuracy do treinamento e a accuracy do teste é um exemplo de **overfitting**: o fato de que os modelos de machine learning tendem a ter um desempenho pior em novos dados do que em seus dados de treinamento.

Isso conclui nosso primeiro exemplo - acabamos de ver como construir e treinar uma rede neural para classificar dígitos escritos à mão em poucas linhas de código Python.