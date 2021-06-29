# Estrutura Matemática das Redes Neurais Artificiais

Compreender o **Deep Learning** requer familiaridade com muitos conceitos matemáticos simples: [tensors](https://en.wikipedia.org/wiki/Tensor), operações de tensors, diferenciação, [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) e assim por diante.

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

## Representações de Dados para Redes Neurais Artificiais

No exemplo anterior, começamos com dados armazenados em arrays NumPy multidimensionais, também chamadas de **tensors**. Em geral, todos os sistemas de machine learning atuais usam tensors como sua estrutura de dados básica. Os tensors são fundamentais para o campo, tão fundamentais que o TensorFlow do Google recebeu o nome deles. Então, o que é um tensor?

Em seu núcleo, um tensor é um contêiner de dados - quase sempre dados numéricos. Você já deve estar familiarizado com matrizes, que são tensors 2D: tensors são uma generalização de matrizes para um número arbitrário de dimensões (observe que, no contexto de tensors, uma **dimensão** é freqüentemente chamada de **eixo**).

A ilustração a seguir mostra diferentes dimensões de dados:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/DataDimensions.png)

### Escalares (Tensors 0D)

Um tensor que contém apenas um número é chamado de escalar (ou tensor escalar, ou tensor 0-dimensional ou tensor 0D). No NumPy, um número **float32** ou **float64** é um tensor escalar (ou array escalar). Você pode exibir o número de eixos de um tensor NumPy por meio do atributo **ndim**; um tensor escalar tem 0 eixos (`ndim == 0`). O número de eixos de um tensor também é chamado de rank. A seguir temos um exemplo de escalar NumPy:

```python
>>> import numpy as np
>>> x = np.array(13)
>>> print(f'x = {x}, dimensões = {x.ndim}')
# x = 13, dimensões = 0
```

### Vetores (Tensors 1D)

Um array de números é chamado de vetor ou tensor 1D. Diz-se que um tensor 1D tem exatamente um eixo. A seguir está um vetor NumPy:

```python
>>> y = np.array([12, 3, 6, 14, 17])
>>> print(f'y = {y}, dimensões = {y.ndim}')
# y = [12  3  6 14 17], dimensões = 1
```

Esse vetor possui cinco entradas e, portanto, é chamado de **vetor 5-dimensional**. Não confunda um vetor 5D com um tensor 5D! Um vetor 5D tem apenas um eixo e cinco dimensões ao longo de seu eixo, enquanto um tensor 5D tem cinco eixos (e pode ter qualquer número de dimensões ao longo de cada eixo). A **dimensionalidade** pode denotar o número de entradas ao longo de um eixo específico (como no caso de nosso vetor 5D) ou o número de eixos em um tensor (como um tensor 5D), o que pode ser confuso às vezes. No último caso, é tecnicamente mais correto falar sobre um tensor de rank 5 (o rank de um tensor sendo o número de eixos), mas a notação ambígua do tensor 5D é comum de qualquer maneira.

### Matrizes (Tensors 2D)

Um array de vetores é uma matriz ou tensor 2D. Uma matriz tem dois eixos (geralmente referidos a **linhas** e **colunas**). Você pode interpretar visualmente uma matriz como uma grade retangular de números. Esta é uma matriz Numpy:

```python
>>> k = np.array([[5, 78, 2, 34, 0], 
                 [6, 79, 3, 35, 1], 
                 [7, 80, 4, 36, 2]])
>>> print(f'k = {k}, dimensões = {k.ndim}\n')
# k = [[ 5 78  2 34  0]
#      [ 6 79  3 35  1]
#      [ 7 80  4 36  2]], dimensões = 2
```

As entradas do primeiro eixo são chamadas de **linhas** e as entradas do segundo eixo são chamadas de **colunas**. No exemplo anterior, `[5, 78, 2, 34, 0]` é a primeira linha de **k** e `[5, 6, 7]` é a primeira coluna.

### Tensors 3D e Tensors de Dimensões Superiores

Se você empacotar essas matrizes em um novo array, obterá um **tensor 3D**, que pode ser interpretado visualmente como um cubo de números. A seguir está um tensor Numpy 3D:

```python
>>> z = np.array([[[5, 33, 2, 34, 0],
                  [6, 32, 3, 35, 1],
                  [19, 22, 7, 32, 3]],
                 [[24, 59, 5, 29, 1],
                  [20, 18, 4, 28, 2],
                  [21, 22, 2, 28, 1]],
                 [[22, 28, 3, 28, 2],
                  [17, 12, 8, 28, 3],
                  [3, 29, 9, 28, 0]]])
>>> print(f'z = {z}, dimensões = {z.ndim}')
# z = [[[ 5 33  2 34  0]
#       [ 6 32  3 35  1]
#       [19 22  7 32  3]]
#
#     [[24 59  5 29  1]
#      [20 18  4 28  2]
#      [21 22  2 28  1]]
#
#     [[22 28  3 28  2]
#      [17 12  8 28  3]
#      [ 3 29  9 28  0]]], dimensões = 3
```

Ao empacotar tensors 3D em um array, você pode criar um tensor 4D e assim por diante. No deep learning, você geralmente manipulará tensors que vão de 0D a 4D, embora você possa ir até 5D se for processar dados de vídeo.

### Atributos Chaves

Um tensor é definido por três atributos principais:

- **Número de eixos** (Rank): Por exemplo, um tensor 3D tem três eixos e uma matriz tem dois eixos. Isso também é chamado de **ndim** do tensor em bibliotecas Python, como NumPy.
- **Forma** (Shape): Esta é uma tupla de inteiros que descreve quantas dimensões o tensor tem ao longo de cada eixo. Por exemplo, o exemplo de matriz que vimos anteriormente tem forma **(3, 5)** e o exemplo de tensor 3D tem forma **(3, 3, 5)**. Um vetor tem uma forma com um único elemento, como **(5,)**, enquanto um escalar tem uma forma vazia, **()**.
- **Tipo de Dados** (geralmente chamado de **dtype** em bibliotecas Python): Este é o tipo de dados contidos no tensor; por exemplo, o tipo de um tensor pode ser **float32**, **uint8**, **float64** e assim por diante. Em raras ocasiões, você pode ver um tensor **char**. Observe que os tensors de string não existem no NumPy (ou na maioria das outras bibliotecas), porque os tensors vivem em segmentos de memória contíguos pré-alocados: e as strings, sendo de comprimento variável, impediriam o uso desta implementação.

Para tornar isso mais concreto, vamos olhar novamente para os dados que processamos no exemplo MNIST. Primeiro, carregamos o conjunto de dados MNIST:

```python
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

A seguir, exibimos o número de eixos do tensor **train_images**, acessando o atributo **ndim**:

```python
print(train_images.ndim) # 3
```

Para descobrir sua forma (shape), podemos acessar o atributo **shape**:

```python
print(train_images.shape) # (60000, 28, 28)
```

E este é seu tipo de dados, o atributo **dtype**:

```python
print(train_images.dtype) # uint8
```

Portanto, o que temos aqui é um tensor 3D de inteiros de 8 bits. Mais precisamente, é uma matriz de **60.000** matrizes de **28×28** inteiros. Cada uma dessas matrizes é uma imagem em tons de cinza, com coeficientes entre **0** e **255**.

Vamos exibir o quarto dígito neste tensor 3D, usando a biblioteca [Matplotlib](https://matplotlib.org/) (parte do pacote científico padrão do Python):

```python
import matplotlib.pyplot as plt

dígito = train_images[4]
plt.imshow(dígito, cmap=plt.cm.binary)
plt.show()
```

Nos será apresentada a quarta amostra em nosso conjunto de dados:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/MNISTDigit.png)

### Manipulando Tensors em NumPy

No exemplo anterior, selecionamos um dígito específico ao longo do primeiro eixo usando a sintaxe `train_images[i]`. A seleção de elementos específicos em um tensor é chamada de *slicing* de tensor. Vejamos as operações de *slicing* de tensor que você pode fazer em arrays Numpy.

O exemplo a seguir seleciona os dígitos **#10** a **#100** (**#100** não está incluído) e os coloca em uma array de forma **(90, 28, 28)**:

```python
fatia = train_images[10:100]
print(fatia.shape) # (90, 28, 28)
```

É equivalente a esta notação mais detalhada, que especifica um índice inicial e um índice final para a fatia ao longo de cada eixo do tensor. Observe que: é equivalente a selecionar todo o eixo:

```python
fatia = train_images[10:100, :, :] # Equivalente ao exemplo anterior
print(fatia.shape) # (90, 28, 28)

fatia = train_images[10:100, 0:28, 0:28] # Também equivalente ao exemplo anterior
print(fatia.shape) # (90, 28, 28)
```

Em geral, você pode selecionar entre quaisquer dois índices ao longo de cada eixo do tensor. Por exemplo, para selecionar **14×14** pixels no canto inferior direito de todas as imagens, faça o seguinte:

```python
fatia = train_images[:, 14:, 14:]
print(fatia.shape) # (60000, 14, 14)
```

Também é possível usar índices negativos. Muito parecido com os índices negativos nas listas do Python, eles indicam uma posição relativa ao final do eixo atual. Para cortar as imagens em fatias de **14×14** pixels centralizadas no meio, faça o seguinte:

```python
fatia = train_images[:, 7:-7, 7:-7]
print(fatia.shape) # (60000, 14, 14)
```

### A Noção de Batches de Dados

Em geral, o primeiro eixo (eixo **0**, porque a indexação começa em **0**) em todos os tensors de dados que você encontrará no deep learning será o eixo de amostras (às vezes chamado de dimensão de amostras). No exemplo MNIST, as amostras são imagens de dígitos.

Além disso, os modelos de deep learning não processam um conjunto de dados inteiro de uma vez; em vez disso, eles dividem os dados em pequenos **batches** (lotes). Concretamente, aqui está um **batch** de nossos dígitos MNIST, com tamanho de **batch** de **128**:

```python
batch = train_images[:128]
print(batch.shape) # (128, 28, 28)
```

E aqui está o próximo **batch**:

```python
batch = train_images[128:256]
```

E o enésimo (neste exemplo, 100º) **batch**:

```python
n = 100
batch = train_images[128 * n:128 * (n + 1)]
```

Ao considerar esse batch tensor, o primeiro eixo (eixo 0) é chamado de *batch axis* ou *batch dimension*. Este é um termo que você encontrará com frequência ao usar Tensorflow/Keras e outras bibliotecas de deep learning.

### Exemplos de Tensors de Dados do Mundo Real

Vamos tornar os tensors de dados mais concretos com alguns exemplos semelhantes ao que você encontrará na realidade. Os dados que você manipulará quase sempre se enquadrarão em uma das seguintes categorias:

- **Dados vetoriais**: tensors 2D de forma (amostras, features).
- **Dados de série temporal** ou **dados de sequência**: tensors 3D de forma (amostras, passos de tempo, features).
- **Imagens**: tensors 4D de forma (amostras, altura, largura, canais) ou (amostras, canais, altura, largura).
- **Vídeo**: tensors 5D de forma (amostras, frames, altura, largura, canais) ou (amostras, frames, canais, altura, largura).

#### Dados Vetoriais

Este é o caso mais comum. Em tal conjunto de dados, cada ponto de dados único pode ser codificado como um vetor e, assim, um batch de dados será codificado como um tensor 2D (ou seja, um array de vetores), onde o primeiro eixo é o **eixo das amostras** e o segundo eixo é o **eixo de features**. Vejamos dois exemplos:

- Um conjunto de dados de pessoas, onde consideramos a idade, o código postal e a renda de cada pessoa. Cada pessoa pode ser caracterizada como um vetor de 3 valores e, portanto, um conjunto de dados inteiro de 100.000 pessoas pode ser armazenado em um tensor 2D de forma **(100000, 3)**.
- Um conjunto de dados de documentos de texto, onde representamos cada documento pela contagem de quantas vezes cada palavra aparece nele (em um dicionário de 20.000 palavras comuns). Cada documento pode ser codificado como um vetor de 20.000 valores (uma contagem por palavra no dicionário) e, portanto, um conjunto de dados inteiro de 500 documentos pode ser armazenado em um tensor de forma **(500, 20000)**.

#### Dados de Série Temporal ou Dados de Sequência

Sempre que o tempo importa em seus dados (ou a noção de ordem de sequência), faz sentido armazená-lo em um tensor 3D com um eixo de tempo explícito. Cada amostra pode ser codificada como uma sequência de vetores (um tensor 2D) e, portanto, um batch de dados será codificado como um tensor 3D, como mostrado na figura a seguir:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/TimeSeriesData.png)

O eixo do tempo é sempre o segundo eixo (eixo do índice **1**), por convenção. Vejamos alguns exemplos:

- Um conjunto de dados de preços de ações. A cada minuto, armazenamos o preço atual da ação, o preço mais alto no minuto anterior e o preço mais baixo no minuto anterior. Assim, cada minuto é codificado como um vetor 3D, um dia inteiro de negociação é codificado como um tensor 2D de forma **(390, 3)** (há 390 minutos em um dia de negociação) e 250 dias de dados podem ser armazenados em um tensor 3D de forma **(250, 390, 3)**. Aqui, cada amostra valeria um dia de dados.
- Um conjunto de dados de tweets, onde codificamos cada tweet como uma sequência de 280 caracteres de um alfabeto de 128 caracteres exclusivos. Nessa configuração, cada caractere pode ser codificado como um vetor binário de tamanho 128 (um vetor composto apenas por zeros, exceto por uma entrada 1 no índice correspondente ao caractere). Então, cada tweet pode ser codificado como um tensor de forma 2D **(280, 128)** e um conjunto de dados de 1 milhão de tweets pode ser armazenado em um tensor de forma **(1000000, 280, 128)**.

#### Dados de Imagens

As imagens normalmente têm três dimensões: **altura**, **largura** e **profundidade de cor**. Embora as imagens em tons de cinza (como nossos dígitos MNIST) tenham apenas um único canal de cor e possam, portanto, ser armazenadas em tensors 2D, por convenção os tensors de imagem são sempre 3D, com um canal de cor unidimensional para imagens em tons de cinza. Um batch de 128 imagens em tons de cinza de tamanho **256×256** pode, portanto, ser armazenado em um tensor de forma **(128, 256, 256, 1)** e um batch de 128 imagens coloridas pode ser armazenado em um tensor de forma **(128, 256, 256, 3)**.

A figura a seguir ilustra um tensor de dados 4D de imagem:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/ImageData.png)

Existem duas convenções para formatos de tensors de imagens: a convenção channels-last (usada pelo Tensorflow) e a convenção channels-first (usada por Theano). O framework de machine learning Tensorflow, do Google, coloca o eixo de profundidade de cor no final: (samples, altura, largura, color_depth). Enquanto isso, Theano posiciona o eixo de profundidade de cor logo após o eixo do batch: (samples, color_depth, altura, largura). Com a convenção de Theano, os exemplos anteriores seriam **(128, 1, 256, 256)** e **(128, 3, 256, 256)**.

#### Dados de Vídeo

Os dados de vídeo são um dos poucos tipos de dados do mundo real para os quais você precisará de tensors 5D. Um vídeo pode ser entendido como uma sequência de frames (quadros), cada frame sendo uma imagem colorida. Como cada frame pode ser armazenado em um tensor 3D (altura, largura, color_depth), uma sequência de frames pode ser armazenada em um tensor 4D (frames, height, width, color_depth) e, portanto, um batch de diferentes vídeos pode ser armazenado em um tensor 5D de forma (amostras, frames, altura, largura, color_depth).

Por exemplo, um clipe de vídeo do YouTube de **144×256** de 60 segundos com amostragem de 4 frames por segundo teria 240 frames. Um batch de quatro desses videoclipes seria armazenado em um tensor de forma **(4, 240, 144, 256, 3)**. Isso é um total de 106.168.320 valores! Se o **dtype** do tensor fosse float32, cada valor seria armazenado em 32 bits, de modo que o tensor representaria 405 MB. Os vídeos que você encontra na vida real são muito mais leves, porque não são armazenados em float32 e são normalmente compactados por um grande fator (como no formato **MPEG**).