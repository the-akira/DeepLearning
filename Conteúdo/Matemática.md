# Estrutura Matemática das Redes Neurais Artificiais

Compreender o **Deep Learning** requer familiaridade com muitos conceitos matemáticos simples: [tensors](https://en.wikipedia.org/wiki/Tensor), operações de tensors, diferenciação, [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) e assim por diante.

## Neurônio Artificial

Um **neurônio artificial** é uma função matemática concebida como um modelo de neurônios biológicos. Neurônios artificiais são unidades elementares em uma **rede neural artificial**. O neurônio artificial recebe uma ou mais **entradas** (representando potenciais pós-sinápticos excitatórios e potenciais pós-sinápticos inibitórios em dendritos neurais) e os soma para produzir uma **saída** (ou ativação, representando o potencial de ação de um neurônio que é transmitido ao longo de seu axônio). Normalmente, cada entrada é *weighted* separadamente e a soma é passada por uma função não-linear conhecida como **função de ativação**.

A ilustração a seguir apresenta o neurônio artificial e seus diferentes componentes:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/Neuron.png)

Como podemos observar, um neurônio basicamente pega **entradas**, faz alguns cálculos com elas e produz uma **saída**.

A seguir temos um exemplo de um neurônio com apenas duas entradas:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/NeuronZoom.png)

Três etapas estão ocorrendo nessa ilustração:

1. **Vermelho**: Cada **entrada** é multiplicada por um *weight*:

```
x1 -> x1 * w1
x2 -> x2 * x2
```

2. **Verde**: Cada **entrada weighted** é adicionada, junto com um termo **bias** **b**:

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

Por exemplo, um clipe de vídeo do YouTube de **144×256** de 60 segundos com amostragem de 4 frames por segundo teria 240 frames. Um batch de quatro desses videoclipes seria armazenado em um tensor de forma **(4, 240, 144, 256, 3)**. Isso é um total de 106.168.320 valores! Se o **dtype** do tensor fosse **float32**, cada valor seria armazenado em 32 bits, de modo que o tensor representaria 405 MB. Os vídeos que você encontra na vida real são muito mais leves, porque não são armazenados em **float32** e são normalmente compactados por um grande fator (como no formato **MPEG**).

## As Engrenagens das Redes Neurais: Operações de Tensors

Assim como qualquer programa de computador pode ser reduzido a um pequeno conjunto de operações binárias em entradas binárias (**AND**, **OR**, **NOR** e assim por diante), todas as transformações aprendidas por deep neural networks podem ser reduzidas a um punhado de operações de tensor aplicadas a tensors de dados numéricos. Por exemplo, é possível adicionar tensors, multiplicar tensors e assim por diante.

Em nosso exemplo inicial, estávamos construindo nossa rede empilhando camadas densas (*Dense Layers*) umas sobre as outras. Uma instância da Keras layer se parece com isto:

```python
rede_neural.add(Dense(512, activation='relu'))
```

Essa camada pode ser interpretada como uma função, que recebe como entrada um tensor 2D e retorna outro tensor 2D - uma nova representação para o tensor de entrada. Especificamente, a função é a seguinte (onde **W** é um tensor 2D e **b** é um vetor, ambos atributos da camada):

```python
output = relu(dot(W, input) + b)
```

Perceba que temos três operações de tensor aqui: um produto escalar (**dot**) entre o tensor de entrada e um tensor denominado **W**; uma adição (**+**) entre o tensor 2D resultante e um vetor **b**; e, finalmente, uma operação **relu**: `relu(x)` é o mesmo que `max(x, 0)`.

### Operações Element-Wise

A operação **relu** e a **adição** são operações *element-wise*: operações que são aplicadas independentemente a cada entrada nos tensors sendo considerados. Isso significa que essas operações são altamente receptivas a implementações massivamente paralelas (implementações vetorizadas, um termo que vem da arquitetura do supercomputador de processador vetorial do período 1970-1990). Se você quiser escrever uma implementação Python ingênua de uma operação *element-wise*, use um loop **for**, como nesta implementação ingênua de uma operação **relu** *element-wise*:

```python
def relu(x):
    assert len(x.shape) == 2 # x é um Tensor NumPy 2D

    x = x.copy() # Evitar sobrescrever o Tensor de input
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

x = np.array([[0.5,0.0,-3.0],[2.3,5.9,-1.3]])
print(relu(x))
```

Fazemos o mesmo para adição:

```python
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
c = add(a,b)
print(c)
```

Seguindo o mesmo princípio, você pode fazer multiplicação, subtração e assim por diante.

Na prática, ao lidar com arrays Numpy, essas operações estão disponíveis como funções Numpy integradas e bem otimizadas, que delegam o trabalho pesado a uma implementação de Basic Linear Algebra Subprograms (BLAS). BLAS são rotinas de manipulação de tensors eficientes, altamente paralelas e de baixo nível que são normalmente implementadas em Fortran ou C.

Portanto, no Numpy, você pode fazer a seguinte operação *element-wise* e será extremamente rápido:

```python
print(np.maximum(x,0.0))
d = a + b
print(d)
```

### Broadcasting

Nossa implementação ingênua anterior **add** suporta apenas a adição de tensors 2D com formas idênticas. Mas na *Dense Layer* introduzida anteriormente, adicionamos um tensor 2D com um vetor. O que acontece com a adição quando as formas dos dois tensors sendo adicionados diferem?

Quando possível, e se não houver ambigüidade, o tensor menor será *broadcasted* para coincidir com a forma do tensor maior. *Broadcasting* consiste em duas etapas:

- Os eixos (chamados *broadcast axes*) são adicionados ao tensor menor para corresponder ao **ndim** do tensor maior.
- O tensor menor é repetido ao longo desses novos eixos para coincidir com a forma completa do tensor maior.

Vejamos um exemplo concreto. Considere **X** com forma **(32, 10)** e **y** com forma **(10,)**. Primeiro, adicionamos um primeiro eixo vazio a **y**, cuja forma se torna **(1, 10)**. Em seguida, repetimos **y** 32 vezes ao longo desse novo eixo, de modo que terminamos com um tensor **Y** com forma **(32, 10)**, onde `Y[i,:] == y` para **i** no intervalo **(0, 32)**. Neste ponto, podemos prosseguir para adicionar **X** e **Y**, porque eles têm a mesma forma.

Em termos de implementação, nenhum novo tensor 2D é criado, porque isso seria terrivelmente ineficiente. A operação de repetição é inteiramente virtual: ela acontece no nível algorítmico e não no nível da memória. Mas pensar no vetor sendo repetido 10 vezes ao longo de um novo eixo é um modelo mental útil. Esta é a aparência de uma implementação ingênua:

```python
def add_matrix_and_vector(x, y):
    assert len(x.shape) == 2 # x é um Tensor NumPy 2D
    assert len(y.shape) == 1 # y é um Vetor NumPy
    assert x.shape[1] == y.shape[0]

    x = x.copy() # Evitar sobrescrever o Tensor de input
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x 

m = np.array([[8,4,9],[4,7,2]])
n = np.array([1,2,3])
print(add_matrix_and_vector(m,n))
```

Com broadcasting, você geralmente pode aplicar operações *element-wise* a dois tensors se um tensor tiver forma **(a, b, ... n, n + 1, ... m)** e o outro tiver forma **(n, n + 1, ... m)**. O broadcasting acontecerá automaticamente para os eixos de **a** até **n - 1**.

O exemplo a seguir aplica a operação **maximum** *element-wise* a dois tensors de formas diferentes por meio de broadcasting:

```python
# X é um tensor aleatório com shape (64, 3, 32, 10)
X = np.random.random((64, 3, 32, 10))
# Y é um tensor aleatório com shape (32, 10) 
Y = np.random.random((32, 10))
# O output Z é um tensor com shape (32, 10)
Z = np.maximum(X, Y)
```

### Tensor dot

A operação **dot**, também chamada de produto de tensor (não deve ser confundida com um produto *element-wise*) é a operação de tensor mais comum e mais útil. Ao contrário das operações *element-wise*, ela combina entradas nos tensors de entrada.

Um produto *element-wise* é feito com o operador `*` em Numpy, Keras, Theano e TensorFlow. A operação **dot** em NumPy é feita com o método **dot**:

```python
print(np.dot(m,n))
```

Em notação matemática, você notaria a operação com um ponto (`.`):

```
k = m . n
```

Matematicamente, o que a operação **dot** faz? Vamos começar com o produto escalar (*dot product*) de dois vetores **p** e **q**. Ele é calculado da seguinte forma:

```python
def vector_dot(x, y):
    assert len(x.shape) == 1 # x é um vetor NumPy
    assert len(y.shape) == 1 # y é um vetor NumPy
    assert x.shape[0] == y.shape[0]

    z = 0 
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

p = np.array([1,2,3,4,5])
q = np.array([3,4,7,8,9])
print(vector_dot(p,q))
```

Você deve ter notado que o produto escalar entre dois vetores é um escalar e que apenas vetores com o mesmo número de elementos são compatíveis para um produto escalar.

Você também pode pegar o produto escalar entre uma matriz **x** e um vetor **y**, que retorna um vetor onde os coeficientes são os produtos escalares entre **y** e as linhas de **x**. Você o implementa da seguinte maneira:

```python
def matrix_vector_dot(x, y):
    assert len(x.shape) == 2 # x é uma matriz NumPy
    assert len(y.shape) == 1 # y é um vetor Numpy
    assert x.shape[1] == y.shape[0] # A dimensão 1 de x deve ser a mesma da dimensão 0 de y

    z = np.zeros(x.shape[0]) # Essa operação retorna um vetor de 0's com a mesma shape de y
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z

x = np.array([[8,4,9],[4,7,2]])
y = np.array([1,2,3])
print(matrix_vector_dot(x,y))
```

Você também pode reutilizar o código que escrevemos anteriormente, que destaca a relação entre um produto de vetor-matriz e um produto de vetor:

```python
def matriz_vetor_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = vector_dot(x[i,:], y)
    return z
```

Observe que, assim que um dos dois tensors tem uma **ndim** maior que 1, o **dot** não é mais simétrico, o que significa que o **dot(x, y)** não é o mesmo que o **dot(y, x)**.

Obviamente, um produto escalar generaliza para tensors com um número arbitrário de eixos. As aplicações mais comuns podem ser o produto escalar entre duas matrizes. Você pode obter o produto escalar de duas matrizes **x** e **y** (**dot(x, y)**) se e somente se `x.shape[1] == y.shape[0]`. O resultado é uma matriz com forma (`x.shape[0], y.shape[1]`), onde os coeficientes são os produtos do vetor entre as linhas de **x** e as colunas de **y**. Aqui está a implementação ingênua:

```python
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

x = np.array([[1,2,3],[6,7,8],[4,2,1]])
y = np.array([[4,5,7],[5,2,1],[9,4,3]])
z = matrix_dot(A,B)
print(z)
```

Para entender a compatibilidade da forma (*shape*) do produto escalar (*dot product*), ajuda a visualizar os tensors de entrada e saída alinhando-os conforme mostrado na figura a seguir:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/MatrixDotProduct.png)

**x**, **y** e **z** são representados como retângulos (caixas de coeficientes). Como as linhas de **x** e as colunas de **y** devem ter o mesmo tamanho, segue-se que a largura de **x** deve corresponder à altura de **y**.

De forma mais geral, você pode pegar o produto escalar entre tensors de dimensões superiores, seguindo as mesmas regras de compatibilidade de forma descritas anteriormente para o caso 2D:

```
(a, b, c, d) . (d,) -> (a, b, c)

(a, b, c, d) . (d, e) -> (a, b, c, e)
```

E assim por diante.

### Tensor Reshaping

Um terceiro tipo de operação de tensor que é essencial entender é a remodelagem (*reshaping*) do tensor. Embora não tenha sido usado nas *Dense layers* em nosso primeiro exemplo de rede neural, nós o usamos quando pré-processamos os dados de dígitos antes de alimentá-los em nossa rede:

```python
train_images = train_images.reshape((60000, 28 * 28))
```

Reshaping de um tensor significa reorganizar suas linhas e colunas para corresponder a uma forma de destino. Naturalmente, o tensor remodelado tem o mesmo número total de coeficientes que o tensor inicial. Reshaping é melhor compreendida por meio de exemplos simples:

```python
x = np.array([[0.3, 1.3],[2.0, 3.0], [4.1, 7.2]])
print(x)
# [[0.3 1.3]
#  [2.  3. ]
#  [4.1 7.2]]
print(x.shape) # (3, 2)
print(x.reshape((6,1)))
# [[0.3]
#  [1.3]
#  [2. ]
#  [3. ]
#  [4.1]
#  [7.2]]
print(x.reshape((2,3)))
# [[0.3 1.3 2. ]
#  [3.  4.1 7.2]]
print(x.flatten()) # [0.3 1.3 2.  3.  4.1 7.2]
```

Um caso especial de reshaping comumente encontrado é a **transposição**. Transpor uma matriz significa trocar suas linhas e colunas, de modo que `x[i,:]` se torne `x[:, i]`:

```python
# Criando uma Matriz de apenas zeros de shape (300,20)
z = np.zeros((300,20))
# O método tranpose inverte as linhas e colunas
z = np.transpose(z)
print(z.shape) # (20, 300)
```

### Interpretação Geométrica das Operações de Tensors

Como o conteúdo dos tensors manipulados por operações de tensor pode ser interpretado como coordenadas de pontos em algum espaço geométrico, todas as operações de tensor têm uma interpretação geométrica. Por exemplo, vamos considerar a adição. Começaremos com o seguinte vetor:

```
A = (0.5, 1)
```

Ele é um ponto em um espaço 2D, é comum imaginar um vetor como uma seta ligando a origem ao ponto, como mostrado na figura a seguir:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/GeometricInterpretationTensors.png)

Vamos considerar um novo ponto, `B = (1, 0,25)`, que adicionaremos ao anterior. Isso é feito geometricamente encadeando as setas do vetor, com a localização resultante sendo o vetor que representa a soma dos dois vetores anteriores, como ilustrado na figura a seguir:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/VecSum.png)

Em geral, as operações geométricas elementares, como *affine transformations*, rotações, dimensionamento e assim por diante, podem ser expressas como operações de tensor. Por exemplo, uma rotação de um vetor 2D por um ângulo **teta** pode ser alcançada por meio de um produto escalar com uma matriz **2×2** `R = [u, v]`, onde **u** e **v** são ambos vetores do plano: `u = [cos(theta), sin(theta)]` e `v = [-sin(theta), cos(theta)]`.

### Interpretação Geométrica de Deep Learning

Aprendemos que as redes neurais consistem inteiramente em cadeias de operações de tensor e que todas essas operações de tensor são apenas transformações geométricas dos dados de entrada. Conclui-se que você pode interpretar uma rede neural como uma transformação geométrica muito complexa em um espaço de alta dimensão, implementada por meio de uma longa série de etapas simples.

Em 3D, a seguinte imagem mental pode ser útil. Imagine duas folhas de papel colorido: uma vermelha e outra azul. Coloque um em cima do outro. Agora amasse-os juntos em uma pequena bola. Essa bola de papel amassada são seus dados de entrada, e cada folha de papel é uma classe de dados em um problema de classificação. O que uma rede neural (ou qualquer outro modelo de machine learning) deve fazer é descobrir uma transformação da bola de papel que a desamassaria, de modo a tornar as duas classes novamente separáveis de forma limpa. Com o deep learning, isso seria implementado como uma série de transformações simples do espaço 3D, como aquelas que você poderia aplicar na bola de papel com os dedos, um movimento de cada vez.

A figura a seguir mostra o processo como o ato de desempacotar uma variedade complicada de dados:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/GeometricDeepLearning.png)

Neste ponto, você deve ter uma boa intuição de por que o deep learning se destaca nisso: ele adota a abordagem de decompor incrementalmente uma transformação geométrica complicada em uma longa cadeia de elementos elementares. Cada camada em uma deep network aplica uma transformação que desemaranha um pouco os dados - e uma pilha profunda de camadas (*deep stack of layers*) torna tratável/possível um processo de desemaranhamento extremamente complicado.