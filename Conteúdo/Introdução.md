# Introdução e Contexto Histórico

## Inteligência Artificial, Machine Learning e Deep Learning

Primeiramente, precisamos definir claramente do que estamos falando quando mencionamos **Inteligência Artificial**. O que é inteligência artificial, machine learning e deep learning? Como eles se relacionam?

A figura a seguir ilustra esse relacionamento:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/AIMLDL.png)

### Inteligência Artificial

A inteligência artificial (IA) nasceu na década de 1950, quando um grupo de pioneiros do campo nascente da ciência da computação começou a se perguntar se os computadores poderiam ser feitos para "pensar" - uma questão cujas ramificações ainda estamos explorando hoje. Uma definição concisa do campo seria a seguinte: *o esforço para automatizar tarefas intelectuais normalmente realizadas por humanos*. Como tal, a IA é um campo geral que abrange o machine learning e o deep learning, mas que também inclui muitas outras abordagens que não envolvem nenhum aprendizado. Os primeiros programas de xadrez, por exemplo, envolviam apenas regras codificadas elaboradas por programadores e não se qualificavam como machine learning. Por um longo tempo, muitos especialistas acreditaram que a inteligência artificial de nível humano poderia ser alcançada fazendo com que os programadores elaborassem um conjunto suficientemente grande de regras explícitas para manipular o conhecimento. Esta abordagem é conhecida como IA simbólica e foi o paradigma dominante em IA dos anos 1950 até o final dos anos 1980. Ele atingiu seu pico de popularidade durante o boom dos *expert systems* na década de 1980.

Embora a IA simbólica tenha se mostrado adequada para resolver problemas lógicos bem definidos, como jogar xadrez, tornou-se intratável descobrir regras explícitas para resolver problemas mais complexos e imprecisos, como classificação de imagens, reconhecimento de fala e tradução de idiomas. Uma nova abordagem surgiu para ocupar o lugar da IA simbólica: **machine learning**.

### Machine Learning

Na Inglaterra vitoriana, Lady **Ada Lovelace** era amiga e colaboradora de **Charles Babbage**, o inventor da [Máquina Analítica](https://en.wikipedia.org/wiki/Analytical_Engine): o primeiro computador mecânico conhecido de uso geral. Embora visionário e muito à frente de seu tempo, a Máquina Analítica não foi concebida como um computador de uso geral quando foi projetada nas décadas de 1830 e 1840, porque o conceito de computação de uso geral ainda estava para ser inventado. Era apenas uma forma de usar operações mecânicas para automatizar certos cálculos do campo da análise matemática - daí o seu nome Máquina Analítica. Em 1843, Ada Lovelace comentou sobre a invenção: “*The Analytical Engine has no pretensions whatever to originate anything. It can do whatever we know how to order it to perform... It's province is to assist us in making available what we're already acquainted with*.”

Esta observação foi posteriormente citada pelo pioneiro da IA, **Alan Turing**, como "a objeção de Lady Lovelace" em seu importante artigo de 1950, "*Computing Machinery and Intelligence*", que introduziu o teste de Turing, bem como conceitos-chave que viriam a moldar a IA. Turing estava citando Ada Lovelace enquanto ponderava se os computadores de uso geral seriam capazes de aprendizado e originalidade, e ele chegou à conclusão de que sim.

O machine learning surge a partir desta questão: um computador poderia ir além “do que sabemos ordenar que ele execute” e aprender por si mesmo como executar uma tarefa específica? Um computador poderia nos surpreender? Em vez de os programadores elaborarem as regras de processamento de dados manualmente, um computador poderia aprender essas regras automaticamente olhando os dados?

Esta questão abre a porta para um novo paradigma de programação. Na programação clássica, e no paradigma da IA simbólica, humanos entram com regras (um programa) e dados a serem processados de acordo com essas regras, e surgem as respostas. Com o machine learning, os humanos inserem dados, bem como as respostas esperadas dos dados, e saem as regras. Essas regras podem então ser aplicadas a novos dados para produzir respostas originais.

A figura a seguir ilustra essa ideia:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/ProgrammingML.png)

Um sistema de machine learning é **treinado** em vez de programado explicitamente. Ele é apresentado com muitos exemplos relevantes para uma tarefa e encontra uma estrutura estatística nesses exemplos que eventualmente permite que o sistema crie regras para automatizar a tarefa. Por exemplo, se você desejasse automatizar a tarefa de classificar fotos de cães e gatos, poderia apresentar um sistema de machine learning com muitos exemplos de fotos já rotuladas por humanos, e o sistema aprenderia regras estatísticas para associar fotos específicas a marcações específicas, então se você apresentar uma nova foto ao sistema, ela irá classificá-la adequadamente como um cão ou gato.

Embora o machine learning só tenha começado a florescer na década de 1990, ele rapidamente se tornou o subcampo mais popular e mais bem-sucedido da IA, uma tendência impulsionada pela disponibilidade de hardware mais rápido e conjuntos de dados maiores. O machine learning está intimamente relacionado à estatística matemática, mas difere da estatística em vários aspectos importantes. Ao contrário da estatística, o machine learning tende a lidar com conjuntos de dados grandes e complexos (como um conjunto de dados de milhões de imagens, cada um consistindo de dezenas de milhares de pixels) para os quais a análise estatística clássica, como a análise bayesiana, seria impraticável. Como resultado, o machine learning, especialmente o **deep learning**, exibe comparativamente pouca teoria matemática - talvez muito pouco - e é orientado para a engenharia. É uma disciplina prática em que as ideias são comprovadas empiricamente com mais frequência do que teoricamente.

### Aprendendo Representações através dos Dados

Para definir o deep learning e entender a diferença entre o deep learning e outras abordagens de machine learning, primeiro precisamos ter alguma ideia do que os algoritmos de machine learning fazem. Afirmamos que o machine learning descobre regras para executar uma tarefa de processamento de dados, fornecidos exemplos do que é esperado como resposta. Portanto, para fazer o machine learning, precisamos de três coisas:

- **Pontos de dados de entrada**: Por exemplo, se a tarefa for reconhecimento de fala, esses pontos de dados podem ser arquivos de som de pessoas falando. Se a tarefa for marcação de imagens, elas podem ser imagens.
- **Exemplos da saída esperada**: Em uma tarefa de reconhecimento de fala, podem ser transcrições de arquivos de som geradas por humanos. Em uma tarefa de imagem, as saídas esperadas podem ser tags como “cachorro”, “gato” e assim por diante.
- **Uma maneira de medir se o algoritmo está fazendo um bom trabalho**: Isso é necessário para determinar a distância entre a saída atual do algoritmo e sua saída esperada. A medição é usada como um sinal de feedback para ajustar a maneira como o algoritmo funciona. Essa etapa de ajuste é o que chamamos de aprendizado (**learning**).

Um modelo de machine learning transforma seus dados de entrada em saídas significativas, um processo que é “aprendido” a partir da exposição a exemplos conhecidos de entradas e saídas. Portanto, o problema central no machine learning e no deep learning é transformar dados de forma significativa: em outras palavras, aprender representações úteis dos dados de entrada disponíveis - representações que nos aproximam da saída esperada.

O que é uma representação? Basicamente, é uma maneira diferente de olhar para os dados - para representar ou codificar dados. Por exemplo, uma imagem colorida pode ser codificada no formato **RGB** (red-green-blue) ou no formato **HSV** (hue-saturation-value): são duas representações diferentes dos mesmos dados. Algumas tarefas que podem ser difíceis com uma representação podem se tornar fáceis com outra. Por exemplo, a tarefa “selecionar todos os pixels vermelhos na imagem” é mais simples no formato RGB, enquanto “tornar a imagem menos saturada” é mais simples no formato HSV.

Os modelos de machine learning tratam de encontrar representações apropriadas para seus dados de entrada - transformações dos dados que os tornam mais acessíveis para a tarefa em mãos, como uma tarefa de classificação.

Vamos tornar isso concreto com um exemplo. Considere um eixo **x**, um eixo **y** e
alguns pontos representados por suas coordenadas no sistema (**x**, **y**), como mostrado na figura a seguir:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/DataSample.png)

Como podemos ver, temos alguns pontos verdes e alguns vermelhos. Digamos que queremos desenvolver um algoritmo que pode capturar as coordenadas (**x**, **y**) de um ponto e mostrar se esse ponto é provavelmente verde ou vermelho. Nesse caso:

- As **entradas** são as coordenadas dos nossos pontos.
- As **saídas** esperadas são as cores dos nossos pontos.
- Uma forma de medir se nosso algoritmo está fazendo um bom trabalho pode ser, por exemplo, a porcentagem de pontos que estão sendo classificados corretamente.

O que precisamos neste caso é uma nova representação de nossos dados que separa claramente os pontos verdes dos pontos vermelhos. Uma transformação que poderíamos usar, entre muitas outras possibilidades, seria uma mudança de coordenadas, ilustrada na figura a seguir:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/LearningRepresentations.png)

Neste novo sistema de coordenadas, as coordenadas de nossos pontos podem ser consideradas uma nova representação de nossos dados. Com esta representação, o problema de classificação verde/vermelho pode ser expresso como uma regra simples: “Os pontos vermelhos são tais que **x > 0**” ou “Os pontos verdes são tais que **x < 0**”. Essa nova representação basicamente resolve o problema de classificação.

Neste caso, definimos a mudança de coordenadas manualmente. Mas se, em vez disso, tentássemos pesquisar sistematicamente diferentes mudanças de coordenadas possíveis e usássemos como feedback a porcentagem de pontos que estão sendo classificados corretamente, estaríamos fazendo machine learning. O aprendizado, no contexto do machine learning, descreve um processo de busca automática por melhores representações.

Todos os algoritmos de machine learning consistem em encontrar automaticamente essas transformações que transformam os dados em representações mais úteis para uma determinada tarefa. Essas operações podem ser mudanças de coordenadas, como acabamos de ver, ou projeções lineares (que podem destruir informações), translações, operações não lineares (como “selecionar todos os pontos de modo que **x > 0**”) e assim por diante. Os algoritmos de machine learning geralmente não são criativos para encontrar essas transformações; eles estão apenas pesquisando um conjunto predefinido de operações, chamado de espaço de hipóteses (**hypothesis space**).

Então isso é o que o machine learning é, tecnicamente: pesquisar representações úteis de alguns dados de entrada, dentro de um espaço predefinido de possibilidades, usando a orientação de um sinal de feedback. Essa ideia simples permite resolver uma ampla gama de tarefas intelectuais, desde o reconhecimento de voz até a direção autônoma de um carro.

Agora que compreendemos o que queremos dizer com aprendizagem, vamos dar uma olhada no que torna o **deep learning** especial.

### O "Deep" em Deep Learning

O deep learning é um subcampo específico do machine learning: uma nova abordagem sobre o aprendizado de representações a partir de dados que enfatiza o aprendizado de camadas (*layers*) sucessivas de representações cada vez mais significativas. O *deep* em *deep learning* não é uma referência a qualquer tipo de compreensão mais profunda alcançada pela abordagem; em vez disso, representa essa ideia de *layers* sucessivas de representações. A quantidade de *layers* que contribuem para um modelo de dados é chamada de profundidade do modelo.

O deep learning moderno muitas vezes envolve dezenas ou até centenas de *layers* sucessivas de representações - e todas são aprendidas automaticamente com a exposição aos dados de treinamento. Enquanto isso, outras abordagens de machine learning tendem a se concentrar no aprendizado de apenas uma ou duas *layers* de representações dos dados; portanto, às vezes são chamados de *shallow learning*.

No deep learning, essas representações em *layers* são (quase sempre) aprendidas por meio de modelos chamados **redes neurais**, estruturados em *layers* literais empilhadas umas sobre as outras. O termo rede neural é uma referência à neurobiologia, mas embora alguns dos conceitos centrais do deep learning tenham sido desenvolvidos em parte por inspiração em nossa compreensão do cérebro, os modelos de deep learning não são modelos do cérebro, para nossos propósitos, o deep learning é uma estrutura matemática para aprender representações a partir de dados.

Como são as representações aprendidas por um algoritmo de deep learning? Vamos examinar como uma rede com várias *layers* de profundidade transforma a imagem de um dígito para reconhecer que dígito ele é:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/DigitClassification.png)

Como podemos ver na figura a seguir, a rede neural transforma a imagem do dígito em representações cada vez mais diferentes da imagem original e cada vez mais informativas sobre o resultado final. Podemos pensar em uma *deep network* como uma operação de destilação de informações em vários estágios, em que as informações passam por filtros sucessivos e saem cada vez mais purificadas (isto é, úteis em relação a alguma tarefa).

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/RepresentationsLearned.png)

Então, é isso que o deep learning é, tecnicamente: uma maneira de aprender representações de dados em múltiplos estágios. É uma ideia simples, mas, o que ocorre, é que mecanismos muito simples, em escala suficiente, podem ser muito poderesos.

### Compreendendo Deep Learning

Neste ponto, sabemos que o machine learning é sobre o mapeamento de entradas (como imagens) para destinos (como o rótulo “gato”), o que é feito observando muitos exemplos de entrada e destinos.

Também sabemos que as *deep neural networks* fazem esse mapeamento de entrada para o destino por meio de uma sequência profunda de transformações de dados simples (camadas/layers) e que essas transformações de dados são aprendidas pela exposição a exemplos. Agora vamos ver como esse aprendizado acontece, concretamente.

A especificação do que uma *layer* faz com seus dados de entrada é armazenada nos *weights* da *layer*, que em essência são um monte de números. Em termos técnicos, diríamos que a transformação implementada por uma *layer* é parametrizada por seus *weights*. (Às vezes, os *weights* também são chamados de parâmetros de uma *layer*). Nesse contexto, aprender significa encontrar um conjunto de valores para os *weights* de todas as camadas de uma rede, de modo que a rede mapeie corretamente as entradas de exemplo para seus destinos associados. Mas há um detalhe: uma rede neural profunda pode conter dezenas de milhões de parâmetros. Encontrar o valor correto para todos eles pode parecer uma tarefa difícil, especialmente considerando que modificar o valor de um parâmetro afetará o comportamento de todos os outros!

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/Weights.png)

Para controlar algo, primeiro você precisa ser capaz de observá-lo. Para controlar a saída de uma rede neural, você precisa ser capaz de medir a que distância essa saída está do que você esperava. Esse é o trabalho da **função Loss** da rede, também chamada de **função objetivo**. A função Loss pega as previsões da rede e o verdadeiro destino/target (o que você queria que a rede produzisse) e calcula uma pontuação de distância, capturando o quão bem a rede se saiu neste exemplo específico.

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/LossFunction.png)

O truque fundamental no deep learning é usar essa pontuação como um sinal de feedback para ajustar um pouco o valor dos *weights*, em uma direção que diminuirá a pontuação Loss. Esse ajuste é trabalho do **otimizador**, que implementa o que é chamado de **algoritmo Backpropagation**: o algoritmo central no deep learning.

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/UnderstandingDeepLearning.png)

Inicialmente, os *weights* da rede são atribuídos a valores aleatórios, então a rede apenas implementa uma série de transformações aleatórias. Naturalmente, sua produção está longe do que deveria ser, e a pontuação Loss é, portanto, muito alta. Mas a cada exemplo que a rede processa, os *weights* são ajustados um pouco na direção correta, e a pontuação Loss diminui.

### Conquistas do Deep Learning

Embora o deep learning seja um subcampo bastante antigo do machine learning, ele só ganhou destaque no início da década de 2010. Nos poucos anos desde então, ela alcançou nada menos que uma revolução no campo, com resultados notáveis em problemas de percepção como visão e audição - problemas envolvendo habilidades que parecem naturais e intuitivas para os humanos, mas há muito tempo são difíceis para as máquinas.

Em particular, o deep learning alcançou os seguintes avanços, todos em áreas historicamente difíceis de machine learning:

- Classificação de imagem quase humana
- Reconhecimento de fala quase humano
- Transcrição de caligrafia quase humana
- Tradução automática aprimorada
- Conversão de texto em fala aprimorada
- Assistentes digitais como Google Now e Amazon Alexa
- Direção autônoma quase de nível humano
- Melhoria na segmentação de anúncios, conforme usado pelo Google, Baidu e Bing
- Resultados de pesquisa aprimorados na web
- Capacidade de responder a perguntas em linguagem natural
- Jogar games em nível superhumano

Ainda estamos explorando toda a extensão do que o deep learning pode fazer. Começamos a aplicá-lo a uma ampla variedade de problemas fora da percepção da máquina e do entendimento da linguagem natural, como o raciocínio formal. Se for bem-sucedido, isso pode anunciar uma era em que o deep learning ajudará os humanos na ciência, no desenvolvimento de software e muito mais.

A ilustração a seguir apresenta algumas conquistas importantes da inteligência artificial como um todo:

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/AITimeLine.png)

### Breve Histórico de Machine Learning

O deep learning atingiu um nível de atenção pública e investimento da indústria nunca antes visto na história da IA, mas não é a primeira forma de machine learning bem-sucedida. É seguro dizer que a maioria dos algoritmos de machine learning usados na indústria hoje não são algoritmos de deep learning. O deep learning nem sempre é a ferramenta certa para o trabalho - às vezes não há dados suficientes para que o deep learning seja aplicável e, às vezes, o problema é melhor resolvido por um algoritmo diferente. 

#### Modelagem Probabilística

A modelagem probabilística é a aplicação dos princípios da estatística à análise de dados. Foi uma das primeiras formas de machine learning e ainda é amplamente usada até hoje. Um dos algoritmos mais conhecidos nesta categoria é o **algoritmo Naive Bayes**.

[Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) é um tipo de classificador de machine learning baseado na aplicação do [teorema de Bayes](https://en.wikipedia.org/wiki/Bayes%27_theorem), enquanto assume que os *features* nos dados de entrada são todos independentes (uma suposição forte ou "ingênua", de onde vem o nome). Esta forma de análise de dados é anterior aos computadores e foi aplicada manualmente décadas antes de sua primeira implementação de computador (provavelmente datando da década de 1950). O teorema de Bayes e os fundamentos da estatística datam do século XVIII.

Um modelo intimamente relacionado é a [regressão logística](https://en.wikipedia.org/wiki/Logistic_regression), que às vezes é considerada o “Hello World” do machine learning moderno. Não se deixe enganar por seu nome - regressão logística é um algoritmo de classificação em vez de um algoritmo de regressão. Muito parecido com Naive Bayes, regressão logística antecede a computação em muito tempo, mas ainda é útil até hoje, graças à sua natureza simples e versátil.

#### Redes Neurais Iniciais

As primeiras iterações de redes neurais foram completamente suplantadas pelas variantes modernas abordadas até então, mas é útil estar ciente de como o deep learning se originou. Embora as ideias centrais das redes neurais tenham sido investigadas em formas de "brinquedo" já na década de 1950, a abordagem levou décadas para começar.

Por muito tempo, a peça que faltava era uma maneira eficiente de treinar grandes redes neurais. Isso mudou em meados da década de 1980, quando várias pessoas redescobriram independentemente o **[algoritmo backpropagation](https://en.wikipedia.org/wiki/Backpropagation)** - uma maneira de treinar cadeias de operações paramétricas usando a otimização **[gradient-descent](https://en.wikipedia.org/wiki/Gradient_descent)**.

A primeira aplicação prática bem-sucedida de redes neurais veio em 1989 da Bell Labs, quando Yann LeCun combinou as ideias anteriores de **convolutional neural networks** e **backpropagation** e as aplicou ao problema de classificação de dígitos manuscritos. A rede resultante, batizada de [LeNet](https://en.wikipedia.org/wiki/LeNet), foi usada pelo Serviço Postal dos Estados Unidos na década de 1990 para automatizar a leitura de CEP's em envelopes de correio.

#### Kernel Methods

À medida que as redes neurais começaram a ganhar algum respeito entre os pesquisadores na década de 1990, graças a esse primeiro sucesso, uma nova abordagem de machine learning tornou-se famosa e rapidamente mandou as redes neurais de volta ao esquecimento: os **kernel methods**. Os kernel methods são um grupo de algoritmos de classificação, o mais conhecido dos quais é o [support vector machine](https://en.wikipedia.org/wiki/Support-vector_machine) (**SVM**). A formulação moderna de um SVM foi desenvolvida por Vladimir Vapnik e Corinna Cortes no início dos anos 1990 na Bell Labs e publicada em 1995, embora uma formulação linear mais antiga tenha sido publicada por Vapnik e Alexey Chervonenkis já em 1963.

Os SVMs visam resolver problemas de classificação encontrando *decision boundaries* entre dois conjuntos de pontos pertencentes a duas categorias diferentes. Um *decision boundary* pode ser considerado como uma linha ou superfície que separa seus dados de treinamento em dois espaços correspondentes a duas categorias. Para classificar novos pontos de dados, você só precisa verificar de que lado do *decision boundary* eles caem.

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/DecisionBoundary.png)

Os SVMs procuram encontrar esses limites em duas etapas:

1. Os dados são mapeados para uma nova representação de alta dimensão onde a *decision boundary* pode ser expressa como um hiperplano (se os dados fossem bidimensionais, como na figura anterior, um hiperplano seria uma linha reta).
2. Uma boa *decision boundary* (um hiperplano de separação) é calculado tentando maximizar a distância entre o hiperplano e os pontos de dados mais próximos de cada classe, uma etapa chamada de maximização da margem. Isso permite que o limite seja bem generalizado para novas amostras fora do conjunto de dados de treinamento.

A técnica de mapear dados para uma representação de alta dimensão onde um problema de classificação se torna mais simples pode parecer boa no papel, mas na prática é muitas vezes intratável computacionalmente. É aí que entra o **kernel trick** (a ideia-chave que dá nome aos **kernel methods**).

Para encontrar bons hiperplanos de *decision boundary* no novo espaço de representação, não precisamos computar explicitamente as coordenadas dos pontos no novo espaço; só precisamos calcular a distância entre pares de pontos naquele espaço, o que pode ser feito de forma eficiente usando uma **função de kernel**. Uma função kernel é uma operação computacionalmente tratável que mapeia quaisquer dois pontos em seu espaço inicial para a distância entre esses pontos em seu espaço de representação de destino, ignorando completamente o cálculo explícito da nova representação. As funções do kernel são normalmente criadas à mão, em vez de aprendidas a partir dos dados - no caso de um SVM, apenas o hiperplano de separação é aprendido.

Na época em que foram desenvolvidos, os SVMs exibiam desempenho de ponta em problemas de classificação simples e eram um dos poucos métodos de machine learning apoiados por extensa teoria e passíveis de análises matemáticas sérias, tornando-os bem compreendidos e facilmente interpretáveis. Por causa dessas propriedades úteis, os SVMs se tornaram extremamente populares no campo por um longo tempo.

Mas os SVMs provaram ser difíceis de escalar para grandes conjuntos de dados e não forneceram bons resultados para problemas de percepção, como classificação de imagem. Como um SVM é um *shallow method*, a aplicação de um SVM a problemas de percepção requer primeiro a extração manual de representações úteis (uma etapa chamada **feature engineering**), o que é difícil e frágil.

#### Árvores de Decisão, Random Forests e Gradient Boosting Machines

As [árvores de decisão](https://en.wikipedia.org/wiki/Decision_tree_learning) são estruturas semelhantes a fluxogramas que permitem classificar pontos de dados de entrada ou prever valores de saída fornecidas entradas. Eles são fáceis de visualizar e interpretar. As árvores de decisão aprendidas com os dados começaram a receber um interesse significativo de pesquisa na década de 2000 e, em 2010, eram frequentemente preferidas aos métodos de kernel.

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/DecisionTree.png)

Uma árvore de decisão: os parâmetros aprendidos são as perguntas sobre os dados. Uma pergunta poderia ser, por exemplo: "O coeficiente 2 nos dados é maior que 3.5?"

Em particular, o algoritmo [Random Forest](https://en.wikipedia.org/wiki/Random_forest) introduziu uma abordagem prática e robusta sobre o aprendizado de árvore de decisão que envolve a construção de um grande número de árvores de decisão especializadas e, em seguida, agrupar seus resultados. Random Forests são aplicáveis a uma ampla gama de problemas. Quando o popular site de competição de machine learning [Kaggle](http://kaggle.com) começou em 2010, as random forests rapidamente se tornaram as favoritas na plataforma - até 2014, quando as *gradient boosting machines* assumiram o controle.

Uma [gradient boosting machine](https://en.wikipedia.org/wiki/Gradient_boosting), bem como uma random forest, é uma técnica de machine learning baseada na combinação de modelos de previsão fracos, geralmente árvores de decisão. Ele usa o *gradient boosting*, uma maneira de melhorar qualquer modelo de machine learning, treinando iterativamente novos modelos especializados em abordar os pontos fracos dos modelos anteriores. Aplicado a árvores de decisão, o uso da técnica de *gradient boosting* resulta em modelos que superam estritamente as random forests na maioria das vezes, embora tenham propriedades semelhantes.

#### Redes Neurais

Por volta de 2010, embora as redes neurais fossem quase totalmente evitadas pela comunidade científica em geral, várias pessoas que ainda trabalhavam em redes neurais começaram a fazer avanços importantes: os grupos de **Geoffrey Hinton** na Universidade de Toronto, **Yoshua Bengio** na Universidade de Montreal, **Yann LeCun** na Universidade de Nova York e IDSIA na Suíça.

Em 2011, Dan Ciresan, da IDSIA, começou a ganhar competições acadêmicas de classificação de imagens com *deep neural networks* treinadas em GPU - o primeiro sucesso prático do deep learning moderno. Mas o momento decisivo veio em 2012, com a entrada do grupo de Hinton no desafio anual de classificação de imagens em grande escala [ImageNet](https://image-net.org/challenges/LSVRC/index.php). O desafio ImageNet era notoriamente difícil na época, consistindo em classificar imagens coloridas de alta resolução em 1.000 categorias diferentes após o treinamento em 1.4 milhão de imagens. Em 2011, a precisão dos cinco primeiros modelos vencedores, com base em abordagens clássicas de visão computacional, foi de apenas 74.3%. Então, em 2012, uma equipe liderada por Alex Krizhevsky e assessorada por Geoffrey Hinton foi capaz de alcançar uma precisão de 83.6% - um avanço significativo. A competição tem sido dominada por *deep convolutional neural networks* todos os anos desde então. Em 2015, o vencedor atingiu uma precisão de 96.4%, e a tarefa de classificação na ImageNet foi considerada um problema totalmente resolvido.

Desde 2012, as *deep convolutional neural networks* (**convnets**) tornaram-se o algoritmo ideal para todas as tarefas de visão computacional; mais geralmente, eles trabalham em todas as tarefas perceptivas. Nas principais conferências de visão computacional em 2015 e 2016, era quase impossível encontrar apresentações que não envolvessem convnets de alguma forma. Ao mesmo tempo, o deep learning também encontrou aplicações em muitos outros tipos de problemas, como o **[natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing)**. Ele substituiu completamente os SVMs e as árvores de decisão em uma ampla gama de aplicações. Por exemplo, por vários anos, a European Organization for Nuclear Research, CERN, usou métodos baseados em árvore de decisão para análise de dados de partículas do detector ATLAS no Large Hadron Collider (LHC); mas o CERN eventualmente mudou para deep neural networks baseadas em [Keras](https://keras.io/) devido ao seu melhor desempenho e facilidade de treinamento em grandes conjuntos de dados.

#### Diferencial de Deep Learning

O principal motivo pelo qual o deep learning decolou tão rapidamente é que ele ofereceu melhor desempenho em muitos problemas. Mas essa não é a única razão. O deep learning também torna a solução de problemas muito mais fácil, porque automatiza completamente o que costumava ser a etapa mais crucial em um fluxo de trabalho de machine learning: **feature engineering**.

As técnicas anteriores de machine learning - shallow learning - envolviam apenas a transformação dos dados de entrada em um ou dois espaços de representação sucessivos, geralmente por meio de transformações simples, como projeções não lineares de alta dimensão (SVMs) ou árvores de decisão. Mas as representações refinadas exigidas por problemas complexos geralmente não podem ser obtidas por tais técnicas. Como tal, os humanos tiveram que ir longe para tornar os dados de entrada iniciais mais acessíveis ao processamento por esses métodos: eles tiveram que criar manualmente boas camadas de representações para seus dados. Isso é chamado de **feature engineering**. O deep learning, por outro lado, automatiza completamente esta etapa: com o deep learning, você aprende todos os features em uma única passagem, em vez de ter que projetá-los sozinho. Isso simplificou muito os fluxos de trabalho de machine learning, muitas vezes substituindo pipelines de vários estágios sofisticados por um modelo de deep learning único e simples.

### Por que Deep Learning?

As duas idéias-chave de deep learning para visão computacional - **convolutional neural networks** e **backpropagation** - já eram bem compreendidos em 1989. O algoritmo [Long Short-Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory) (LSTM), que é fundamental para o deep learning para séries temporais, foi desenvolvido em 1997 e quase não mudou desde então. Então, por que o deep learning só decolou depois de 2012? O que mudou nessas duas décadas?

Em geral, três forças técnicas estão impulsionando os avanços no machine learning:

- Hardware
- Conjuntos de dados e benchmarks
- Avanços algorítmicos

Como o campo é guiado por descobertas experimentais e não pela teoria, os avanços algorítmicos só se tornam possíveis quando dados e hardware apropriados estão disponíveis para tentar novas ideias (ou ampliar ideias antigas, como costuma ser o caso). O machine learning não é matemática ou física, onde grandes avanços podem ser feitos com uma caneta e um pedaço de papel. É uma ciência da engenharia.

Os verdadeiros gargalos nas décadas de 1990 e 2000 eram os dados e o hardware. Mas eis o que aconteceu durante esse tempo: a internet decolou e chips gráficos de alto desempenho foram desenvolvidos para as necessidades do mercado de jogos.

#### Hardware

Entre 1990 e 2010, as CPUs disponíveis no mercado tornaram-se mais rápidas por um fator de aproximadamente 5.000, como resultado, hoje em dia é possível executar pequenos modelos de deep learning em seu laptop, enquanto isso seria intratável 25 anos atrás.

Mas os modelos típicos de deep learning usados em visão computacional ou reconhecimento de fala exigem ordens de magnitude mais poder computacional do que o que seu laptop pode oferecer. Ao longo dos anos 2000, empresas como **NVIDIA** e **AMD** têm investido bilhões de dólares no desenvolvimento de chips rápidos e massivamente paralelos (graphical processing units [GPUs]) para alimentar os gráficos de videogames cada vez mais fotorrealistas - supercomputadores baratos e de uso único, projetados para renderizar cenas 3D complexas em sua tela em tempo real. Esse investimento veio beneficiar a comunidade científica quando, em 2007, a NVIDIA lançou o [CUDA](https://developer.nvidia.com/about-cuda), uma interface de programação para sua linha de GPUs. Um pequeno número de GPUs começou a substituir enormes clusters de CPU em vários aplicativos altamente paralelizáveis, começando com a modelagem física. Deep neural networks, consistindo principalmente de muitas pequenas multiplicações de matrizes, também são altamente paralelizáveis; e por volta de 2011, alguns pesquisadores começaram a escrever implementações CUDA de redes neurais - Dan Ciresan e Alex Krizhevsky estavam entre os primeiros.

Além do mais, a indústria de deep learning está começando a ir além das GPUs e está investindo em chips cada vez mais especializados e eficientes para deep learning. Em 2016, em sua convenção anual I/O, o Google revelou seu projeto de **tensor processing unit** (TPU): um novo design de chip desenvolvido desde o início para executar deep neural networks, que é supostamente 10 vezes mais rápido e muito mais eficiente em termos de energia do que os GPUs de primeira linha.

#### Dados

A IA às vezes é anunciada como a nova revolução industrial. Se o deep learning é a máquina a vapor dessa revolução, os dados são o seu carvão: a matéria-prima que move nossas máquinas inteligentes, sem a qual nada seria possível. Quando se trata de dados, além do progresso exponencial em hardware de armazenamento nos últimos 20 anos (seguindo a [lei de Moore](https://en.wikipedia.org/wiki/Moore%27s_law)), a virada do jogo foi a ascensão da Internet, tornando viável a coleta e distribuição de conjuntos de dados muito grandes para machine learning. Hoje, grandes empresas trabalham com conjuntos de dados de imagens, conjuntos de dados de vídeo e conjuntos de dados de linguagem natural que não poderiam ser coletados sem a Internet. Tags de imagens geradas pelos usuários no Flickr, por exemplo, têm sido um tesouro de dados para visão computacional. Os vídeos do YouTube também. E a Wikipedia é um conjunto de dados chave para o processamento de linguagem natural.

Se há um conjunto de dados que tem sido um catalisador para o surgimento do deep learning, este é o conjunto de dados ImageNet, que consiste em 1.4 milhão de imagens que foram anotadas manualmente com 1.000 categorias de imagens (1 categoria por imagem). Mas o que torna o ImageNet especial não é apenas seu grande tamanho, mas também a competição anual associada a ele.

#### Algoritmos

Além de hardware e dados, até o final dos anos 2000, não havia uma maneira confiável de treinar deep neural networks. Como resultado, as redes neurais ainda eram bastante superficiais, usando apenas uma ou duas camadas (*layers*) de representações; assim, eles não eram capazes de brilhar contra *shallow methods* mais refinados, como SVMs e random forests. A questão principal era a propagação do gradiente através de pilhas profundas de camadas (*deep stacks of layers*). O sinal de feedback usado para treinar as redes neurais desapareceria à medida que o número de camadas aumentasse.

Isso mudou em torno de 2009-2010 com o advento de várias melhorias algorítmicas simples, mas importantes, que permitiram uma melhor propagação de gradiente:

- Melhores **funções de ativação** para camadas neurais
- Melhores esquemas de inicialização de *weight*, começando com o pré-treinamento em camadas, que foi rapidamente abandonado
- Melhores esquemas de otimização, como **RMSProp** e **Adam**

Somente quando essas melhorias iniciaram, começaram a permitir o treinamento de modelos com 10 ou mais camadas, o deep learning então começou a brilhar.

Finalmente, em 2014, 2015 e 2016, maneiras ainda mais avançadas de ajudar a propagação de gradiente foram descobertas, como *batch normalization*, conexões residuais e *depth-wise separable convolutions*.

## A Singularidade Tecnológica

A [singularidade tecnológica](https://en.wikipedia.org/wiki/Technological_singularity) é um ponto hipotético no tempo em que o crescimento tecnológico se torna incontrolável e irreversível, resultando em mudanças imprevisíveis na civilização humana. De acordo com a versão mais popular da hipótese da singularidade, chamada de explosão de inteligência, um agente inteligente atualizável acabará entrando em uma "reação descontrolada" de ciclos de autoaperfeiçoamento, cada geração nova e mais inteligente aparecendo cada vez mais rapidamente, causando uma "explosão" em inteligência e resultando em uma superinteligência poderosa que ultrapassa qualitativamente de longe toda inteligência humana.

O primeiro a usar o conceito de "singularidade" no contexto tecnológico foi John von Neumann. Stanislaw Ulam relata uma discussão com von Neumann "centrada no progresso acelerado da tecnologia e nas mudanças no modo de vida humana, o que dá a impressão de se aproximar de alguma singularidade essencial na história da raça além da qual os assuntos humanos, como os conhecemos, não poderia continuar". Os autores subsequentes seguiram esse ponto de vista.

O modelo de "explosão de inteligência" de I. J. Good prevê que uma futura superinteligência desencadeará uma singularidade.

O conceito e o termo "singularidade" foram popularizados por Vernor Vinge em seu ensaio de 1993 **The Coming Technological Singularity**, no qual ele escreveu que isso marcaria o fim da era humana, pois a nova superinteligência continuaria a se atualizar e avançaria tecnologicamente a uma taxa incompreensível. Ele escreveu que ficaria surpreso se ocorresse antes de 2005 ou depois de 2030.

Figuras públicas como Stephen Hawking e Elon Musk expressaram preocupação de que a inteligência artificial total (IA) poderia resultar na extinção humana.

Quatro pesquisas de pesquisadores de IA, conduzidas em 2012 e 2013 por Nick Bostrom e Vincent C. Müller, sugeriram uma estimativa de probabilidade média de 50% de que a inteligência artificial geral (AGI) seria desenvolvida até 2040-2050.

### Emergência de Superinteligência

Uma superinteligência é um agente hipotético que possui uma inteligência muito superior à das mentes humanas mais brilhantes e talentosas. "Superinteligência" também pode se referir à forma ou grau de inteligência possuída por tal agente. John von Neumann, Vernor Vinge e Ray Kurzweil definem o conceito em termos da criação tecnológica da superinteligência, argumentando que é difícil ou impossível para os humanos de hoje prever como seriam as vidas dos seres humanos em um mundo pós-singularidade.

### Melhorias de Velocidade

O crescimento exponencial da tecnologia de computação sugerido pela lei de Moore é comumente citado como uma razão para esperar uma singularidade em um futuro relativamente próximo, e vários autores propuseram generalizações da lei de Moore. O cientista da computação e futurista Hans Moravec propôs em um livro de 1998 que a curva de crescimento exponencial poderia ser estendida por meio de tecnologias de computação anteriores ao circuito integrado.

Ray Kurzweil postula uma **lei de retornos acelerados** em que a velocidade da mudança tecnológica (e mais geralmente, todos os processos evolutivos) aumenta exponencialmente, generalizando a lei de Moore da mesma maneira que a proposta de Moravec, e também incluindo a tecnologia de materiais (especialmente quando aplicada à nanotecnologia), tecnologia médica e outros. Entre 1986 e 2007, a capacidade específica das aplicações de máquina para computar informações per capita praticamente dobrou a cada 14 meses; a capacidade per capita dos computadores de uso geral do mundo dobrou a cada 18 meses; a capacidade global de telecomunicações per capita dobrou a cada 34 meses; e a capacidade mundial de armazenamento per capita dobrou a cada 40 meses.

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/LawAcceleratingReturns.png)

Ray Kurzweil escreve que, devido a mudanças de paradigma, uma tendência de crescimento exponencial estende a lei de Moore de circuitos integrados para transistores anteriores, válvulas de vácuo, relés e computadores eletromecânicos. Ele prevê que o crescimento exponencial continuará e que em algumas décadas o poder de computação de todos os computadores excederá o dos cérebros humanos ("não aprimorados"), com a inteligência artificial super-humana aparecendo na mesma época.

Kurzweil reserva o termo "singularidade" para um rápido aumento da inteligência artificial (em oposição a outras tecnologias), escrevendo por exemplo que "A Singularidade nos permitirá transcender as limitações de nossos corpos biológicos e cérebros. Não haverá distinção, pós-Singularidade, entre homem e máquina".

### Evolução Sociobiológica

Um artigo de 2016 na **Trends in Ecology & Evolution** argumenta que "os humanos já adotam fusões de biologia e tecnologia. Passamos a maior parte do nosso tempo de vigília nos comunicando por canais mediados digitalmente, confiamos na inteligência artificial em nossas vidas por meio da frenagem antibloqueio em carros e pilotos automáticos em aviões. Com um em cada três casamentos na América começando online, os algoritmos digitais também estão desempenhando um papel na união e reprodução do casal humano".

O artigo argumenta ainda que, da perspectiva da evolução, várias Transições Principais nas Evoluções anteriores transformaram a vida por meio de inovações no armazenamento e replicação de informações (RNA, DNA, multicelularidade e cultura e linguagem). No estágio atual da evolução da vida, a biosfera baseada em carbono gerou um sistema cognitivo (humanos) capaz de criar tecnologia que resultará em uma transição evolutiva comparável.

![img](https://raw.githubusercontent.com/the-akira/DeepLearning/master/Imagens/Evolution.png)

**Figura**: Linha do tempo esquemática de *Information and Replicators in the Biosphere*: "major evolutionary transitions" de Gillings et al. No processamento de informações.

A informação digital criada por humanos atingiu uma magnitude semelhante à informação biológica na biosfera. Desde a década de 1980, a quantidade de informações digitais armazenadas dobrou a cada 2.5 anos, atingindo cerca de 5 zetabytes em 2014 (5×10^21 bytes).

Se o crescimento do armazenamento digital continuar em sua taxa atual de 30-38% de crescimento anual composto por ano, ele irá rivalizar com o conteúdo total de informações contidas em todo o DNA em todas as células da Terra em cerca de 110 anos. Isso representaria o dobro da quantidade de informações armazenadas na biosfera em um período total de apenas 150 anos.