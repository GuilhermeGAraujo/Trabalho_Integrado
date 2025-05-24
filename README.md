# Trabalho_Integrado
  Respositório para o trabalho apresentado em 19/04/23 para conclusão da Pós-Graduação em Ciência de Dados e Big Data pela PUC-Minas

# FORECASTING DE SECAS UTILIZANDO REDES NEURAIS CNN-LSTM

## Objetivo:

  Este trabalho tem como objetivo utilizar tecnologias e métodos de aprendizado de máquina e ciência de dados para criar um modelo capaz de prever a intensidade da seca em regiões dos Estados Unidos com 6 semanas de antecedência utilizando dados meteorológicos e geográficos/geológicos coletados nos últimos 6 meses.

## Coleta de Dados:

  Os dados utilizados neste trabalho foram obtidos na plataforma do Kaggle no dia 12/12/2021, onde o usuário Christoph Minixhofer publicou dados coletados e limpados por ele, obtidos do U.S. Drought Monitor e NASA POWER Project. O script de coleta e limpeza dos dados pode ser encontrado no GitHub (https://github.com/MiniXC/droughted_scripts).

  O dataset meteorológico é uma junção dos dados NASA POWER Project e U.S. Drought Monitor e está dividido em três tabelas meteorológicas, treinamento, validação e teste. A tabela de treinamento possui dados do período de 2000 a 2016 e dimensão de 19.300.680x21, a tabela de validação possui dados do período de 2017-2018 e dimensão de 2.268.840x21, a tabela de teste possui dados do período de 2019-2020 e dimensão de 2.271.948x21. 

## Processamento e Tratamento de Dados:

  A base de dados completa possui mais de 3GB de data, para diminuir a dificuldade em manipular e analisar os dados em um computador pessoal primeiramente foi feita uma redução da base, essa redução foi realizada selecionando os estados em regiões de grande importância para a agricultura dos Estados Unidos, uma atividade altamente impactada pelas secas.
  
  O Wheat Belt e Corn Belt são regiões produtoras principalmente de trigo e milho e compreendem os estados do Kansas, Oklahoma, Texas, Nebraska, Colorado, Montana,
Dakota do Norte, Dakota do Sul, Minnesota, Indiana, Illinois, Iowa e do Missouri de acordo com a Enciclopédia Britannica. Estes estados foram selecionados na base de dados através do código fips de acordo com o United States Census Bureau.

  A redução foi aplicada nos datasets “train_timeseries.csv”, “validation_timeseries.csv”, “test_timeseries.csv”, “soil_data.csv” gerando novos datasets “redu_train”, “redu_valid”, “redu_test”, “redu_soil” respectivamente que foram salvos como objetos pickle comprimidos no formato pbz2.
  
  Como resultado, a base de dados foi reduzida em 60%.
  
  
  ![image](https://github.com/GuilhermeGAraujo/Trabalho_Integrado/assets/65792526/e60a3e3f-a528-41bd-be73-90eaf3c90145)

  A variável “score” é do tipo float, apresentando valores não inteiros, isso deve ao fato de o criador do dataset ter usado um método de interpolação para preencher os valores ausentes na data de publicação do relatório. Como a variável score é o target do problema de classificação, ela foi arredondada para o inteiro mais próximo e tipada como categoria.
  
  Devido à forma como o dataset foi gerado - dados de seca semanais e dados climáticos diários - apenas 1/7 das entradas possuem um score sendo o restante null. Estes
dados poderiam ser preenchidos replicando o score ao longo da semana até a próxima publicação, porém isto assumiria que não houve uma mudança gradual e que de certa forma o que ocorreu ao longo da semana não teria efeito no clima, como se a mudança acontecesse repentinamente; também poderia ser feito um preenchimento usando interpolação. Porém, como a proporção nula é muito maior, a precisão da interpolação não seria muito alta, além de gerar números inteiros também geraria números racionais que ao serem interpolados poderiam aumentar ainda mais a imprecisão. Para este trabalho foi decidido não utilizar as datas com score ausente, porém seus dados foram agregados e adicionados como novas colunas nas datas onde o score estava presente.

  O clima tem diversos padrões cíclicos e as secas como um componente do clima também segue esses padrões, por isso a data onde cada evento ocorre é importante na detecção de padrões e da ciclicidade das secas. Porém a data no formato datetime, ou como string não é possível de ser utilizada por modelos matemáticos sendo necessário codificá-la de forma numérica preservando a ciclicidade. A codificação foi realizada através do seno e cosseno.
  
  A codificação do seno e cosseno preserva a ciclicidade da data, mas em contrapartida divide um único atributo em dois, isto pode ser um problema dependendo do modelo a ser utilizado, porém não é para LSTM. O seno e cosseno não são aplicados diretamente à data mas sim a uma representação numérica dela. Como nosso dataset possui dados semanais que se repetem a cada ano, essa representação será o número da semana do ano, ou seja, cada semana será representada por um número de 1 a 52. Estes representação então é normalizada de forma corresponder a ciclo de 0-2π, a data codificada então é finalmente calculada aplicando o seno e cosseno ao “x” normalizado gerando cada componente correspondente.

  Este método foi utilizado para criar as colunas “data_sen” e “data_cos”. A coluna de data foi excluída do dataset de criação do modelo. Como o objetivo deste trabalho é usar uma rede neural LSTM para a previsão de secas é importante que os dados estejam em uma escala compatível a redes neurais, redes neurais raramente se comportam bem com valores elevados de inputs ou com escala desses inputs muito diferente eles, como temos variáveis que são frações e outras que podem chegar a centenas o dado precisa ser padronizado para ser alimentado à rede neural. Foi utilizado o método MinMaxScaler, a normalização gaussiana não foi considerada pelo fato de muitos dos dados não possuírem distribuição normal, como será evidenciado na exploração de dados.
  
  Todos estes métodos foram aplicados aos datasets test, valid e train, ao dataset de solo foi aplicado apenas a redução de entradas por estado e a padronização.
    
## Análise e Exploração dos Dados:

A primeira análise realizada foi quanto a distribuição dos scores em cada dataset, é importante observar se cada dataset tem uma distribuição semelhante, pois caso o dataset de treino tenha uma distribuição muito diferente dos datasets de validação ou treino, nosso modelo não terá um bom desempenho.


![treino_dataset](https://github.com/user-attachments/assets/9856e8e6-20d1-4721-b397-e84d85f2d288 " distribuição dataset de treino")
![teste_dataset](https://github.com/user-attachments/assets/cfc36683-93bc-4184-962c-fe03c172abdd "distribuição dataset de teste")
![valid_dataset](https://github.com/user-attachments/assets/0588ce87-8c18-4fae-8e2f-775bee0036b8 "distribuição dataset de validação")

Os datasets possuem distribuição assimétrica a direita. Também é possível notar que as classes são desbalanceadas, com mais da metade tendo score 0.

Também foi realizada a análise da distribuição de score para cada uma das variáveis principais -não levando em consideração as variáveis de max, min e amplitude. A análise nos mostra que com exceção dos extremos condição normal (0) e seca extrema (5) as distribuições das demais classificações são muito semelhantes.

O boxplot de chuva (PRECTOT) é interessante pois quase todos os dias do dataset não chove ou chove muito pouco, os dias que chovem uma quantidade relevante são outliers. Outro ponto interessante é como a média e quartis da umidade específica do ar (QV2M) para seca extrema aparenta ser muito próxima da classificação de clima normal e maior que todas as outras classificações, porém um teste de hipótese com nível de confiança de 95% mostra o contrário, a única distribuição estatisticamente não semelhante à classificação seca é a classificação normal.

Também é possível observar que temperaturas altas estão associadas com seca extrema, fato corroborado por teste de hipótese com nível de confiança de 95%, essa é uma informação promissora pois indica que estas variáveis podem ser capazes de ajudar modelos a classificarem corretamente o nível de seca. 

![box_plots](https://github.com/user-attachments/assets/ce66c17c-f9f5-4146-a842-7a4081c84c81 "Box plots")


Analisando os pairplots  das demais variáveis para cada estado é possível notar que cada região possui distribuições e correlações diferentes. Por exemplo a distribuição dos valore de pressão (PS) nos estados de Illinois e Iowa são bem próximas de uma distribuição normal, já colorado não tem uma distribuição normal; ou como a temperatura de bulbo úmido (T2MWET) em Illinois lembra mais uma distribuição uniforme, a de colorado uma distribuição assimétrica à direita e Iowa uma distribuição multimodal. Essa diferença das distribuições entre as regiões pode dificultar a criação de um modelo capaz de generalizar todos os estados.

Apesar das correlações entres as features variarem de estado para estado algumas são constantes entre os estados. Velocidade alta de vento estão correlacionadas a pouca chuva, baixas temperatura de bulbo úmido são correlacionadas a pouca chuva (isto é esperado pois quanto mais seco está o ar menor é a temperatura de bulbo úmido).

A alta correlação entre variáveis semelhantes como vento a 10 e 50 metros ou as diferentes temperaturas era esperado. A curva formada entre a umidade e as demais temperaturas seguem os padrões de equações físico-químicas conhecidas.
![colorado](https://github.com/user-attachments/assets/04b060fd-c6d0-4612-a5fe-b0620383c0b0 "Pairplot de Colorado")
![illinois](https://github.com/user-attachments/assets/8f75abce-1f57-4b72-a753-f7799f315f7f "Pairplot de Illinois")
![iowa](https://github.com/user-attachments/assets/17ee820b-a3d0-4ff3-82b6-b3e3fedd2e0b "Pairplot de Iowa")


Também foi realizada a análise do dataset de solo com relação ao score de nível de secas através de stripplots. Essa análise trouxe algumas observações interessantes.

Áreas urbanas raramente passam por secas severas, isso provavelmente se dá pelo fato de cidades serem geralmente formadas onde os recursos hídricos são abundantes o suficiente para sustentar a população. Regiões de cultivo regadas por chuva possuem mais observações de seca intensa que regiões de cultivo irrigado, porém também possuem mais observações com situação normal. 

Outros aspectos interessantes é que regiões com inclinação majoritariamente do tipo 1 não são afetadas por secas severas - essas regiões, no entanto, são limitadas a dois condados no estado de Illinois - e que as regiões afetadas por secas severas estão mais concentradas entre as latitudes -105º e -95º.

![Stripplot da fração de terra urbana para cada classificação de seca](https://github.com/user-attachments/assets/e034fa5e-f899-4237-b7cb-5b2c8a432507 "Stripplot da fração de terra urbana para cada classificação de seca")
![Stripplot da fração de área cultivada irrigada por chuva para cada classificação de seca](https://github.com/user-attachments/assets/e665eead-e747-4bd5-9d7a-1b92419176f9 "Stripplot da fração de área cultivada irrigada por chuva para cada classificação de seca")
![Stripplot da fração de área cultivada irrigada por sistema de irrigação para cada classificação de seca](https://github.com/user-attachments/assets/ef46b837-57fa-4fb3-a22e-2d43c3175bd6 "Stripplot da fração de área cultivada irrigada por sistema de irrigação para cada classificação de seca")
![Stripplot da fração de área com inclinação do tipo 1 para cada classificação de seca](https://github.com/user-attachments/assets/85caefea-c136-4f7b-b91b-78728029a724 "Stripplot da fração de área com inclinação do tipo 1 para cada classificação de seca")
![Stripplot da longitude de cada região para cada classificação de seca](https://github.com/user-attachments/assets/c80a5313-4b49-4ffb-8e58-b3f71c09e6ae "Stripplot da longitude de cada região para cada classificação de seca")

Outro ponto avaliado foi a independência das séries temporais de cada condados (fips), massas de ar frio, quente, úmidas ou secas se deslocam alterando o clima a medida que se movem, Foi realizado um teste causalidade de Granger para verificar se o estado de um condado no tempo x pode ter efeito no estado de outro condado num tempo x + lag. Realizando o teste de Granger, com nível de confiança de 95% e entre um condado do Nebraska com um do Colorado dois de Iowa (estados vizinhos) e outro com Texas(estado não vizinho) - em tese não de javer depednência das séries temproais entre regiões muito distantes -  para um lag de 6 obtivemos que para:

fips: 31147 (Nebraska) e 8053(Colorado) o teste foi positivo para lags de 1 a 6
fips: 31147 e 19035 (Iowa) o teste foi positivo para lags de 3 a 6
fips: 31147 e 31181 (Nebraska) o teste foi positivo para lags de 1 a 6
fips: 31147 e 48089 (Texas) o teste foi negativo para todos os lags
fips: 31147 e 19113 (Iowa) o teste foi positivo para lags de 3 a 6

O teste de causalidade de Granger indica que há dependência entre alguns pares de séries temporais, porém ele não elimina a possibilidade do acaso. No entanto estes resultado associados aos fenômenos metereológicos conhecidos pode-se concluir que a uma causalidade entre as séries temporais.

## Criação do modelo de Machine Learning

Um modelo de redes neurais muito utilizado para trabalhar com sequências sejam textos ou timeseries é a modelo de rede  LSTM (Long short-term memory), uma estrutura de rede neural artificial recorrente com conexões de feedback. Porém como os dados são compostos de múltiplas séries multivariadas correlacionadas apenas a LSTM não seria capaz de ver e interpretar essa relação entre séries. Por isso será utilizado redes convolucionais (CNNs) juntamente com a LSTM.

A abordagem de CNN-LSTM para predição de séries temporais já é amplamente abordada com a inspiração para este trabalho vindo do artigo Multivariate CNN-LSTM Model for Multiple Parallel Financial Time-Series Prediction de autoria de  Harya Widiputra, Adele Mailangkay , e Elliana Gautam.

Devido a dificuldades do terinamento e manipulação dos dados do modelo em um computador pessoal, a base foi novamente reduzida, limitando apenas para o estado do Colorado.

As redes neurais serão construídas utilizando keras via tensorflow.

O primeiro modelo foi construído utilizando 2 camadas CNNs com a primeira com 16 filtros e a segunda com 32, ambas com kernel de tamanho 3 e função de ativação relu e MaxPolling de tamanho 2 após cada camada. Devido à dimensão tempo da série temporária todas as camadas com exceção da LSTM serão envolvidas por camadas TimeDistributed.

Após o segundo MaxPooling o output é passado por um Dropout com razão de dropout de 0,2 e por uma camada Flatten preparando o dado para ser imputado na camada LSTM.
A camada LSTM é composta de 64 unidades com um dropout recorrente de 0,5. O output da LSTM passa por um reshape preparando para uma última camada convolucional, dessa vez com 6 filtros (o número de classes do nosso problema) com kernel de tamanho e função de ativação softmax. A função desta última camada é retornar as probabilidades das classes.
No total o modelo tem 154,012 parâmetros para treinar.

O treinamento foi realizado utilizando cross-entropia categórica, otimizador adam com taxa de aprendizagem de 0,001 e acurácia como métrica com 100 épocas. Devido ao fato de o dataset de validação possuir 104 time steps e o de treino ter 887 e a necessidade do batchsize para treinamento e validação serem os mesmos o modelo foi treinado múltiplas vezes sobre intervalos de tempo diferentes com 104 time steps

![image](https://github.com/user-attachments/assets/6a1e82e9-b9bc-4fd1-9af8-bb7e84cdf976 "Modelo 1")

O treinamento foi realizado utilizando cross-entropia categórica, otimizador adam com taxa de aprendizagem de 0,001 e acurácia como métrica com 100 épocas.

Devido ao fato de o dataset de validação possuir 104 time steps e o de treino ter 887 e a necessidade do batchsize para treinamento e validação serem os mesmos o modelo foi treinado múltiplas vezes sobre intervalos de tempo diferentes com 104 time steps

Este modelo teve uma acurácia de treino de 69% e 39% de acurácia de validação.

Com intuito de melhorar o desempenho do modelo e aproximar a acurácia de validação do treino foi alterado tamanho do kernel da primeira camada convolucional, tamanho da poll na primeira  MaxPolling , e razões de dropout após a CNN e após a LSTM

![image](https://github.com/user-attachments/assets/5ead334d-736d-464d-bdb8-39ad66939fd4 "Modelo 2")

Este segundo modelo teve o mesmo desempenho do primeiro.

Foi feito tentativas entre vários outros modelos, alterando a taxa de aprendizagem do otimizadores, acrescentando mais camadas CNN e LSTM, alterando o polling de max para média, aumentando o número de épocas do treinamento de 100 para 1000, mas nenhum dos modelos teve um desempenho superior a 69% para o treino e 39% para a validação.

O último modelo treinado e utilizado para o teste é o modelo abaixo.

![image](https://github.com/user-attachments/assets/6f02594b-208c-415c-9b65-e797ee54c5fa "Modelo final")

O desempenho no modelo do teste não foi bom, além da baixa acurácia os pontos que mostram a baixa qualidade do modelo para representação do problema estão nos recalls das classes. O modelo teve um recall de 100% para classe 0 (condição climática normal) e zero para as demais. Ou seja, o modelo está classificando tudo como sendo condição normal.

![image](https://github.com/user-attachments/assets/00acd4f8-93db-4e32-90ba-34352f558cde "Scores do modelo")


# Conclusão

 A análise exploratória mostrou que há pouca diferença entre a condição de seca e os dados meteorológicos e geográficos, isso pode ser uma das razões para o baixo desempenho do modelo em predizer a condição climática. Esta semelhança somada ao imbalanço do dataset quanto às classes dificulta a construção de um modelo que realmente busca identificar as classes e não maximizar sua acurácia enviesando sua classificação para a classe mais comum.

Um insight importante da análise que impactou a escolha do modelo foi a correlação entre as séries temporais de regiões diferentes, esse insight mostrou necessário a escolha de um modelo capaz de identificar essas relações por isso a escolha de camadas CNN antes da LSTM.

Uma solução para este problema poderia ser utilizar dados de mais estados adicionando mais informações ao modelo assim como a utilização do dataset de solo, como observado na exploração algumas das variáveis aparentam ter comportamento correlacionado à intensidade da seca


## Notebooks

Os notebooks com os códigos utilizado para otrabalho podem ser acessados nos links abaixo

https://github.com/GuilhermeGAraujo/Trabalho_Integrado/blob/master/Reducao_dataset.ipynb
https://github.com/GuilhermeGAraujo/Trabalho_Integrado/blob/master/Processamento_e_visualizacao.ipynb
https://github.com/GuilhermeGAraujo/Trabalho_Integrado/blob/master/Testes.ipynb
https://github.com/GuilhermeGAraujo/Trabalho_Integrado/blob/master/shape_data.ipynb
https://github.com/GuilhermeGAraujo/Trabalho_Integrado/blob/master/modelagem.ipynb.
