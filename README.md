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
    

