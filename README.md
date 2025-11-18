🎓 Projeto Individual 1 (PI1): Análise de Evasão e Previsão de Retorno

Este o Projeto Individual 1 foi desenvolvido para a disciplina Mineração de Dados e é focado na aplicação de técnicas de Machine Learning Supervisionado para resolver o problema de evasão no ensino superior.

📋 Sobre o Projeto

O Problema:

O trancamento de matrículas e a evasão representam perdas significativas para instituições de ensino, tanto financeiras (perda de receita recorrente) quanto acadêmicas. Tentar reengajar todos os alunos indistintamente é ineficiente e custoso.

Solução:

Uma aplicação interativa desenvolvida em Python e Streamlit que utiliza dados históricos para calcular a probabilidade de retorno de um aluno. O sistema permite:

Identificar padrões de comportamento em alunos que retornaram versus os que evadiram.

Simular cenários em tempo real para apoiar a tomada de decisão.

Comparar modelos de Inteligência Artificial para garantir a melhor precisão.

🚀 Funcionalidades

Carregamento Inteligente de Dados: O sistema detecta e carrega automaticamente a base de dados (.csv ou .xlsx) presente no diretório.

ETL Automatizado: Tratamento de valores nulos, conversão de variáveis categóricas (One-Hot Encoding) e normalização de dados.

Comparação de Modelos: Treinamento e avaliação simultânea de dois algoritmos de Ensemble:

🌲 Random Forest (Floresta Aleatória)

⚡ Gradient Boosting

Simulador Interativo: Uma barra lateral que permite ajustar o perfil do aluno (idade, média, dívida) e ver a probabilidade de retorno mudar instantaneamente.

Dashboard Visual: Gráficos de distribuição e boxplots para análise exploratória dos dados.

🛠️ Tecnologias Utilizadas

Linguagem: Python 3.x

Interface Web: Streamlit

Manipulação de Dados: Pandas, NumPy

Machine Learning: Scikit-learn (RandomForestClassifier, GradientBoostingClassifier)

Visualização: Matplotlib, Seaborn

📂 Estrutura de Arquivos

Certifique-se de que os seguintes arquivos estejam no mesmo diretório:

📁 projeto_pi1/
│
├── 📄 app.py                          # O código fonte da aplicação
├── 📄 base_alunos_trancamento.csv     # A base de dados (ou .xlsx)
├── 📄 README.md                       # Este arquivo de documentação
└── 📄 requirements.txt                # Lista de dependências (opcional)


⚙️ Como Executar o Projeto

1. Pré-requisitos

Certifique-se de ter o Python instalado. Em seguida, instale as bibliotecas necessárias executando o comando abaixo no terminal:

pip install streamlit pandas numpy matplotlib seaborn scikit-learn openpyxl


2. Executando a Aplicação

Navegue até a pasta do projeto pelo terminal e execute:

streamlit run app.py


O navegador abrirá automaticamente no endereço http://localhost:8501 exibindo a aplicação.

📊 Análise dos Resultados

A aplicação exibe a acurácia dos modelos nos dados de teste. Historicamente, observamos que variáveis como 'divida_pendente' e 'media_anterior' são os fatores mais determinantes para a previsão.

Alta Probabilidade (> 70%): Indica forte tendência de retorno.

Probabilidade Moderada (40% - 70%): Indica indecisão; ideal para campanhas de marketing ou renegociação.

Baixa Probabilidade (< 40%): Retorno difícil; requer avaliação de custo-benefício.

Desenvolvido para a disciplina de Mineração de Dados / Projeto Individual.
