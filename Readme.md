# Predição de Lutas do UFC com Machine Learning

Trabalho final da disciplina **Treinamento e Otimização de Modelos de Machine Learning**.

Neste projeto utilizamos o dataset **UFC Datasets 1994–2025** (Kaggle) para construir modelos de Machine Learning capazes de:

1. Prever **quem vence a luta** (corner vermelho ou azul);  
2. Prever o **método de vitória** (Decisão, KO/TKO, Finalização, Other);  
3. Estimar a **duração da luta em segundos**.

O notebook foi desenvolvido em Python (Jupyter/Colab), com foco em boas práticas de limpeza de dados, engenharia de atributos, validação cruzada e comparação entre diferentes modelos.

---

## 1. Motivação

O UFC é uma das maiores organizações de MMA do mundo e gera uma grande quantidade de dados a cada evento: estatísticas de golpes, quedas, tempo de luta, categorias de peso, estilos dos lutadores, entre outros.

Esses dados permitem responder perguntas relevantes para:

- **Equipes técnicas e analistas de performance** – entender quais características mais influenciam vitórias, métodos de luta e tempo de combate;  
- **Fãs e comentaristas** – apoiar análises com dados objetivos;  
- **Estudos acadêmicos** – aplicação de técnicas de Machine Learning em esportes de combate.

O objetivo do projeto é mostrar, de forma prática, como transformar esses dados em modelos preditivos que resolvem problemas reais.

---

## 2. Problemas de Machine Learning

A partir do arquivo consolidado `UFC.csv` (uma linha por luta), foram formulados **três problemas supervisionados**:

### 2.1 Problema 1 – Classificação binária (vencedor da luta)

- **Pergunta:** qual corner tem maior probabilidade de vencer a luta – vermelho ou azul?  
- **Target:** `winner_bin`  
  - `1` → vencedor é o lutador do corner vermelho  
  - `0` → vencedor é o lutador do corner azul  

### 2.2 Problema 2 – Classificação multiclasse (método de vitória)

- **Pergunta:** qual será o método de vitória mais provável?  
- **Target:** `method_simplified`, com as seguintes classes:
  - `Decision`
  - `KO/TKO`
  - `Submission`
  - `Other`

### 2.3 Problema 3 – Regressão (duração da luta)

- **Pergunta:** qual é a duração esperada da luta, em segundos?  
- **Target:** `duration_sec`, derivada da coluna `match_time_sec`.

---

## 3. Dataset

- **Fonte:** Kaggle – [UFC Datasets 1994–2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)  
- **Arquivo principal utilizado:** `UFC.csv`  
- **Tamanho aproximado:**  
  - ~8.300 lutas  
  - 120+ colunas, incluindo:
    - informações do evento (`event_name`, `date`, `location`, `division`, `title_fight`);  
    - atributos do lutador vermelho (`r_*`) e do lutador azul (`b_*`);  
    - estatísticas de golpes, quedas, controle, etc.;  
    - resultado (`winner`, `method`, `finish_round`, `match_time_sec`, `total_rounds`).

---

## 4. Metodologia

### 4.1 Carregamento e limpeza

- Leitura do arquivo `UFC.csv` diretamente a partir do caminho baixado via `kagglehub`.  
- Conversão da coluna `date` para o tipo `datetime` e remoção de lutas sem data válida.  
- Remoção de registros sem informação do target em cada problema (ex.: lutas sem vencedor/método/duração).

### 4.2 Split temporal (treino x teste)

Para evitar *data leakage* temporal, o particionamento foi feito por data:

- **Treino:** lutas com `date < 2022-01-01`  
- **Teste:** lutas com `date ≥ 2022-01-01`  

Isso garante que os modelos sejam treinados apenas com lutas mais antigas e avaliados em lutas recentes, simulando o uso em produção.

### 4.3 Engenharia de atributos

Foi criada uma série de **features de diferença** entre lutador vermelho (`r_*`) e azul (`b_*`), por exemplo:

- `kd_diff` = `r_kd` − `b_kd`  
- `sig_str_diff` = `r_sig_str_landed` − `b_sig_str_landed`  
- `td_diff` = `r_td_landed` − `b_td_landed`  
- `sub_att_diff` = `r_sub_att` − `b_sub_att`  
- `height_diff` = `r_height` − `b_height`  
- `weight_diff` = `r_weight` − `b_weight`  
- `reach_diff` = `r_reach` − `b_reach`  

Essas variáveis representam a **vantagem relativa** de um lutador sobre o outro em aspectos físicos e técnicos, o que se mostrou bastante informativo.

### 4.4 Remoção de *data leakage*

Para cada problema foram removidas das features todas as colunas que revelam diretamente o resultado da luta, tais como:

- `winner`, `winner_id`, `winner_bin`;  
- `method`, `method_simplified`;  
- `finish_round`, `match_time_sec`, `total_rounds`, `duration`, `duration_sec` (quando é target);  
- IDs e nomes (`event_id`, `event_name`, `fight_id`, `r_name`, `b_name`, etc.).

### 4.5 Pré-processamento

Foi utilizado um pipeline com `ColumnTransformer`:

- **Variáveis numéricas:**
  - imputação de valores faltantes com mediana (`SimpleImputer`);  
  - padronização com `StandardScaler`.  

- **Variáveis categóricas:**  
  - imputação com valor mais frequente;  
  - codificação *one-hot* (`OneHotEncoder`).

Esse pipeline é reutilizado em todos os modelos, garantindo consistência e facilitando validação cruzada.

---

## 5. Modelos e Validação

Para cada problema foram treinados **dois modelos diferentes**, atendendo ao requisito de comparação.

### 5.1 Problema 1 – Vencedor (classificação binária)

Modelos testados:

1. **Regressão Logística**
2. **Random Forest Classifier**

Métricas principais: **Accuracy** e **AUC-ROC**.

**Resultados no conjunto de teste:**

- **Regressão Logística**  
  - Accuracy ≈ **0,884**  
  - AUC-ROC ≈ **0,947**

- **Random Forest Classifier**  
  - Accuracy ≈ 0,862  
  - AUC-ROC ≈ 0,945  

A Regressão Logística apresentou a melhor acurácia e um comportamento mais equilibrado entre vitórias do corner vermelho e azul, sendo escolhida como modelo final para este problema.

> **Modelo escolhido (Problema 1): Regressão Logística.**

---

### 5.2 Problema 2 – Método de vitória (multiclasse)

Classes de `method_simplified` no treino:  
Decision (~45,5%), KO/TKO (~33,1%), Submission (~19,9%), Other (~1,4%).

Modelos testados:

1. **Regressão Logística multinomial**  
2. **Random Forest Classifier** (`class_weight="balanced"`)

Métricas principais: **Accuracy** e **F1-score macro**.

**Resultados no conjunto de teste:**

- **Regressão Logística multinomial**  
  - Accuracy ≈ **0,832**  
  - F1 macro ≈ **0,631**

- **Random Forest Classifier**  
  - Accuracy ≈ 0,786  
  - F1 macro ≈ 0,561  

A LogReg multinomial apresentou melhor desempenho global e equilíbrio entre as principais classes (Decision, KO/TKO, Submission). O Random Forest mostrou viés mais forte para a classe `Decision`.

> **Modelo escolhido (Problema 2): Regressão Logística multinomial.**

---

### 5.3 Problema 3 – Duração da luta (regressão)

Target: `duration_sec` (tempo total da luta em segundos).  
Duração média das lutas no teste: ~232,6 s (desvio padrão ~89,4 s).

Modelos testados:

1. **Regressão Linear**  
2. **Random Forest Regressor**

Métricas principais: **RMSE**, **MAE** e **R²**.

**Resultados no conjunto de teste:**

- **Regressão Linear**
  - RMSE ≈ 132,96 s  
  - MAE ≈ 98,76 s  
  - R² ≈ **−1,212** (pior que prever sempre a média)

- **Random Forest Regressor**
  - RMSE ≈ **60,79 s**  
  - MAE ≈ **38,81 s**  
  - R² ≈ **0,538**

O Random Forest reduziu o erro médio em aproximadamente 60% em relação à Regressão Linear e passou a explicar cerca de 54% da variância da duração da luta, mostrando-se muito mais adequado para esse problema.

> **Modelo escolhido (Problema 3): Random Forest Regressor.**

---

## 6. Tecnologias Utilizadas

Principais bibliotecas:

- `numpy`  
- `pandas`  
- `matplotlib`  
- `scikit-learn`  
- `kagglehub` (para download do dataset)  

Opcionalmente, o projeto pode ser estendido com:

- `seaborn` (visualizações)  
- `xgboost`, `lightgbm` (modelos adicionais)  
- `shap` (interpretação de modelos complexos)

---

## 7. Como Executar

### 7.1 Pré-requisitos

- Python 3.9+  
- Jupyter Notebook ou Google Colab  

### 7.2 Passos

1. Clonar o repositório ou fazer download dos arquivos.  
2. Instalar as dependências (exemplo):

   ```bash
   pip install -r requirements.txt
