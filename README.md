
# Projeto de Sistemas Inteligentes I - FP-Growth, DBSCAN e K-Médias

## Aluna:

- Isabely Toledo de Melo

## Descrição do Projeto

Este repositório apresenta a implementação manual de três algoritmos estudados na disciplina de Sistemas Inteligentes I, cada um resolvendo uma tarefa distinta de análise de dados:

- **FP-Growth**: Descoberta de padrões frequentes e regras de associação em um conjunto de transações de supermercado.
- **DBSCAN**: Algoritmo de agrupamento baseado em densidade aplicado ao conjunto de dados sintético `Two Moons`.
- **K-Médias**: Algoritmo de agrupamento baseado em centróides, aplicado a dados gerados com `make_blobs`.

As implementações foram feitas do zero, sem o uso de bibliotecas de machine learning como `scikit-learn` ou `mlxtend`, conforme as diretrizes da avaliação. Apenas bibliotecas auxiliares (`numpy`, `pandas`, `matplotlib`, `os`) foram utilizadas para manipulação de dados e visualizações.

## Estrutura de Arquivos

```
├── questao-1/
│   ├── questao-1.py
│   ├── Market_Basket_Optimisation.csv
│   └── graficos/
│       ├── itens_frequentes.png
│       ├── conjuntos_frequentes.png
│       └── regras_associacao.png
│
├── questao-2/
│   ├── questao-2.py
│   └── graficos/
│       ├── grafico_dbscan.png
├── questao-3/
│   ├── questao-3.py
│   └── graficos/
│       ├── original.png
│       └── kmeans_clusters.png
│
└── README.md
```

## Instruções de Execução

1. Instale as dependências:
```bash
pip install numpy pandas matplotlib
```

2. Execute os scripts de cada questão:

### Questão 1 - FP-Growth

```bash
python questao-1/questao-1.py
```

- **Arquivo de entrada**: `Market_Basket_Optimisation.csv`
- **Parâmetros**:
  - Suporte mínimo: 300
  - Confiança mínima: 0.3
- **Saídas**:
  - Gráficos de itens frequentes, conjuntos frequentes e regras de associação
  - Impressão dos padrões mais relevantes no terminal

### Questão 2 - DBSCAN (Two Moons)

```bash
python questao-2/questao-2.py
```

- **Dados gerados manualmente (tipo "two moons")**
- **Parâmetros**:
  - `eps = 0.15`
  - `min_samples = 5`
- **Saídas**:
  - Gráfico dos clusters gerados pelo DBSCAN (`dbscan_clusters.png`)

### Questão 3 - K-Médias (Blobs)

```bash
python questao-3/questao-3.py
```

- **Dados gerados com `make_blobs`**
- **Parâmetros**:
  - `k = 4`
- **Saídas**:
  - Gráfico dos dados originais (`original.png`)
  - Gráfico dos clusters gerados pelo K-Médias (`kmeans_clusters.png`)
  - Impressão da contagem de pontos por cluster no terminal

## Considerações Finais

- Todos os algoritmos foram implementados manualmente conforme especificado na prova.
- O projeto reforça o entendimento prático dos principais algoritmos de agrupamento e descoberta de padrões frequentes.
- Os gráficos foram organizados por questão em suas respectivas pastas.
