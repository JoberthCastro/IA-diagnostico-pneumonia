# Classificador de Pneumonia usando Deep Learning

Este projeto implementa um classificador de imagens de raio-X para detecção de pneumonia utilizando uma rede neural convolucional (CNN) com TensorFlow/Keras.

## Arquitetura do Modelo

O modelo utiliza uma arquitetura CNN com as seguintes características:
- 3 camadas convolucionais com ReLU
- Camadas de MaxPooling
- Camadas densas com dropout para regularização
- Função de ativação sigmoid na saída (classificação binária)

## Pré-processamento das Imagens

As imagens são pré-processadas com:
- Conversão para escala de cinza
- Equalização de histograma
- Filtro gaussiano para redução de ruído
- Normalização para valores entre 0 e 1
- Redimensionamento para 225x225 pixels

## Resultados do Treinamento

### Configuração de Treinamento
- Número de épocas (EPOCHS): **15**

### Métricas de Performance
- Acurácia no conjunto de teste: 0.8750 (87.50%)
- Relatório de Classificação:
  ```
                precision    recall  f1-score   support
             0       0.88      0.87      0.87       234
             1       0.87      0.88      0.87       390
      accuracy                           0.87       624
     macro avg       0.87      0.87      0.87       624
  weighted avg       0.87      0.87      0.87       624
  ```

### Matriz de Confusão
```
[[203  31]
 [ 47 343]]
```
- Verdadeiros Negativos: 203
- Falsos Positivos: 31
- Falsos Negativos: 47
- Verdadeiros Positivos: 343

### Gráficos Gerados Automaticamente
O script gera automaticamente os seguintes gráficos para análise de desempenho:
- `training_history.png`: Histórico de acurácia e perda durante o treinamento
- `confusion_matrix.png`: Matriz de confusão do conjunto de teste
- `classification_metrics.png`: Gráfico de barras com precisão, revocação (recall) e F1-score por classe e médias
- `roc_curve.png`: Curva ROC com valor de AUC

## Estrutura do Projeto

```
.
├── Dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── pneumonia_classifier.py
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- pandas

## Instalação

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Organize suas imagens nos diretórios train/val/test conforme a estrutura acima
2. Execute o script principal:
```bash
python pneumonia_classifier.py
```

O script irá:
- Treinar o modelo por 15 épocas
- Salvar o melhor modelo como 'best_model.h5'
- Gerar gráficos de treinamento em 'training_history.png'
- Gerar matriz de confusão em 'confusion_matrix.png'
- Gerar gráfico de métricas por classe em 'classification_metrics.png'
- Gerar curva ROC em 'roc_curve.png'

## Observações

- O modelo foi treinado com data augmentation para melhor generalização
- Early stopping foi implementado para evitar overfitting
- O melhor modelo é salvo baseado na acurácia de validação

# Detecção de Pneumonia Viral com Radiografias

## 📁 Fonte dos Dados

Os dados utilizados neste projeto foram obtidos da seguinte base disponível no Kaggle:

🔗 [COVID-19 Radiography Database - Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

---

## 📊 Divisão do Conjunto de Dados

O conjunto de dados foi dividido em duas categorias principais: **radiografias normais** e **radiografias com pneumonia**. Cada categoria foi subdividida em três partes: treino, validação e teste, conforme a tabela abaixo:

| Conjunto     | Pneumonia | Normal | Total |
|--------------|-----------|--------|-------|
| Treinamento  | 807       | 1.200  | 2.007 |
| Validação    | 269       | 400    | 669   |
| Teste        | 269       | 400    | 669   |

---

## 🧠 Objetivo

O objetivo principal do projeto é desenvolver um modelo de classificação que seja capaz de identificar casos de pneumonia viral com base em imagens de radiografias torácicas.

Este documento será atualizado conforme o projeto avançar com novas etapas de pré-processamento, modelagem e avaliação de desempenho.

---

## 🚧 Status

🔄 Projeto em fase inicial: organização dos dados e planejamento da modelagem.

---

## 👥 Autores

- **Joberth Castro**  
  [GitHub](https://github.com/JoberthCastro) | [LinkedIn](https://www.linkedin.com/in/joberth-castro-013840252/)

- **Maria Clara Cutrim**  
  [GitHub](https://github.com/MariaclaraCutrim) | [LinkedIn](https://www.linkedin.com/in/maria-clara-cutrim-nunes-costa-55b7a8248/)

- **Maria Fernanda Mirabile**  
  [GitHub](https://github.com/mfernandamirabile) | [LinkedIn](https://www.linkedin.com/in/fernanda-mirabile/)
