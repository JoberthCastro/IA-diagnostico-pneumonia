# Sistema de Diagnóstico de Pneumonia por IA

Este sistema utiliza inteligência artificial para diagnosticar pneumonia viral através de imagens de raio-x do tórax. O modelo foi treinado com um dataset balanceado de imagens normais e casos de pneumonia viral.

## 🚨 Download do Modelo Treinado

➡️ O modelo treinado otimizado **`best_model.keras`** pode ser baixado através do seguinte link:  
**[👉 Download do Modelo Treinado](https://drive.google.com/file/d/1jXTWGgX3iQofwEfT4088TGazn4mhYVne/view?usp=drive_link)**

---

## Base de Dados

O modelo foi treinado com a base pública disponível em:  
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

## Arquivos do Sistema

- `train.py`: Script para treinar o modelo de IA e fazer download da base de dados
- `classifier.py`: Módulo com funções para predição (para uso em produção)
- `best_model.keras`: Modelo treinado otimizado
- `requirements.txt`: Dependências do projeto

## Requisitos do Sistema

Todas as dependências necessárias estão listadas no `requirements.txt` e foram testadas em um ambiente virtual limpo. Para instalar:

```bash
pip install -r requirements.txt
```

## Como Usar o Sistema de Predição

1. Copie os seguintes arquivos para seu projeto:
   - `best_model.keras`
   - As funções de predição do `classifier.py`

2. Importe as dependências necessárias:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model
```

3. Utilize as duas funções principais:

### Função de Pré-processamento

```python
def preprocess_image(image):
    """
    Pré-processa a imagem para o formato esperado pelo modelo

    Parâmetros:
    - image: np.ndarray, imagem em formato BGR (formato padrão do OpenCV)

    Retorna:
    - np.ndarray: imagem pré-processada com shape (1, 225, 225, 1)
    """
    IMG_SIZE = 225
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # batch
    image = np.expand_dims(image, axis=-1)  # canal
    return image
```

### Função de Predição

```python
def carregar_modelo_e_prever(caminho_modelo, imagem_array):
    """
    Carrega o modelo e faz a predição

    Parâmetros:
    - caminho_modelo: str, caminho para o arquivo best_model.keras
    - imagem_array: np.ndarray, imagem já pré-processada

    Retorna:
    - np.ndarray: array com a probabilidade de pneumonia viral (0 a 1)
    """
    modelo = load_model(caminho_modelo)
    if len(imagem_array.shape) == 3:
        imagem_array = np.expand_dims(imagem_array, axis=0)
    predicao = modelo.predict(imagem_array)
    return predicao
```

## Exemplo de Uso

```python
# Carregar e pré-processar a imagem
imagem = cv2.imread('caminho/para/imagem.jpg')
imagem_processada = preprocess_image(imagem)

# Fazer a predição
probabilidade = carregar_modelo_e_prever('best_model.keras', imagem_processada)

# A probabilidade é um número entre 0 e 1
# Valores próximos de 1 indicam alta probabilidade de pneumonia viral
# Valores próximos de 0 indicam alta probabilidade de normal
print(f"Probabilidade de pneumonia viral: {float(probabilidade[0]):.4f}")
```

## Notas Importantes

1. O modelo espera imagens em escala de cinza com dimensões 225x225 pixels  
2. As imagens são automaticamente normalizadas para valores entre 0 e 1  
3. A saída é uma probabilidade entre 0 e 1:
   - Valores > 0.5 sugerem pneumonia viral
   - Valores < 0.5 sugerem normal  
4. O modelo foi treinado com um dataset balanceado e validado com métricas de performance

## Métricas do Modelo

| Classe | Precision | Recall | F1-Score | Suporte |
|--------|----------|-------|---------|--------|
| **0** (Normal) | 0.98 | 0.97 | 0.97 | 1044 |
| **1** (Pneumonia Viral) | 0.96 | 0.97 | 0.96 | 785 |

- **Acurácia geral**: 97%  
- **Média Macro**: Precision 0.97, Recall 0.97, F1-Score 0.97  
- **Média Ponderada**: Precision 0.97, Recall 0.97, F1-Score 0.97

Total de amostras: **1829**

## Autores

**Joberth Castro**  
[LinkedIn](https://www.linkedin.com/in/joberth-castro-013840252/)  
[GitHub](https://github.com/JoberthCastro)  
[WhatsApp](https://wa.me/559885864235?text=Ol%C3%A1%2C%20gostaria%20de%20falar%20sobre%20o%20reposit%C3%B3rio%20IA-diagnostico-pneumonia%2C%20podemos%20conversar%3F)

**Maria Clara Cutrim**  
[LinkedIn](https://www.linkedin.com/in/maria-clara-cutrim-nunes-costa-55b7a8248/)  
[GitHub](https://github.com/MariaclaraCutrim)  
[WhatsApp](https://wa.me/559885316743?text=Ol%C3%A1%2C%20gostaria%20de%20falar%20sobre%20o%20reposit%C3%B3rio%20IA-diagnostico-pneumonia%2C%20podemos%20conversar%3F)

**Maria Fernanda Mirabile**  
[LinkedIn](https://www.linkedin.com/in/fernanda-mirabile/)  
[GitHub](https://github.com/mfernandamirabile)  
[WhatsApp](https://wa.me/559881850349?text=Ol%C3%A1%2C%20gostaria%20de%20falar%20sobre%20o%20reposit%C3%B3rio%20IA-diagnostico-pneumonia%2C%20podemos%20conversar%3F)
