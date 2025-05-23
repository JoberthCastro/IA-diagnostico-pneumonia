import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Configurações
IMG_SIZE = 225

def preprocess_image(image):
    """
    Pré-processa a imagem para o formato esperado pelo modelo
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # batch
    image = np.expand_dims(image, axis=-1)  # canal
    return image

def carregar_modelo_e_prever(caminho_modelo, imagem_array):
    """
    Carrega um modelo .keras e faz a predição de uma imagem fornecida como array.
    Parâmetros:
    - caminho_modelo: str, caminho para o arquivo .keras.
    - imagem_array: np.ndarray, imagem já pré-processada e com shape adequado ao modelo (ex: (1, altura, largura, canais)).
    Retorna:
    - np.ndarray com as predições do modelo.
    """
    modelo = load_model(caminho_modelo)
    if len(imagem_array.shape) == 3:
        imagem_array = np.expand_dims(imagem_array, axis=0)
    predicao = modelo.predict(imagem_array)
    return predicao


# Teste da função: 

"""# Caminhos das pastas
pasta_normal = r'C:\Users\Sony Vaio\Desktop\IA-diagnostico-pneumonia-main\dataset\COVID-19_Radiography_Dataset\Normal\images'
pasta_viral = r'C:\Users\Sony Vaio\Desktop\IA-diagnostico-pneumonia-main\dataset\COVID-19_Radiography_Dataset\Viral Pneumonia\images'

# Pegar as 10 primeiras imagens de Normal
arquivos_normal = sorted([f for f in os.listdir(pasta_normal) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
nomes_normal = arquivos_normal[:10]
caminhos_normal = [os.path.join(pasta_normal, nome) for nome in nomes_normal]

# Pegar as 10 últimas imagens de Viral Pneumonia
arquivos_viral = sorted([f for f in os.listdir(pasta_viral) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
nomes_viral = arquivos_viral[-10:]
caminhos_viral = [os.path.join(pasta_viral, nome) for nome in nomes_viral]

# Juntar tudo
caminhos = caminhos_normal + caminhos_viral

# Pré-processar todas as imagens selecionadas
imagens_processadas = []
for caminho in caminhos:
    img = cv2.imread(caminho)
    img_proc = preprocess_image(img)
    imagens_processadas.append(img_proc[0])  # remove a dimensão do batch para empilhar

# Empilhar todas as imagens em um array (N, 225, 225, 1)
batch = np.stack(imagens_processadas, axis=0)

# Fazer predição em lote
probs = carregar_modelo_e_prever('best_model.keras', batch)  # shape (N, 1)

# Mostrar apenas os números float
for prob in probs:
    print(f"{float(prob[0]):.4f}")