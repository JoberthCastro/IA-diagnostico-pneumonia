import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import seaborn as sns
import cv2
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import kagglehub

# Configurações
IMG_SIZE = 225
BATCH_SIZE = 32
EPOCHS = 50

def download_database(dataset_name):
    # Caminho da pasta do usuário
    home_dir = os.path.expanduser("~")

    # Caminho onde os datasets do kagglehub são armazenados
    dataset_folder = dataset_name.replace("/", os.sep)  
    dataset_path = os.path.join(home_dir, ".cache", "kagglehub", "datasets", dataset_folder, "versions", "5")

    # Verifica se o dataset já foi instalado
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("Baixando o dataset...")
        path = kagglehub.dataset_download(dataset_name)
        print("Download concluído! Arquivos em:", path)
    else:
        print("Dataset já instalado. Usando arquivos locais em:", dataset_path)
        
    return dataset_path

def prepare_dataset(base_dir):
    """
    Prepara o dataset dividindo em treino (60%), validação (20%) e teste (20%)
    mantendo o balanceamento entre as classes
    """
    print("Preparando o dataset...")

    # Diretórios de origem (agora usando a subpasta 'images')
    normal_dir = os.path.join(base_dir, 'Normal', 'images')
    pneumonia_dir = os.path.join(base_dir, 'Viral Pneumonia', 'images')
    
    print(f"\nVerificando diretórios:")
    print(f"Normal: {normal_dir}")
    print(f"Pneumonia: {pneumonia_dir}")

    # Verificar se os diretórios existem
    if not os.path.exists(normal_dir):
        print(f"Erro: Diretório {normal_dir} não encontrado!")
        return None, None, None
    if not os.path.exists(pneumonia_dir):
        print(f"Erro: Diretório {pneumonia_dir} não encontrado!")
        return None, None, None
    
    # Listar imagens de cada classe (apenas arquivos de imagem)
    def get_image_files(directory):
        image_files = []
        for file in os.listdir(directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(file)
        return image_files
    
    normal_images = get_image_files(normal_dir)
    pneumonia_images = get_image_files(pneumonia_dir)
    
    print(f"\nEncontradas:")
    print(f"Imagens normais: {len(normal_images)}")
    print(f"Imagens de pneumonia: {len(pneumonia_images)}")
    
    if len(normal_images) == 0 or len(pneumonia_images) == 0:
        print("Erro: Nenhuma imagem encontrada em um ou mais diretórios!")
        return None, None, None
    
    # Encontrar o número mínimo de imagens entre as classes
    min_images = min(len(normal_images), len(pneumonia_images))
    print(f"\nUsando {min_images} imagens por classe para manter o balanceamento")
    
    # Criar diretórios de destino
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, 'Normal'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'Viral Pneumonia'), exist_ok=True)
    
    # Embaralhar as imagens
    np.random.shuffle(normal_images)
    np.random.shuffle(pneumonia_images)
    
    # Dividir as imagens
    train_size = int(min_images * 0.6)
    val_size = int(min_images * 0.2)
    
    # Função para copiar imagens
    def copy_images(images, source_dir, target_dir, start, end):
        copied = 0
        for img in images[start:end]:
            try:
                src = os.path.join(source_dir, img)
                dst = os.path.join(target_dir, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
            except Exception as e:
                print(f"Erro ao copiar {img}: {str(e)}")
        return copied
    
    print("\nCopiando imagens...")
    
    # Copiar imagens para os diretórios
    # Normal
    train_normal = copy_images(normal_images, normal_dir, os.path.join(train_dir, 'Normal'), 0, train_size)
    val_normal = copy_images(normal_images, normal_dir, os.path.join(val_dir, 'Normal'), train_size, train_size + val_size)
    test_normal = copy_images(normal_images, normal_dir, os.path.join(test_dir, 'Normal'), train_size + val_size, min_images)
    
    # Pneumonia
    train_pneumonia = copy_images(pneumonia_images, pneumonia_dir, os.path.join(train_dir, 'Viral Pneumonia'), 0, train_size)
    val_pneumonia = copy_images(pneumonia_images, pneumonia_dir, os.path.join(val_dir, 'Viral Pneumonia'), train_size, train_size + val_size)
    test_pneumonia = copy_images(pneumonia_images, pneumonia_dir, os.path.join(test_dir, 'Viral Pneumonia'), train_size + val_size, min_images)
    
    print(f"\nDivisão do dataset:")
    print(f"Treino: {train_normal} imagens normais, {train_pneumonia} imagens de pneumonia")
    print(f"Validação: {val_normal} imagens normais, {val_pneumonia} imagens de pneumonia")
    print(f"Teste: {test_normal} imagens normais, {test_pneumonia} imagens de pneumonia")
    
    return train_dir, val_dir, test_dir

def create_model():
    """
    Cria o modelo CNN
    """
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plota os gráficos de acurácia e perda durante o treinamento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de acurácia
    ax1.plot(history.history['accuracy'], label='Treino')
    ax1.plot(history.history['val_accuracy'], label='Validação')
    ax1.set_title('Acurácia durante o treinamento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    
    # Gráfico de perda
    ax2.plot(history.history['loss'], label='Treino')
    ax2.plot(history.history['val_loss'], label='Validação')
    ax2.set_title('Perda durante o treinamento')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_classification_metrics(y_true, y_pred, class_names, output_path='classification_metrics.png'):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    data = {
        'Categoria': [f'{class_names[0]} (Classe 0)', f'{class_names[1]} (Classe 1)', 'Média Macro', 'Média Ponderada'],
        'Precisão': [precision[0], precision[1], macro[0], weighted[0]],
        'Revocação': [recall[0], recall[1], macro[1], weighted[1]],
        'F1-Score': [f1[0], f1[1], macro[2], weighted[2]],
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Categoria', var_name='Métricas', value_name='Pontuação')

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_melted, x='Categoria', y='Pontuação', hue='Métricas')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3), textcoords='offset points')
    plt.ylim(0, 1.05)
    plt.title('Desempenho do Modelo por Classe e Métrica')
    plt.ylabel('Pontuação')
    plt.xlabel('Categorias')
    plt.legend(title='Métricas')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_score, output_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Baixando Dataset e retorna caminho da base
    base_dir = download_database("tawsifurrahman/covid19-radiography-database")
    
    # Preparar o dataset
    train_dir, val_dir, test_dir = prepare_dataset(base_dir)
    
    if train_dir is None or val_dir is None or test_dir is None:
        print("Erro ao preparar o dataset. Encerrando...")
        return

    try:
        # Geração de dados com normalização e aumento de dados
        train_datagen = ImageDataGenerator(
        rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Geradores de dados
        train_generator = train_datagen.flow_from_directory(
            train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
        print('Mapeamento de classes:', train_generator.class_indices)

        validation_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Cria o modelo
    model = create_model()

        # CALLBACKS
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

    # Treinamento
    history = model.fit(
            train_generator,
        epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # Salvar o modelo final (garantia extra)
        model.save('last_model.keras')
        print("Modelo final salvo como 'last_model.keras'.")

        # Avaliação no conjunto de teste
        test_loss, test_acc = model.evaluate(test_generator)
        print(f'\nAcurácia no conjunto de teste: {test_acc:.4f}')

        # Fazer previsões
        predictions = model.predict(test_generator)
        y_pred = (predictions > 0.5).astype(int)
        y_true = test_generator.classes

        # Calcular métricas
        print('\nRelatório de Classificação:')
        print(classification_report(y_true, y_pred))

        # Plotar matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Previsto')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Plotar gráfico de métricas por classe
        class_names = ['Normal', 'Pneumonia']
        plot_classification_metrics(y_true, y_pred, class_names)

        # Plotar curva ROC
        plot_roc_curve(y_true, predictions)

        print("Treinamento finalizado e modelo salvo como 'best_model.keras'.")

    except Exception as e:
        print(f"Erro durante o treinamento: {str(e)}")

if __name__ == "__main__":
    main()