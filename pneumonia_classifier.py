import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import seaborn as sns
import cv2
import pandas as pd

# Configurações
IMG_SIZE = 225
BATCH_SIZE = 32
EPOCHS = 15

def preprocess_image(image):
    """
    Função para pré-processar imagens de prints de tela
    """
    # Converter para escala de cinza se a imagem for colorida
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Garantir que a imagem está no formato correto (uint8) para equalizeHist
    image = image.astype(np.uint8)
    
    # Aplicar equalização de histograma para melhorar contraste
    image = cv2.equalizeHist(image)
    
    # Aplicar filtro gaussiano para reduzir ruído
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Normalizar para [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Redimensionar para o tamanho esperado
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Adicionar canal de cor para compatibilidade com a CNN
    image = np.expand_dims(image, axis=-1)
    
    return image

def create_model():
    """
    Cria o modelo CNN
    """
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
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
    # Carregar e pré-processar os dados
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )
    
    train_generator = train_datagen.flow_from_directory(
        'Dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )
    
    validation_generator = test_datagen.flow_from_directory(
        'Dataset/val',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )
    
    test_generator = test_datagen.flow_from_directory(
        'Dataset/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
    )
    
    # Criar e treinar o modelo
    model = create_model()
    
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Plotar histórico de treinamento
    plot_training_history(history)
    
    # Avaliar o modelo
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
    plot_classification_metrics(y_true, y_pred, class_names, output_path='classification_metrics.png')

    # Plotar curva ROC
    plot_roc_curve(y_true, predictions, output_path='roc_curve.png')

if __name__ == '__main__':
    main()