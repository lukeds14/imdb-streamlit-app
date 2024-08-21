import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from dataset import load_and_preprocess_data
from model import create_model, train_model

# Cargar el conjunto de datos y el modelo
train_dataset, test_dataset, info = load_and_preprocess_data()
model = create_model()

# Título
st.title("IMDB Reviews Dataset")

# Mostrar información del conjunto de datos
st.header("Información sobre el Conjunto de Datos IMDB Reviews")
st.subheader("Información del Dataset:")
st.text(str(info))

# Callback personalizado para mostrar el progreso del entrenamiento
class TrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        st.write(f"Época {epoch + 1}:")
        st.write(f"- Precisión del entrenamiento: {logs['accuracy']:.4f}")
        st.write(f"- Pérdida del entrenamiento: {logs['loss']:.4f}")
        st.write(f"- Precisión de validación: {logs['val_accuracy']:.4f}")
        st.write(f"- Pérdida de validación: {logs['val_loss']:.4f}")

# Función para entrenar el modelo y mostrar las gráficas
def train_and_display():
    try:
        # Entrenar el modelo con el callback
        history = train_model(model, train_dataset, test_dataset, TrainingCallback())

        # Crear la gráfica de pérdida
        fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
        ax_loss.plot(history.history['loss'], label='Pérdida de Entrenamiento')
        ax_loss.plot(history.history['val_loss'], label='Pérdida de Validación')
        ax_loss.set_title('Función de Costo (Pérdida) durante el Entrenamiento')
        ax_loss.set_xlabel('Épocas')
        ax_loss.set_ylabel('Pérdida')
        ax_loss.legend()
        st.pyplot(fig_loss)

        # Calcular predicciones y obtener las etiquetas verdaderas
        y_true = np.concatenate([y for x, y in test_dataset], axis=0)
        y_pred_probs = model.predict(test_dataset)
        y_pred = np.where(y_pred_probs > 0.5, 1, 0)

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

        # Graficar la matriz de confusión
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        ax_cm.matshow(cm, cmap=plt.cm.Blues)
        ax_cm.set_title('Matriz de Confusión')
        ax_cm.set_xlabel('Predicción')
        ax_cm.set_ylabel('Real')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha='center', va='center', color='red')
        st.pyplot(fig_cm)

        # Curva ROC y AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        # Graficar la curva ROC
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title('Curva ROC')
        ax_roc.set_xlabel('Tasa de Falsos Positivos')
        ax_roc.set_ylabel('Tasa de Verdaderos Positivos')
        ax_roc.legend()
        st.pyplot(fig_roc)

        # Mostrar las métricas finales después del entrenamiento
        st.write(f"Entrenamiento completado.")
        st.write(f"Última precisión: {history.history['accuracy'][-1]:.4f}")
        st.write(f"Última pérdida: {history.history['loss'][-1]:.4f}")
        st.write(f"Última precisión en validación: {history.history['val_accuracy'][-1]:.4f}")
        st.write(f"Última pérdida en validación: {history.history['val_loss'][-1]:.4f}")
        st.write(f"Área bajo la curva ROC: {roc_auc:.4f}")
        
    except Exception as e:
        st.error(f"Ocurrió un error durante el entrenamiento: {e}")

# Botón para entrenar el modelo
if st.button("Entrenar Modelo"):
    train_and_display()
