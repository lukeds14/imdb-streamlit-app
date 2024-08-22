import tkinter as tk
from tkinter import scrolledtext, messagebox
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# Cargar el conjunto de datos
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


# Preprocesar los datos
def preprocess_text(text, label):
    text = tf.strings.regex_replace(text, '[^a-zA-Z0-9 ]', '')  # Eliminar caracteres especiales
    return text, label


train_dataset = train_dataset.map(preprocess_text)
test_dataset = test_dataset.map(preprocess_text)

# Tokenización
tokenizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_mode='int', output_sequence_length=200)
tokenizer.adapt(train_dataset.map(lambda x, y: x))

# Convertir las reseñas a secuencias de enteros
train_dataset = train_dataset.map(lambda x, y: (tokenizer(x), y)).batch(32)
test_dataset = test_dataset.map(lambda x, y: (tokenizer(x), y)).batch(32)

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=32, mask_zero=True),
    tf.keras.layers.LSTM(24),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Crear la ventana principal
root = tk.Tk()
root.title("IMDB Reviews Dataset")
root.geometry("800x1600")  # Ventana más alta para acomodar más gráficas

# Crear un contenedor Canvas para permitir el desplazamiento
canvas = tk.Canvas(root)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)

# Crear un frame dentro del canvas con las mismas dimensiones del frame principal
frame = tk.Frame(canvas, width=800, height=1600)
frame.pack_propagate(False)  # Evitar que el frame cambie de tamaño

# Configurar el canvas para ser desplazable
canvas.create_window((0, 0), anchor='nw', window=frame)
canvas.update_idletasks()
canvas.configure(scrollregion=canvas.bbox('all'), yscrollcommand=scroll_y.set)

# Colocar el canvas y la barra de desplazamiento en la ventana principal
canvas.pack(fill='both', expand=True, side='left')
scroll_y.pack(fill='y', side='right')

# Título
title_label = tk.Label(frame, text="Información sobre el Conjunto de Datos IMDB Reviews", font=("Arial", 16))
title_label.pack(pady=10)

# Mostrar información del conjunto de datos
info_label = tk.Label(frame, text="Información del Dataset:", font=("Arial", 12))
info_label.pack(pady=10)

# Crear un área de texto desplazable para mostrar la información
info_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=10)
info_text.pack(pady=10)

# Insertar la información del conjunto de datos
info_text.insert(tk.END, str(info))
info_text.config(state=tk.DISABLED)  # Deshabilitar la edición


# Callback personalizado para mostrar el progreso del entrenamiento
class TrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        metrics_text.config(state=tk.NORMAL)
        metrics_text.insert(tk.END, f"Época {epoch + 1}:\n")
        metrics_text.insert(tk.END, f"- Precisión del entrenamiento: {logs['accuracy']:.4f}\n")
        metrics_text.insert(tk.END, f"- Pérdida del entrenamiento: {logs['loss']:.4f}\n")
        metrics_text.insert(tk.END, f"- Precisión de validación: {logs['val_accuracy']:.4f}\n")
        metrics_text.insert(tk.END, f"- Pérdida de validación: {logs['val_loss']:.4f}\n\n")
        metrics_text.config(state=tk.DISABLED)
        metrics_text.yview(tk.END)  # Desplazar hacia abajo para mostrar la última entrada
        root.update_idletasks()


# Función para entrenar el modelo y mostrar las gráficas
def train_model():
    try:
        metrics_text.config(state=tk.NORMAL)
        metrics_text.delete(1.0, tk.END)  # Limpiar el área de texto
        metrics_text.config(state=tk.DISABLED)

        # Entrenar el modelo con el callback
        history = model.fit(train_dataset, epochs=3, validation_data=test_dataset, callbacks=[TrainingCallback()])

        # Crear la gráfica de pérdida
        fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
        ax_loss.plot(history.history['loss'], label='Pérdida de Entrenamiento')
        ax_loss.plot(history.history['val_loss'], label='Pérdida de Validación')
        ax_loss.set_title('Función de Costo (Pérdida) durante el Entrenamiento')
        ax_loss.set_xlabel('Épocas')
        ax_loss.set_ylabel('Pérdida')
        ax_loss.legend()

        # Mostrar la gráfica de pérdida en Tkinter
        canvas_plot_loss = FigureCanvasTkAgg(fig_loss, master=frame)
        canvas_plot_loss.draw()
        canvas_plot_loss.get_tk_widget().pack(pady=10)

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

        # Mostrar la gráfica de matriz de confusión en Tkinter
        canvas_plot_cm = FigureCanvasTkAgg(fig_cm, master=frame)
        canvas_plot_cm.draw()
        canvas_plot_cm.get_tk_widget().pack(pady=10)

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

        # Ajustar los márgenes para que no se corten los ejes
        fig_roc.tight_layout()

        # Mostrar la gráfica de curva ROC en Tkinter
        canvas_plot_roc = FigureCanvasTkAgg(fig_roc, master=frame)
        canvas_plot_roc.draw()
        canvas_plot_roc.get_tk_widget().pack(pady=10)

        # Actualizar el canvas para que incluya las gráficas en la región desplazable
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'))

        # Mostrar las métricas finales después del entrenamiento
        metrics_text.config(state=tk.NORMAL)
        metrics_text.insert(tk.END, f"Entrenamiento completado.\n")
        metrics_text.insert(tk.END, f"Última precisión: {history.history['accuracy'][-1]:.4f}\n")
        metrics_text.insert(tk.END, f"Última pérdida: {history.history['loss'][-1]:.4f}\n")
        metrics_text.insert(tk.END, f"Última precisión en validación: {history.history['val_accuracy'][-1]:.4f}\n")
        metrics_text.insert(tk.END, f"Última pérdida en validación: {history.history['val_loss'][-1]:.4f}\n")
        metrics_text.insert(tk.END, f"Área bajo la curva ROC: {roc_auc:.4f}\n")
        metrics_text.config(state=tk.DISABLED)
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error durante el entrenamiento: {e}")


# Botón para entrenar el modelo
train_button = tk.Button(frame, text="Entrenar Modelo", command=train_model)
train_button.pack(pady=10)

# Área de texto para mostrar métricas de entrenamiento
metrics_label = tk.Label(frame, text="Métricas de Entrenamiento:", font=("Arial", 12))
metrics_label.pack(pady=10)

metrics_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=10)
metrics_text.pack(pady=10)
metrics_text.config(state=tk.DISABLED)  # Deshabilitar la edición

# Ejecutar la aplicación
root.mainloop()
