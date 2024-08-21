# IMDB Streamlit App

# Presentado por: Luis Eduardo Solarte R. Alejandro Valencia. Diego Alejandro Rodriguez.

## Descripción

Este proyecto es una aplicación web desarrollada en Python utilizando Streamlit que permite cargar, entrenar y evaluar un modelo de aprendizaje profundo utilizando el dataset de reseñas de IMDB. La aplicación incluye visualizaciones de métricas de entrenamiento, matriz de confusión, y la curva ROC, todo desde una interfaz interactiva y fácil de usar.

## Características

- **Carga y preprocesamiento de datos:** El dataset de reseñas de IMDB se carga y se preprocesa automáticamente.
- **Entrenamiento de modelos:** La aplicación permite entrenar un modelo de clasificación de texto utilizando TensorFlow.
- **Visualización de métricas:** Se generan gráficos de la función de pérdida, matriz de confusión, y curva ROC.
- **Interfaz de usuario interactiva:** La aplicación está construida con Streamlit, lo que permite una interacción dinámica y visualización en tiempo real.

## Requisitos

- Python 3.7 o superior
- TensorFlow 2.x
- Streamlit
- TensorFlow Datasets
- Scikit-learn
- Matplotlib

## Descripción

1. 
  Al ejecurar el contenedor localmente se debe abrir el navegador y digitar la siguiente direccion web 'http://localhost:8501/' donde
  se abrira la descripcion del data set.
  ![Pantalla principal de la aplicación](imagenPrincipal.png)
2. Para ejecutar la aplicacion se debe dar click en 'Entrenar Modelo' en el botón al final de la descripcion
  ![Pantalla principal de la aplicación](boton.png)
3. La aplicacion ira mostrando la información de cada epoca de entrenamiento y las metricas resultantes.
  ![Pantalla principal de la aplicación](epoca.png)
4. Al final del entrenamiento el usuario visualizará las graficará la función de costo, la matriz de confusión y la curva Roc resultante 
   del fin del entrenamiento.
   ![Pantalla principal de la aplicación](grafica.png) 

5. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/lukeds14/imdb-streamlit-app.git
   cd imdb-streamlit-app
