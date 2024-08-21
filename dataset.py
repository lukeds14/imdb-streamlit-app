import tensorflow_datasets as tfds
import tensorflow as tf

def load_and_preprocess_data():
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

    return train_dataset, test_dataset, info
