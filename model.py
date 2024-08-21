import tensorflow as tf

def create_model():
    # Crear el modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=32, mask_zero=True),
        tf.keras.layers.LSTM(24),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_dataset, test_dataset, callback):
    # Entrenar el modelo
    return model.fit(train_dataset, epochs=5, validation_data=test_dataset, callbacks=[callback])
