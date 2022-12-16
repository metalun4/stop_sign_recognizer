from keras import Model, layers, applications


def base_model():
    in_x = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, 5, padding='same', activation='relu')(in_x)
    x = layers.Conv2D(32, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(.5)(x)
    out_x = layers.Dense(4, 'sigmoid')(x)

    return Model(in_x, out_x)


def inc_model():
    inc = applications.InceptionV3(include_top=False, input_tensor=layers.Input(shape=(224, 224, 3)))
    inc.trainable = False

    x = layers.MaxPooling2D(3, 2)(inc.output)
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, 'swish')(x)
    out_x = layers.Dense(4, 'sigmoid')(x)

    return Model(inc.input, out_x)
