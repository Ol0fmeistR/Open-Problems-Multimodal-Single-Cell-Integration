# hyperparameters
LR_START = 0.01
BATCH_SIZE = 256

def my_model(hp, n_inputs = X.shape[1]):
    """Sequential neural network
    
    Returns a compiled instance of tensorflow.keras.models.Model
    """
    activation = 'selu'
    reg1 = hp.Float("reg1", min_value=1e-8, max_value=1e-4, sampling="log")
    reg2 = hp.Float("reg2", min_value=1e-10, max_value=1e-5, sampling="log")
    
    inputs = Input(shape=(n_inputs, ))
    x0 = Dense(hp.Choice('units1', [64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation = activation,
             )(inputs)
    x0 = Dropout(0.1)(x0)
    x1 = Dense(hp.Choice('units2', [64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation = activation,
             )(x0)
    x1 = Dropout(0.1)(x1)
    x2 = Dense(hp.Choice('units3', [32, 64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation = activation,
             )(x1)
    x2 = Dropout(0.1)(x2)
    x3 = Dense(hp.Choice('units4', [32, 64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation = activation,
             )(x2)
    x3 = Dropout(0.1)(x3)
    x = Concatenate()([x0, x1, x2,x3])
    x = Dense(Y.shape[1], kernel_regularizer=tf.keras.regularizers.l2(reg2))(x)
    regressor = Model(inputs, x)
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_START),
                      metrics=[negative_correlation_loss],
                      loss=negative_correlation_loss
                     )
    
    return regressor
