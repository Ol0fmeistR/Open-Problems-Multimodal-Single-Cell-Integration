# hyperparameters
LR_START = 0.01
BATCH_SIZE = 512

def create_model():
    reg1 = 9.613e-06
    reg2 = 1e-07
    REG1 = tf.keras.regularizers.l2(reg1)
    REG2 = tf.keras.regularizers.l2(reg2)
    DROP = 0.1

    activation = 'selu'
    inputs = Input(shape=(train_inputs.shape[1],))

    x0 = Dense(256, kernel_regularizer=REG1, activation=activation)(inputs)
    x0 = Dropout(DROP)(x0)
    
    x1 = Dense(512, kernel_regularizer=REG1, activation=activation)(x0)
    x1 = Dropout(DROP)(x1)

    x2 = Dense(512, kernel_regularizer=REG1, activation=activation)(x1) 
    x2= Dropout(DROP)(x2)
    
    x3 = Dense(target.shape[1], kernel_regularizer=REG1, activation=activation)(x2)
    x3 = Dropout(DROP)(x3)
 
    x = Concatenate()([x0, x1, x2, x3])
    x = Dense(target.shape[1], kernel_regularizer=REG2, activation='linear')(x)
    
    model = Model(inputs, x)
    return model
