%%time
if TUNE:
    tuner = keras_tuner.BayesianOptimization(
        my_model,
        overwrite=True,
        objective = keras_tuner.Objective("val_negative_correlation_loss", direction="min"),
        max_trials = 20,
        directory='/kaggle/temp',
        seed=1)
    lr = ReduceLROnPlateau(monitor="val_loss", factor = 0.5, 
                           patience = 4, verbose=0)
    es = EarlyStopping(monitor = "val_loss",
                       patience = 12, 
                       verbose= 0,
                       mode = "min", 
                       restore_best_weights = True)
    callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]
    
    X_tr, X_va, y_tr, y_va = train_test_split(X, Y, test_size = 0.2, random_state=10)
    
    tuner.search(X_tr, y_tr,
                 epochs = 1000,
                 validation_data = (X_va, y_va),
                 batch_size = BATCH_SIZE,
                 callbacks = callbacks, verbose=2)
    
    del X_tr, X_va, y_tr, y_va, lr, es, callbacks

if TUNE:
    tuner.results_summary()
    
    # display top 10 trials
    display(pd.DataFrame([hp.values for hp in tuner.get_best_hyperparameters(10)]))
    
    # keep the best hyper params
    best_hp = tuner.get_best_hyperparameters(1)[0]

# set hyperparams manually once you've tuned your network
if not TUNE:
    best_hp = keras_tuner.HyperParameters()
    best_hp.values = {'reg1': 6.89e-6,
                      'reg2': 0,
                      'units1': 256,
                      'units2': 256,
                      'units3': 256,
                      'units4': 128
                     }
