## Open-Problems-Multimodal-Single-Cell-Integration
MLP approach for the open problems multimodal single cell integration Kaggle competition.

### CITEseq

#### :broom: Preprocessing 
1. Seperated constant columns from important columns (features which match the name of a target protein).
2. Dropped constant columns and then applied `TruncatedSVD(n_components=128)` on the remaining columns (minus the important ones).
3. Horizontally stacked the important columns with the SVD columns and then split into train and test sets.
4. Normalized the target variable row wise and then applied `TruncatedSVD(n_components=128)`.

#### :swimming_man: Training and Inference 
1. Cross Validation scheme: GroupKFold on `donor` + `day` combination (9 different combinations in total) using the metadata.
2. Model: Deep Neural Network consisting of 4 hidden layers with `[256, 256, 256, 128]` hidden units + `Dropout(0.1)` at each step.
3. Hyperparameter tuning: Keras Tuner was used to determine the `tf.keras.regularizer.l2` + `num(hidden units)` and `Dropout`.

### Multiome

#### :broom: Preprocessing 
1. Converted multiome data to sparse matrices first in order to reduce file size.
2. Combined train and test set and then applied `TruncatedSVD` with `n_components=128`.
3. Applied `TruncatedSVD(n_components=128)` on the target variable.

#### :swimming_man: Training and Inference 
1. Cross Validation scheme: GroupKFold on `donor` + `day` combination (trained only on 4 folds) using the metadata.
2. Model: Deep Neural Network consisting of 4 hidden layers with `[256, 512, 512, 128]` hidden units + `Dropout(0.1)` at each step.
3. Hyperparameter tuning: Keras Tuner was used to determine the `tf.keras.regularizer.l2` + `num(hidden units)` and `Dropout`.

#### :alarm_clock: Training and Inference time: `1hr 30mins`

#### :computer: Hardware
1. 30 gigs of RAM + Intel Xeon CPU (Kaggle's notebook hardware)

#### :medal_sports: Results
1. 146/1266 (top 11.5%) (Private Score: 0.767757)

#### :horse_racing: Team Details
1. Name: Arindam Baruah
2. Total members: 1
