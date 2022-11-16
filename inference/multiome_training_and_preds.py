# load necessary libraries
import numpy as np
import pandas as pd
import pickle, os, math
import scipy.sparse
import gc

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, BatchNormalization, Activation


# load train and test sets
train_inputs = scipy.sparse.load_npz("../input/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_values.sparse.npz")
test_inputs = scipy.sparse.load_npz("../input/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_values.sparse.npz")

# load index files for train and test sets
train_idx = np.load("../input/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_idxcol.npz", allow_pickle=True)["index"]
test_idx = np.load("../input/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_idxcol.npz", allow_pickle=True)["index"]

# load target values along with the indexes
target = scipy.sparse.load_npz("../input/multimodal-single-cell-as-sparse-matrix/train_multi_targets_values.sparse.npz")
target_idx = np.load("../input/multimodal-single-cell-as-sparse-matrix/train_multi_targets_idxcol.npz", allow_pickle=True)["index"]

# load metadata and generate combinations
metadata_df = pd.read_csv('../input/open-problems-multimodal/metadata.csv',index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology=="multiome"]

conditions = [
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(13176) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(31800) & metadata_df['day'].eq(7),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(2),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(3),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(4),
    metadata_df['donor'].eq(32606) & metadata_df['day'].eq(7)
    ]

# create a list of the values we want to assign for each condition
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# create a new column and use np.select to assign values to it using our lists as arguments
metadata_df['combination'] = np.select(conditions, values)

# set index as per train and test
meta_train = metadata_df.reindex(train_idx)

# apply truncated SVD to train and test together
inputs = scipy.sparse.vstack([train_inputs, test_inputs])
del train_inputs, test_inputs
_ = gc.collect()
print(f"Input Shape: {str(inputs.shape):14} {inputs.size*4/1024/1024/1024:2.3f} GByte")

def SVD_Save(name, model):
    with open(name, 'wb') as f:
        pickle.dump(model, f)

# dimensions is a hyperparameter
dimensions = 128
path = f'../input/svd-nonbinarized/Multiome_svd_{dimensions}_nonbinarized.pkl'

if os.path.exists(path):
    print("Path exists...")
    with open(path, 'rb') as f:
        trunc_SVD = pickle.load(f)
    print(f"Explained variance: {trunc_SVD.explained_variance_ratio_.sum()}")
    inputs = trunc_SVD.transform(inputs)

else:
    print("Path doesn't exist...")
    trunc_SVD = TruncatedSVD(n_components=dimensions, random_state=42)
    inputs = trunc_SVD.fit_transform(inputs)
    print(f"Explained variance: {trunc_SVD.explained_variance_ratio_.sum()}")
    SVD_Save(path, trunc_SVD)

train_inputs = inputs[:105942].astype(np.float32)
test_inputs = inputs[105942:].astype(np.float32)

del inputs
_ = gc.collect()

# apply truncated SVD on target variable
trunc_SVD_target = TruncatedSVD(n_components=dimensions, random_state=42)
target = trunc_SVD_target.fit_transform(target)
print(f"Explained variance: {trunc_SVD_target.explained_variance_ratio_.sum()}")

# load truth labels for calculation of correlation score
y_values = scipy.sparse.load_npz("../input/multimodal-single-cell-as-sparse-matrix/train_multi_targets_values.sparse.npz")

# training and validation
import warnings
warnings.filterwarnings("ignore")

VERBOSE = 0
N_SPLIT = 4
kf = GroupKFold(n_splits=N_SPLIT)
scores = []

for fold, (idx_tr,idx_va) in enumerate(kf.split(train_inputs, groups=meta_train.combination)):
    print("Splitting data...")
    X_tr, X_va = train_inputs[idx_tr], train_inputs[idx_va]
    y_tr, y_va = target[idx_tr], target[idx_va]
    org_tr, org_va = y_values[idx_tr], y_values[idx_va]
    
    model = create_model()
    
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.9, 
                           patience=4, verbose=VERBOSE)
    
    es = EarlyStopping(monitor="val_loss", patience=30, 
                       verbose=VERBOSE, mode="min", restore_best_weights=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=None)
    
    print(f"Training fold {fold}...")
    model.fit(X_tr, y_tr,
              validation_data=(X_va,y_va),
              epochs=500,
              verbose=VERBOSE,
              batch_size=256,
              callbacks=[es,lr]
             )
    pred = model.predict(X_va)
    
    print(f'\n --------- FOLD {fold} -----------')
    print(f'Mean squared error: {np.round(mean_squared_error(y_va,pred), 2)}')
    print(f'Correlation score: {correlation_score(org_va.todense(),trunc_SVD_target.inverse_transform(pred))}')
    scores.append(correlation_score(org_va.todense(),trunc_SVD_target.inverse_transform(pred)))
   
    filename = f"model_{fold}"
    model.save(filename)
    print('model saved :',filename)
        
    del X_tr,X_va,y_tr,y_va, org_tr, org_va
    _ = gc.collect()
    
# Show overall score
print(f"Average  corr: {np.array(scores).mean():.5f}")

# generate test predictions
preds = np.zeros((test_inputs.shape[0], 23418), dtype='float16')

for fold in range(N_SPLIT):
    print(f'fold {fold} prediction')
    model = tf.keras.models.load_model(f"model_{fold}")
    preds += (model.predict(test_inputs)@trunc_SVD_target.components_)/N_SPLIT
    _ = gc.collect()

# load evaluation ids for submission
eval_ids = pd.read_parquet("../input/multimodal-single-cell-as-sparse-matrix/evaluation.parquet")
eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())

# generate empty submission file
submission = pd.Series(name='target',
                       index=pd.MultiIndex.from_frame(eval_ids), 
                       dtype=np.float32)

# fillup multiome indices with our predictions (in empty submission file)
target_columns = np.load("../input/multimodal-single-cell-as-sparse-matrix/train_multi_targets_idxcol.npz", allow_pickle=True)["columns"]

cell_dict = dict((k,v) for v,k in enumerate(test_idx)) 
assert len(cell_dict)  == len(test_idx)

gene_dict = dict((k,v) for v,k in enumerate(target_columns))
assert len(gene_dict) == len(target_columns)

eval_ids_cell_num = eval_ids.cell_id.apply(lambda x:cell_dict.get(x, -1))
eval_ids_gene_num = eval_ids.gene_id.apply(lambda x:gene_dict.get(x, -1))
valid_multi_rows = (eval_ids_gene_num !=-1) & (eval_ids_cell_num!=-1)

submission.iloc[valid_multi_rows] = preds[eval_ids_cell_num[valid_multi_rows].to_numpy(),
eval_ids_gene_num[valid_multi_rows].to_numpy()]

del eval_ids_cell_num, eval_ids_gene_num, valid_multi_rows, eval_ids, test_idx, target_columns
_ = gc.collect()

# save submission file to be used later with citeseq predictions
submission.reset_index(drop=True, inplace=True)
submission.index.name = 'row_id'
with open("multiome_submission_nonbinarized_v3.pickle", 'wb') as f: pickle.dump(submission, f)
submission
