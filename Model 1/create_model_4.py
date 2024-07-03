import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, GaussianNoise


### Define parameters of the model
vector_size = 200 #input vector size
gru_count = 64 #GRU Counts
drop = 0.1 #Dropout
rec_drop = 0.2 #Recurrent Dropout
g_n = 0.05 #Gaussian Noise
pca_model_name = "pca_200.pkl" #PCA model
model_name = "model_4"


############ Convert string to vector ###########
def string_to_float32_array(input_str):
    # Remove the square brackets and split the string into a list of number strings
    str_data = input_str.strip('[]').split()
    
    # Convert the list of strings to a NumPy array of type float32
    array_data = np.array(str_data, dtype=np.float32)
    
    return array_data
#################################################

'''
# Example usage
str_data = "[-0.2006275   0.21524101 -0.10322126  0.01488737  0.09281762 -0.205015  ]"
array_data = string_to_float32_array(str_data)

print(array_data)
print(array_data.dtype)
'''


################### Merge All the points ########
def merge_all_points(record):
    vector_1 = string_to_float32_array(record['s_gru'])
    vector_2 = string_to_float32_array(record['j_gru'])
    result = np.concatenate((vector_1, vector_2))
    
    result = np.append(result, record['s_rft'] + record['j_rft'])
    result = np.append(result, record['s_cue'] + record['j_cue'])
    result = np.append(result, record['s_hdg'] + record['j_hdg'])
    result = np.append(result, record['s_imp'] + record['j_imp'])
    result = np.append(result, record['s_pol'])
    result = np.append(result, record['j_pol'])
    result = np.append(result, record['s_mpqa'])
    result = np.append(result, record['j_mpqa'])
    result = np.append(result, record['s_emo'])
    result = np.append(result, record['j_emo'])
    result = np.append(result, record['cosin_sim'])
    result = np.append(result, record['word_occurance'])
    
    return result
#################################################


############### Normalize Data ##################
def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / (std)
    return X_norm
#################################################


############### Apply PCA on train data #########
def apply_pca(data, target_dim = vector_size):
    
    # Initialize PCA with the target number of components
    pca = PCA(n_components = target_dim)
    
    data_matrix = np.stack(data.values)
    # Fit and transform the data series
    transformed_data = pca.fit_transform(data_matrix)
    
    return pca, transformed_data  
#################################################


############### Get target data #################
def get_target(x):
    if x == "mostly-true":
        return 0
    
    elif x == "half-true":
        return 1
    
    else:
        return 2
#################################################


############# Create Neural Network #############
def create_neural_network():
    model = Sequential()
    
    # Add Input Shape
    #model.add(Dense(64, input_shape = (vector_size, 1), activation = 'relu'))
    
    # Add Gaussian Noise
    model.add(GaussianNoise(0.05, input_shape=(200, 1)))
    #model.add(GaussianNoise(0.05))
    
    # Add GRU Layer
    model.add(GRU(gru_count, activation = 'relu'))
    
    # Add dropout and recurrent dropout
    model.add(Dropout(drop))
    model.add(Dropout(rec_drop))
    
    # Add output layer with a single unit and a sigmoid activation function for binary classification
    model.add(Dense(3, activation = 'softmax'))
    
    return model
#################################################


train_dataset_path = '../liar plus dataset/dataset/pre_processed_train.csv'

train_data = pd.read_csv(train_dataset_path)

train_data['new_points'] = train_data.apply(lambda x: merge_all_points(x), axis = 1)



print("All points Merged")

## Build PCA model
pca_model, x_train = apply_pca(train_data['new_points'], target_dim = vector_size)
x_train = normalize_data(x_train)
joblib.dump(pca_model, pca_model_name)


print("All points compressed to required dimensions")

## Get target values
y_train = train_data['label'].apply(lambda x: get_target(str(x).lower()))
a = (y_train == 0).sum()
b = (y_train == 1).sum()
c = (y_train == 2).sum()
n = a + b + c
class_weights = {0: (n/(3*a)), 1: (n/(3*b)), 2: (n/(3*c))}

print("Target Values Generated ")


## Create Model
model = create_neural_network()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print()
model.summary()
print()

# Train Model
model.fit(x_train, y_train, epochs = 5, batch_size = 64, class_weight=class_weights)

# Save Model
model.save(model_name + ".keras")

print("Model Saved")