import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential


### Define parameters of the model
vector_size = 200 #input vector size
gru_count = 100 #GRU Counts
drop = 0.1 #Dropout
rec_drop = 0.2 #Recurrent Dropout
g_n = 0.05 #Gaussian Noise
pca_model_name = "pca_200.pkl" #PCA model
model_name = "model_2"

############ Convert string to vector ###########
def string_to_float32_array(input_str):
    # Remove the square brackets and split the string into a list of number strings
    str_data = input_str.strip('[]').split()
    
    # Convert the list of strings to a NumPy array of type float32
    array_data = np.array(str_data, dtype=np.float32)
    
    return array_data
#################################################


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


################ Transform test data ############
def transform_new_data(pca_model, new_points):
    # Reshape the new data point to 2D array (1 sample, 212 features)
    #new_data_point = new_points.reshape(1, -1)
    new_data_point = np.stack(new_points.values)
    
    # Transform the new data point using the PCA model
    transformed_new_data_point = pca_model.transform(new_data_point)
    
    return transformed_new_data_point
#################################################


############### Normalize Data ##################
def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / (std)
    return X_norm
#################################################



test_dataset_path = '../liar plus dataset/dataset/pre_processed_test.csv'

test_data = pd.read_csv(test_dataset_path)

test_data['new_points'] = test_data.apply(lambda x: merge_all_points(x), axis = 1)

print("All points Merged")


## Get new reduced embeddings
loaded_pca_model = joblib.load(pca_model_name)
x_test = transform_new_data(loaded_pca_model, test_data['new_points'])
x_test = normalize_data(x_test)

print("All points compressed to required dimensions")

print("Target Values Generated")

#Load Model
model = tf.keras.models.load_model(model_name + ".keras")

# Predict with the loaded model
predictions = np.argmax(model.predict(x_test), axis = 1)

print("Prediction Done")

prediction_output = pd.read_csv("output.csv")

prediction_output[model_name] = predictions

prediction_output.to_csv("output.csv", index = False)

print("Result Saved")