from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd


############### Map the Labels ##################
def map_labels(x):
    if str(x['model_1']) == "0":
        if str(x['model_2']) == "0":
            return "true"
        elif str(x['model_2']) == "1":
            if str(x['model_4']) == "0":
                return "mostly-true"
            elif str(x['model_4']) == "1":
                return "half-true"
            else:
                return "false"
        else:
            return "pants-fire"
        
    else:
        if str(x['model_3']) == "0":
            if str(x['model_5']) == "0":
                return "barely-true"
            elif str(x['model_5']) == "1":
                return "false"
            else:
                return "true"
        elif str(x['model_3']) == "1":
            return "pants-fire"
        else:
            return "mostly-true"
        
#################################################


file_path = '../liar plus dataset/dataset/pre_processed_test.csv'
true_labels = pd.read_csv(file_path)

prediction = pd.read_csv("output.csv")

predicted_labels = prediction.apply(lambda x: map_labels(x), axis = 1)


print("Classification Report:\n")
print(classification_report(true_labels['label'], predicted_labels))


print("\n\nConfusion Matrix:\n")
print(confusion_matrix(predicted_labels, true_labels['label']))


print("\nAccuracy Score:")
print(accuracy_score(true_labels['label'], predicted_labels))
