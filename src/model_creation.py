# Part 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
accuracy_score,
confusion_matrix,
ConfusionMatrixDisplay,
f1_score,
classification_report,
)


# This is the file path for the complete banking data csv
bank_file_path = "../data/bank/bank-full.csv"
def replace_unknown_with_nan(data_frame):
    data_frame_clean = data_frame.copy()

    # Replace 'unknown' with NaN for all columns
    data_frame_clean = data_frame_clean.replace('unknown', np.nan)
    print("\n\n")
    # Print summary of NaN values per column
    nan_summary = data_frame_clean.isna().sum()
    # Calculate percentage of NaN values
    nan_percentage = (data_frame_clean.isna().sum() /
                      len(data_frame_clean) * 100).round(2)
    return data_frame_clean


def print_message(arg0, arg1):
    print(arg0)
    print(arg1[arg1 > 0])

    print("\n\n")

# This is the Data Set we are going to make the Analysis
bank_data = pd.read_csv(bank_file_path, sep=";")
bank_data_clean = replace_unknown_with_nan(bank_data)


# Naïve Bayes classifier
def encode_categorical_columns(data_frame, columns):
    category_mappings = {}
    
    # Encode each specified column
    for col in columns:
        data_frame[col], categories = pd.factorize(data_frame[col], sort=True)
        
        category_mappings[col] = dict(zip(categories, range(len(categories))))
    
    # If you need to read the mapping
    return category_mappings

categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
# Drop all the Null Values 
bank_data_clean.dropna(inplace=True)



encode_categorical_columns(bank_data_clean,categorical_columns)

# Set the X and y
y = bank_data_clean.y
X = bank_data_clean.drop(['y'], axis= 1)

print(bank_data_clean)


def nivebase_bayes_classifier(X_train,X_test,y_train,y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="weighted")
    print("The accuracy of my Naive Bayes Model is:", accuracy)
    print("The F1 Score of my Naive Bayes Model is:", f1)
    
    print("")
    return accuracy, f1

def train_and_test_data(X, y,test_size,train_size,number_of_loops):
    f1_list = []
    accuracy_scores_list = []

    for index in range(number_of_loops):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,train_size=train_size, random_state=index)
        accuracy, f1 = nivebase_bayes_classifier(X_train, X_test, y_train, y_test)
        
        f1_list.append(f1)
        accuracy_scores_list.append(accuracy)
    
    return f1_list,accuracy_scores_list
    
f1_list_70train_30test,accuracy_scores_list_70train_30test = train_and_test_data(X,y,test_size=0.7,train_size=0.3,number_of_loops=10)




# Part 2
# We need to find the average and standard deviation for F1- scores
# For this I have already stored the f1 scores in the f1_score_list

def find_the_f1_score_mean_and_standard_diviation(f1_list):
    plt.figure(figsize=(12, 5))
    # Histogram subplot
    plt.subplot(1, 2, 1)
    plt.hist(f1_list, bins=5, edgecolor='black')
    plt.title('Histogram of Data')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Line plot subplot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(f1_list)), f1_list, marker='o')
    plt.title('Line Plot of Data')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Add horizontal lines for mean and standard deviation of the f1 score
    mean = np.mean(f1_list)
    std_dev = np.std(f1_list)
    plt.axhline(y=mean, color='r', linestyle='--', label=f'Mean ({mean:.4f})')
    plt.axhline(y=mean + std_dev, color='g', linestyle=':', label=f'Mean + Std Dev ({mean+std_dev:.4f})')
    plt.axhline(y=mean - std_dev, color='g', linestyle=':', label=f'Mean - Std Dev ({mean-std_dev:.4f})')
    plt.legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Print statistical information
    print(f"Average: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Minimum Value: {min(f1_list)}")
    print(f"Maximum Value: {max(f1_list)}")
    
find_the_f1_score_mean_and_standard_diviation(f1_list_70train_30test)



# Part 3
f1_list_30train_70test,accuracy_scores_list_30train_70test = train_and_test_data(X,y,test_size=0.3,train_size=0.7,number_of_loops=10)
find_the_f1_score_mean_and_standard_diviation(f1_list_30train_70test)



# Part 4
def train_multiple_nivebase_bayes_classifier(number_of_classifier):
    for index in range(1,number_of_classifier):
        print("\n")
        print(f"This is for {(0.1*index)*100}% train and {(1-0.1*index)*100}% test size data")
        f1_list,accuracy_scores = train_and_test_data(X,y,test_size=1-0.1*index,train_size=0.1*index,number_of_loops=10)
        find_the_f1_score_mean_and_standard_diviation(f1_list)
# We need to test for 8 times
train_multiple_nivebase_bayes_classifier(8)


