# %% ==================    ==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# This is the file path for the complete banking data csv
bank_file_path = "../data/bank/bank-full.csv"

# This is the Data Set we are going to make the Analysis
bank_data = pd.read_csv(bank_file_path, sep=";")
bank_data.head()

# %% ==================    ==================
bank_data.describe()

# %% ==================    ==================
bank_data.columns

# %% ==================    ==================
# First we need to check what is present in the Data Set we need to work
bank_data.info()

# %% ==================    ==================
def replace_unknown_with_nan(data_frame):
    data_frame_clean = data_frame.copy()

    # Replace 'unknown' with NaN for all columns
    data_frame_clean = data_frame_clean.replace('unknown', np.nan)
    print("\n\n")
    # Print summary of NaN values per column
    nan_summary = data_frame_clean.isna().sum()
    print_message(
        "\nNumber of NaN values per column:", nan_summary
    )
    # Calculate percentage of NaN values
    nan_percentage = (data_frame_clean.isna().sum() /
                      len(data_frame_clean) * 100).round(2)
    print_message(
        "\nPercentage of NaN values per column:", nan_percentage
    )
    return data_frame_clean


def print_message(arg0, arg1):
    print(arg0)
    print(arg1[arg1 > 0])

    print("\n\n")

# %% ==================    ==================
bank_data_clean = replace_unknown_with_nan(bank_data)
bank_data_clean

# %% ==================    ==================
# Summary statistics
print("Summary Statistics for Age:")
print(bank_data_clean['age'].describe())

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(bank_data_clean['age'], kde=True, bins=20, color='skyblue')
plt.title("Age Distribution", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=bank_data_clean['age'], color='lightcoral')
plt.title("Age Boxplot", fontsize=14)
plt.xlabel("Age")
plt.show()

# %% ==================    ==================
# Value counts
print("Occupation Value Counts:")
print(bank_data_clean['job'].value_counts())

# Bar plot
plt.figure(figsize=(10, 6))
sns.countplot(
    y=bank_data_clean['job'], order=bank_data_clean['job'].value_counts().index, palette="pastel")
plt.title("Occupation Distribution", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Job")
plt.show()

# %% ==================    ==================
# Value counts
print("Marital Status Value Counts:")
print(bank_data_clean['marital'].value_counts())

# Bar plot
plt.figure(figsize=(8, 5))
sns.countplot(x=bank_data_clean['marital'], order=bank_data_clean['marital'].value_counts(
).index, palette="pastel")
plt.title("Marital Status Distribution", fontsize=14)
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.show()

# %% ==================    ==================
# Value counts
print("Education Level Value Counts:")
print(bank_data_clean['education'].value_counts())

# Bar plot
plt.figure(figsize=(8, 5))
sns.countplot(x=bank_data_clean['education'],
              order=bank_data_clean['education'].value_counts().index, palette="pastel")
plt.title("Education Level Distribution", fontsize=14)
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.show()

# %% ==================    ==================
# Encoding categorical variables for pairplot
encoder = LabelEncoder()
bank_data_clean['marital_encoded'] = encoder.fit_transform(
    bank_data_clean['marital'])
bank_data_clean['education_encoded'] = encoder.fit_transform(
    bank_data_clean['education'])
bank_data_clean['job_encoded'] = encoder.fit_transform(bank_data_clean['job'])

# Pairplot
sns.pairplot(bank_data_clean, vars=['age'],
             hue='marital', palette="pastel", height=3)
plt.title("Pairplot: Age and Marital Status", y=1.2)
plt.show()

# %% ==================    ==================
# Cross-tabulation for Age grouped by Marital Status and Education
cross_tab = pd.crosstab(
    bank_data_clean['marital'], bank_data_clean['education'])
print("Cross-tabulation of Marital Status and Education Level:")
print(cross_tab)

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Heatmap: Marital Status vs Education Level", fontsize=14)
plt.xlabel("Education Level")
plt.ylabel("Marital Status")
plt.show()

# %% ==================    ==================
# Value counts grouped by 'y'
print("Occupation Value Counts Grouped by Outcome:")
print(bank_data_clean.groupby('y')['job'].value_counts())

# Stacked bar plot for Occupation by Outcome
occupation_counts = pd.crosstab(bank_data_clean['job'], bank_data_clean['y'])
occupation_counts.plot(kind='bar', stacked=True, figsize=(
    12, 6), color=['skyblue', 'orange'])
plt.title("Occupation Distribution Grouped by Outcome", fontsize=14)
plt.xlabel("Occupation")
plt.ylabel("Count")
plt.legend(title="Outcome (y)")
plt.xticks(rotation=45)
plt.show()

# %% ==================    ==================
# Value counts grouped by 'y'
print("Marital Status Value Counts Grouped by Outcome:")
print(bank_data_clean.groupby('y')['marital'].value_counts())

# Stacked bar plot for Marital Status by Outcome
marital_counts = pd.crosstab(bank_data_clean['marital'], bank_data_clean['y'])
marital_counts.plot(kind='bar', stacked=True, figsize=(
    8, 5), color=['lightgreen', 'salmon'])
plt.title("Marital Status Distribution Grouped by Outcome", fontsize=14)
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.legend(title="Outcome (y)")
plt.xticks(rotation=0)
plt.show()

# %% ==================    ==================
# Value counts grouped by 'y'
print("Education Level Value Counts Grouped by Outcome:")
print(bank_data_clean.groupby('y')['education'].value_counts())

# Stacked bar plot for Education Level by Outcome
education_counts = pd.crosstab(
    bank_data_clean['education'], bank_data_clean['y'])
education_counts.plot(kind='bar', stacked=True, figsize=(
    10, 6), color=['cornflowerblue', 'tomato'])
plt.title("Education Level Distribution Grouped by Outcome", fontsize=14)
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.legend(title="Outcome (y)")
plt.xticks(rotation=45)
plt.show()

# %% ==================    ==================
# Summary statistics for Age grouped by 'y'
print("Summary Statistics for Age Grouped by Outcome:")
print(bank_data_clean.groupby('y')['age'].describe())

# Boxplot for Age grouped by Outcome
plt.figure(figsize=(8, 5))
sns.boxplot(x='y', y='age', data=bank_data_clean, palette='Set2')
plt.title("Boxplot of Age Grouped by Outcome", fontsize=14)
plt.xlabel("Outcome (y)")
plt.ylabel("Age")
plt.show()

# Histogram for Age grouped by Outcome
plt.figure(figsize=(10, 6))
sns.histplot(data=bank_data_clean, x='age', hue='y',
             kde=True, bins=20, palette='coolwarm')
plt.title("Age Distribution Grouped by Outcome", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# %% ==================    ==================
# Value counts grouped by 'y'
print("Education Level Value Counts Grouped by Outcome:")
print(bank_data_clean.groupby('y')['age'].value_counts())

# Stacked bar plot for Education Level by Outcome
education_counts = pd.crosstab(
    bank_data_clean['age'], bank_data_clean['y'])
education_counts.plot(kind='bar', stacked=True, figsize=(
    10, 6), color=['cornflowerblue', 'tomato'])
plt.title("Age Distribution Grouped by Outcome", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(title="Outcome (y)")
plt.xticks(rotation=45)
plt.show()

# %% ==================    ==================
# Value counts of Education Level grouped by 'y'
print("Value Counts for Education Level Grouped by Outcome:")
print(bank_data_clean.groupby('y')['education'].value_counts())

# Stacked bar plot for Education Level grouped by Outcome
education_counts = pd.crosstab(
    bank_data_clean['education'], bank_data_clean['y'])
education_counts.plot(kind='bar', stacked=True, figsize=(
    10, 6), color=['skyblue', 'orange'])
plt.title("Education Level Distribution Grouped by Outcome", fontsize=14)
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.legend(title="Outcome (y)")
plt.xticks(rotation=45)
plt.show()

# %% ==================    ==================
# Cross-tabulation of Education Level and Age grouped by Outcome
bank_data_clean['age_group'] = pd.cut(bank_data_clean['age'], bins=[
                                      0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
cross_tab = pd.crosstab([bank_data_clean['education'],
                        bank_data_clean['age_group']], bank_data_clean['y'])

print("Cross-tabulation of Age Groups and Education Level Grouped by Outcome:")
print(cross_tab)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
plt.title("Heatmap: Age Groups and Education Level Grouped by Outcome", fontsize=14)
plt.xlabel("Outcome (y)")
plt.ylabel("Education Level and Age Group")
plt.show()

# %% ==================    ==================
# Stratified sampling (30% of the data)
stratified_sample = bank_data_clean.groupby('y', group_keys=False).apply(
    lambda x: x.sample(frac=0.3, random_state=42))

# Verify the stratified sample distribution
print("Original Data Distribution:")
print(bank_data_clean['y'].value_counts(normalize=True))
print("\nStratified Sample Distribution:")
print(stratified_sample['y'].value_counts(normalize=True))

# Check the size of the sample
print("\nSize of the Stratified Sample:", stratified_sample.shape)

# %% ==================    ==================
# Descriptive statistics for Age
print("\nDescriptive Statistics for Age:")
print(stratified_sample['age'].describe())

# Mean, median, mode, and standard deviation for Age
print("Mean:", stratified_sample['age'].mean())
print("Median:", stratified_sample['age'].median())
print("Mode:", stratified_sample['age'].mode()[0])
print("Standard Deviation:", stratified_sample['age'].std())

# Plot: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(stratified_sample['age'], kde=True, bins=20, color="skyblue")
plt.title("Age Distribution in Stratified Sample", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# %% ==================    ==================
# Descriptive statistics for Occupation
print("\nOccupation Value Counts:")
print(stratified_sample['job'].value_counts())

# Plot: Occupation Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=stratified_sample, y='job',
              order=stratified_sample['job'].value_counts().index, palette="pastel")
plt.title("Occupation Distribution in Stratified Sample", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Occupation")
plt.show()

# %% ==================    ==================
# Descriptive statistics for Marital Status
print("\nMarital Status Value Counts:")
print(stratified_sample['marital'].value_counts())

# Plot: Marital Status Distribution
plt.figure(figsize=(8, 4))
sns.countplot(data=stratified_sample, x='marital',
              order=stratified_sample['marital'].value_counts().index, palette="coolwarm")
plt.title("Marital Status Distribution in Stratified Sample", fontsize=14)
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.show()

# %% ==================    ==================
# Descriptive statistics for Education Level
print("\nEducation Level Value Counts:")
print(stratified_sample['education'].value_counts())

# Plot: Education Level Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=stratified_sample, x='education',
              order=stratified_sample['education'].value_counts().index, palette="pastel")
plt.title("Education Level Distribution in Stratified Sample", fontsize=14)
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# %% ==================    ==================
# Full dataset statistics
full_stats = bank_data_clean[['age']].agg(
    ['mean', 'median', lambda x: x.mode()[0], 'std']).T.rename(columns={0: 'Mode'})
full_stats.columns = ['Full_Mean', 'Full_Median', 'Full_Mode', 'Full_Std']

# Stratified sample statistics
sample_stats = stratified_sample[['age']].agg(
    ['mean', 'median', lambda x: x.mode()[0], 'std']).T.rename(columns={0: 'Mode'})
sample_stats.columns = ['Sample_Mean',
                        'Sample_Median', 'Sample_Mode', 'Sample_Std']

# Combine results
comparison = pd.concat([full_stats, sample_stats], axis=1)
print("Comparison of Metrics (Full Dataset vs Stratified Sample):")
print(comparison)


