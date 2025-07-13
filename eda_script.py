import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"\nCSV file '{file_path}' loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"file '{file_path}' is not found.")
        return None
    except Exception as e:
        print(f"Kuch toh gadbad hai: {e}")
        return None

def get_columns_by_type(file_path):
    df = load_csv(file_path)
    if df is None:
        return None
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return {'categorical': categorical, 'numerical': numerical}

def show_basic_details(df):
    output = "\n=== DataFrame Info ===\n"
    output += "\n1. Columns in DataFrame:\n"
    output += str(df.columns.tolist()) + "\n"
    output += "\n2. DataFrame Shape (Rows, Columns):\n"
    output += str(df.shape) + "\n"
    output += "\n3. Data Types of Columns:\n"
    output += str(df.dtypes) + "\n"
    return output

def show_description(df):
    output = "\n=== Statistical Description (Numerical Columns) ===\n"
    if df.select_dtypes(include=['int64', 'float64']).empty:
        output += "No numerical columns found.\n"
    else:
        output += str(df.describe()) + "\n"
    return output

def show_null_values(df):
    output = "\n=== Null/Missing Values ===\n"
    output += "\n1. Null Values in Each Column:\n"
    output += str(df.isnull().sum()) + "\n"
    output += "\n2. Total Missing Values:\n"
    output += str(df.isnull().sum().sum()) + "\n"
    return output

def plot_histogram(df, column, output_path):
    if column not in df.columns:
        return f"Column '{column}' nahi hai DataFrame mein!\n"
    if df[column].dtype not in ['int64', 'float64']:
        return f"Column '{column}' numerical nahi hai, histogram nahi ban sakta!\n"
    plt.figure(figsize=(8, 6))
    plt.hist(df[column].dropna(), bins=20, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    return f"Histogram saved at {output_path}\n"

def plot_frequency(df, column, output_path):
    if column not in df.columns:
        return f"Column '{column}' is not present is dataframe\n"
    if df[column].dtype not in ['object', 'category']:
        return f"Column '{column}' is not categorical, can't plot frequency plot\n"
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f'Frequency of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    return f"Frequency plot saved at {output_path}\n"

def run_eda(file_path, prompt, output_dir):
    df = load_csv(file_path)
    if df is None:
        return "Failed to load CSV file\n"
    
    prompt = prompt.lower().strip()
    if prompt == 'show basic details':
        return show_basic_details(df)
    elif prompt == 'show description':
        return show_description(df)
    elif prompt == 'show null values':
        return show_null_values(df)
    elif prompt.startswith('histogram of '):
        column = prompt.replace('histogram of ', '').strip()
        output_path = os.path.join(output_dir, f"histogram_{column}.png")
        return plot_histogram(df, column, output_path)
    elif prompt.startswith('frequency of '):
        column = prompt.replace('frequency of ', '').strip()
        output_path = os.path.join(output_dir, f"frequency_{column}.png")
        return plot_frequency(df, column, output_path)
    else:
        return "Invalid prompt. Use 'show basic details', 'show description', 'show null values', 'histogram of column_name', or 'frequency of column_name'\n"