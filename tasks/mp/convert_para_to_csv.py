import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')  # download the stop words if you haven't already

# Read the CSV file into a pandas DataFrame
train_df = pd.read_csv("mimic/MP_IN_adm_train.csv")
val_df = pd.read_csv("mimic/MP_IN_adm_val.csv")
test_df = pd.read_csv("mimic/MP_IN_adm_test.csv")
# print(train_df.head())
def filter_admission_text(df) -> pd.DataFrame:
    """
    Filter text information by section and preserve other columns except 'text'.
    Returns DataFrame with original columns (except 'text') plus separated text sections.
    """
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()

    admission_sections = {
        "CHIEF_COMPLAINT": "chief complaint:",
        "PRESENT_ILLNESS": "present illness:",
        "MEDICAL_HISTORY": "medical history:",
        "MEDICATION_ADM": "medications on admission:",
        "ALLERGIES": "allergies:",
        "PHYSICAL_EXAM": "physical exam:",
        "FAMILY_HISTORY": "family history:",
        "SOCIAL_HISTORY": "social history:"
    }

    # replace linebreak indicators
    result_df['text'] = result_df['text'].str.replace(r"\n", r"\\n")

    # extract each section by regex
    for key in admission_sections.keys():
        section = admission_sections[key]
        result_df[key] = result_df.text.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'
                                                   .format(section))

        result_df[key] = result_df[key].str.replace(r'\\n', r' ')
        result_df[key] = result_df[key].str.strip()
        result_df[key] = result_df[key].fillna("")
        result_df[result_df[key].str.startswith("[]")][key] = ""

    # filter notes with missing main information
    result_df = result_df[(result_df.CHIEF_COMPLAINT != "") |
                         (result_df.PRESENT_ILLNESS != "") |
                         (result_df.MEDICAL_HISTORY != "")]

    # Drop the original TEXT column
    result_df = result_df.drop(columns=['text'])

    # Verify and clean the text in each section
    text_columns = list(admission_sections.keys())
    for col in text_columns:
        # Convert to string type if not already
        result_df[col] = result_df[col].astype(str)
        # Remove any remaining special characters or extra whitespace
        result_df[col] = result_df[col].apply(lambda x: ' '.join(x.split()))

    # Print information about the resulting DataFrame
    print(f"Original columns: {df.columns.tolist()}")
    print(f"New columns added: {text_columns}")
    print(f"Final columns: {result_df.columns.tolist()}")
    print(f"Number of rows: {len(result_df)}")

    return result_df

# Example usage:
# df = filter_admission_text(your_dataframe)

# To verify the results:
def verify_text_sections(df):
    """
    Print statistics about the text sections to verify the extraction.
    """
    text_columns = ['CHIEF_COMPLAINT', 'PRESENT_ILLNESS', 'MEDICAL_HISTORY',
                    'MEDICATION_ADM', 'ALLERGIES', 'PHYSICAL_EXAM',
                    'FAMILY_HISTORY', 'SOCIAL_HISTORY']

    print("\nText Section Statistics:")
    for col in text_columns:
        non_empty = (df[col].str.len() > 0).sum()
        print(f"{col}:")
        print(f"  Non-empty entries: {non_empty} ({(non_empty/len(df)*100):.2f}%)")
        print(f"  Average length: {df[col].str.len().mean():.2f} characters")
        print(f"  Sample: {df[col].iloc[0][:100]}...")
        print()


# function to clean text data
def clean_text(text):
    # convert text to lowercase
    text = text.lower()
    
    # remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    # remove numbers
    #text = re.sub(r'\d+', '', text)
    
    # remove special characters
    text = re.sub(r'[@#]', '', text)
    
    # remove leading/trailing white spaces
    text = text.strip()
    
    return text

def stopword(text):

    stop_words = set(stopwords.words('english'))  # get the set of English stop words
    stop_words.discard('no')
    words = text.split()  # split the text into words

    # filter out the stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # join the remaining words back into a string
    filtered_text = ' '.join(filtered_words)

    return filtered_text

# You can use it like this:
train_df = filter_admission_text(train_df)
val_df = filter_admission_text(val_df)
test_df = filter_admission_text(test_df)
# verify_text_sections(train_df)

missing_data = train_df.isnull().sum()

# Calculate the percentage of missing values for each column
total_entries = len(train_df)
percent_missing = (missing_data / total_entries) * 100

# Create a new DataFrame to display the results
missing_df = pd.DataFrame({'Missing Count': missing_data, 'Percent Missing': percent_missing})

print(missing_df)

# apply the cleaning function to the 'text' column
train_df['CHIEF_COMPLAINT'] = train_df['CHIEF_COMPLAINT'].apply(clean_text)
train_df['PRESENT_ILLNESS'] = train_df['PRESENT_ILLNESS'].apply(clean_text)
train_df['MEDICAL_HISTORY'] = train_df['MEDICAL_HISTORY'].apply(clean_text)
train_df['MEDICATION_ADM'] = train_df['MEDICATION_ADM'].apply(clean_text)
train_df['ALLERGIES'] = train_df['ALLERGIES'].apply(clean_text)
train_df['PHYSICAL_EXAM'] = train_df['PHYSICAL_EXAM'].apply(clean_text)
train_df['FAMILY_HISTORY'] = train_df['FAMILY_HISTORY'].apply(clean_text)
train_df['SOCIAL_HISTORY'] = train_df['SOCIAL_HISTORY'].fillna('')
train_df['SOCIAL_HISTORY'] = train_df['SOCIAL_HISTORY'].apply(clean_text)

val_df['PRESENT_ILLNESS'] = val_df['PRESENT_ILLNESS'].apply(clean_text)
val_df['MEDICAL_HISTORY'] = val_df['MEDICAL_HISTORY'].apply(clean_text)
val_df['MEDICATION_ADM'] = val_df['MEDICATION_ADM'].apply(clean_text)
val_df['ALLERGIES'] = val_df['ALLERGIES'].apply(clean_text)
val_df['PHYSICAL_EXAM'] = val_df['PHYSICAL_EXAM'].apply(clean_text)
val_df['FAMILY_HISTORY'] = val_df['FAMILY_HISTORY'].apply(clean_text)
val_df['SOCIAL_HISTORY'] = val_df['SOCIAL_HISTORY'].fillna('')
val_df['SOCIAL_HISTORY'] = val_df['SOCIAL_HISTORY'].apply(clean_text)

test_df['PRESENT_ILLNESS'] = test_df['PRESENT_ILLNESS'].apply(clean_text)
test_df['MEDICAL_HISTORY'] = test_df['MEDICAL_HISTORY'].apply(clean_text)
test_df['MEDICATION_ADM'] = test_df['MEDICATION_ADM'].apply(clean_text)
test_df['ALLERGIES'] = test_df['ALLERGIES'].apply(clean_text)
test_df['PHYSICAL_EXAM'] = test_df['PHYSICAL_EXAM'].apply(clean_text)
test_df['FAMILY_HISTORY'] = test_df['FAMILY_HISTORY'].apply(clean_text)
test_df['SOCIAL_HISTORY'] = test_df['SOCIAL_HISTORY'].fillna('')
test_df['SOCIAL_HISTORY'] = test_df['SOCIAL_HISTORY'].apply(clean_text)


print(train_df.head())

# Save dataframe to CSV file
train_df.to_csv("mimic/mimic_col_train.csv", index=False)
val_df.to_csv("mimic/mimic_col_val.csv", index=False)
test_df.to_csv("mimic/mimic_col_test.csv", index=False)

