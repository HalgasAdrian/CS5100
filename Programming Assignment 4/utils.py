import pandas as pd


# Encoding dictionaries for categorical features
SEX_ORDINAL_ENCODING = {
    "Male": 0,
    "Female": 1
}

RACE_ORDINAL_ENCODING = {
    'White': 0,
    'Black': 1,
    'Other': 2
}

OCCUPATION_ORDINAL_ENCODING = {
    'Skilled Labor': 0,
    'Professional': 1,
    'Administrative/Clerical': 2,
    'Service': 3,
    'Sales': 4,
    'Military': 5
}

MARITAL_STATUS_ORDINAL_ENCODING = {
    'Never-married': 0,
    'Married': 1,
    'Divorced': 2,
    'Separated': 3,
    'Widowed': 4
}

EDUCATION_ORDINAL_ENCODING = {
    'Preschool': 0,
    'Primary School': 1,
    'Middle School': 2,
    'High School (Incomplete)': 3,
    'High School Graduate': 4,
    'Some College/Associate Degree': 5,
    "Bachelor's Degree": 6,
    'Graduate School': 7
}

WORKCLASS_ORDINAL_ENCODING = {
    'Private Sector': 0,
    'Government': 1,
    'Self-employed': 2,
    'Unpaid': 3
}

def encode_df_for_ml_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features in the input DataFrame for use in machine learning models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing categorical features to encode.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical features suitable for machine learning models.
    """
    
    df_encoded = df.copy()
    df_encoded['sex'] = df['sex'].map(SEX_ORDINAL_ENCODING)
    df_encoded['race'] = df['race'].map(RACE_ORDINAL_ENCODING)
    df_encoded['occupation'] = df['occupation'].map(OCCUPATION_ORDINAL_ENCODING)
    df_encoded['marital-status'] = df['marital-status'].map(MARITAL_STATUS_ORDINAL_ENCODING)
    df_encoded['education'] = df['education'].map(EDUCATION_ORDINAL_ENCODING)
    df_encoded['workclass'] = df['workclass'].map(WORKCLASS_ORDINAL_ENCODING)

    return df_encoded
