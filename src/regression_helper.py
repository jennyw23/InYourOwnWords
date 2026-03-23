import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm

STANDARDIZED_RACE_FORMULA = "race_indicator_Asian + race_indicator_Some_Other_Race + race_indicator_White + \
race_indicator_Black_or_African_American + race_indicator_Native_Hawaiian_or_Pacific_Islander + \
race_indicator_American_Indian_or_Alaska_Native + race_indicator_Middle_Eastern_or_North_African + \
race_indicator_Hispanic_or_Latino"

STANDARDIZED_GENDER_FORMULA = "C(describe_gender)*C(gender_trans, Treatment('No'))"

STANDARDIZED_SEXUAL_ORIENTATION_FORMULA = "sexual_orientation_indicator_Pansexual + \
sexual_orientation_indicator_Straight_or_heterosexual + sexual_orientation_indicator_Questioning + \
sexual_orientation_indicator_Sexually_fluid + sexual_orientation_indicator_Other_sexual_identity_or_orientation_please_specify + \
sexual_orientation_indicator_Asexual_or_aromantic + sexual_orientation_indicator_I_prefer_not_to_answer + \
sexual_orientation_indicator_Bisexual + sexual_orientation_indicator_Queer + \
sexual_orientation_indicator_Demisexual + sexual_orientation_indicator_Lesbian + sexual_orientation_indicator_Gay"

def create_identity_closed_indicators(df, IDENTITY, col="race_closed"):
    """
    Create binary indicator columns for each identity category found in the comma-separated 
    'race_closed' and 'sexuality_closed responses.
    Each indicator column is named 'race_indicator_<category>'.
    """
    # Get all unique categories from all responses
    categories = set()
    for resp in df[col].dropna():
        for cat in [c.strip() for c in resp.split(",")]:
            categories.add(cat)

    # Create indicator columns
    for cat in categories:
        cleaned_cat = cat.replace(' ', '_').replace('(', '').replace(')', '')
        col_name = f"{IDENTITY}_indicator_{cleaned_cat}"
        df[col_name] = df[col].fillna("").apply(lambda x: int(cat in [c.strip() for c in x.split(",")]))
    return df.copy()

##############################################################################
 
# Define mappings for ordered categorical responses
_ordered_mappings = {
    "health_rating": {
        "I prefer not to answer": np.nan, "Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5
    },
    "importance_scale": {
        "Not at all important": 1, "Not very important": 2, "Somewhat important": 3, "Very important": 4, "Extremely important": 5
    },
    "discrimination_experience": {
        "None at all": 1, "Only a little": 2, "Some": 3, "A lot": 4
    },
    "perceived_similarity": {
        "Mostly the same": 5, "Somewhat the same and somewhat different": 4, "Unsure": 3, "Mostly different": 2, "Completely different": 1, 
    },
    "frequency_experience": {
        "Never": 1, "Less than once a year": 2, "A few times a year": 3, "A few times a month": 4, "At least once a week": 5, "Almost everyday": 6
    },
    "age": {
        "18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65-74": 6, "75 or older": 7, "I prefer not to answer": np.nan
    },
    "education": {
        "Less than high school (Grades 1-8 or no formal schooling)": 1,
        "High school incomplete (Grades 9-11 or Grade 12 with no diploma)": 2,
        "High school graduate (Grade 12 with diploma or GED certificate)": 3,
        "Some college, no degree (includes some community college)": 4,
        "Two-year associate degree from a college or university": 5,
        "Four-year college or university degree/Bachelor's degree": 6,
        "Some postgraduate or professional schooling, no postgraduate degree": 7,
        "Postgraduate or professional degree, including master's, doctorate, medical or law degree": 8,
        "I prefer not to answer": np.nan
    },
    "income": {
        "No income": 1, "Less than $25,000": 2, "$25,000 to $50,000": 3, "$50,000 to $75,000": 4,
        "$75,000 to $100,000": 5, "$100,000 to $200,000": 6, "$200,000 to $500,000": 7, "More than $500,000": 8,
        "I prefer not to answer": np.nan
    },
    "years_in_country": {
        "Less than 5 years": 1, "At least 5 years but less than 10 years": 2,
        "At least 10 years but less than 20 years": 3, "20 years or more": 4,
        "I prefer not to answer": np.nan
    }
}

def _normalize_index_col(df, variable_prefix, mapping, min_val, max_val):
    '''min-max scaled'''
    # print(mapping)
    df['index'] = 0
    for c in df.columns:
        if variable_prefix in c:
            df['index'] += df[c].map(lambda x:mapping[x])

    index_questions = [c for c in df.columns if variable_prefix in c]

    # Normalize the summed index to 0-1
    min_possible_score = len(index_questions) * min_val
    max_possible_score = len(index_questions) * max_val
    
    df['index'] = (
        (df['index'] - min_possible_score) /
        (max_possible_score - min_possible_score)
    )
    return df['index']

def _convert_to_z_scores(df, outcome_columns):
    """
    Convert outcome variables to z-scores (standardized scores).
    For Likert scale data, this function assumes the data can be treated as interval-level.

    Notes:
    ------
    - For Likert scale data, this assumes the data can be treated as interval-level
    - Missing values (NaN) are preserved in the output
    - The function will print warnings if:
        * The data has fewer than 5 unique values (typical Likert scale)
        * The data is highly skewed
    """
    df_z = df.copy()
    
    for col in outcome_columns:
        if col in df.columns:
            # Check if data looks like Likert scale
            unique_vals = df[col].nunique()
            if unique_vals < 5:
                print(f"Warning: {col} has fewer than 5 unique values. This might be a Likert scale.")
            
            # Check for skewness
            skewness = df[col].skew()
            if abs(skewness) > 1:
                print(f"Warning: {col} is highly skewed (skewness = {skewness:.2f}). Z-scores may not be appropriate.")
            
            # Calculate mean and standard deviation, ignoring NaN values
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                print(f"Warning: {col} has zero standard deviation. Cannot compute z-scores.")
                continue
                
            # Convert to z-scores: (x - mean) / std
            df_z[col] = (df[col] - mean) / std
            
    return df_z

def clean_outcome_vars(df, IDENTITY, convert_to_z_scores=False):
    """
    Cleans and normalizes outcome variables in the dataframe.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        identity (str): The identity category (e.g., "race", "gender").

    Returns:
        pd.DataFrame: The cleaned dataframe.
        List[str]: List of outcome variable names.
    """
    # Normalize personal discrimination index
    if "personal_discrimination_index" not in df.columns or not pd.api.types.is_numeric_dtype(df["personal_discrimination_index"]):
        df["personal_discrimination_index"] = _normalize_index_col(
            df,
            variable_prefix="discrim_personal",
            mapping=_ordered_mappings["frequency_experience"],
            min_val=1,
            max_val=6
        )

    # Map importance scale columns
    importance_mappings = [
        ("identity_import_1", _ordered_mappings["importance_scale"], "race_importance"),
        ("identity_import_2", _ordered_mappings["importance_scale"], "gender_importance"),
        ("identity_import_3", _ordered_mappings["importance_scale"], "sexual_orientation_importance"),
    ]
    for col, mapping, new_col in importance_mappings:
        if new_col not in df.columns or not pd.api.types.is_numeric_dtype(df[new_col]):
            df[new_col] = df[col].map(mapping)

    # Map health outcomes
    health_mappings = [
        ("physical_health", _ordered_mappings["health_rating"]),
        ("mental_health", _ordered_mappings["health_rating"]),
    ]
    for col, mapping in health_mappings:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(mapping)

    # Ensure life satisfaction is numeric
    if "life_satisfaction" not in df.columns or not pd.api.types.is_numeric_dtype(df["life_satisfaction"]):
        df["life_satisfaction"] = df["life_satisfaction"].astype(int)

    if "income" not in df.columns or not pd.api.types.is_numeric_dtype(df["income"]):
        df["income"] = df["income"].map(_ordered_mappings["income"])

    # Define outcome variables
    outcomes = [
        "personal_discrimination_index",
        f"{IDENTITY}_importance",
        "physical_health",
        "mental_health",
        "life_satisfaction",
        "income"
    ]

    if convert_to_z_scores:
        df = _convert_to_z_scores(df, outcomes)

    return df, outcomes

def benjamini_hochberg_correction(df, alpha=0.05):
    '''Multiple hypothesis correction, corrects for many DVs with Benjamini–Hochberg procedure '''
    rej, pval_corr = smm.fdrcorrection(df["p-value"], method="indep", alpha=alpha)
    df["BH_corrected_p"] = pval_corr 
    df["BH_reject_null"] = rej

    return df

def bonferroni_correction(df, alpha=0.05):
    """
    Perform Bonferroni correction for multiple comparisons.
    """
    p_values = df["p-value"].tolist()
    num_tests = len(p_values)
    corrected_p_values = [min(p * num_tests, 1.0) for p in p_values]
    reject_null = [p <= alpha for p in corrected_p_values]
    df["Bon_corrected_p"] = corrected_p_values 
    df["Bon_reject_null"] = reject_null   
    return df