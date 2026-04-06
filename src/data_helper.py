import ast 
import numpy as np
import pandas as pd

def get_identity_datasets(path):
    '''Returns race_df (402), gender_df (302), sexual_orientation_df (300), i think'''
    # Call 3 mini datasets
    full_df = pd.read_csv(path)
    full_df["race"] = full_df["race_closed"].apply(lambda x: multiracial_category_hispa_not(x))
    full_df["gender"] = full_df.apply(lambda row: crosstab_gender(row), axis=1)
    full_df["sexual_orientation"] = full_df["sexuality_closed"].apply(group_bi_pan_gay_lesb_queer)

    standardized_cols = ["race", "gender", "sexuality"]
    for identity in standardized_cols:
        embed_col = f"embedded_{identity}_open"
        if embed_col not in full_df.columns:
            continue  # embeddings not pre-computed in this CSV; skip
        if type(full_df[embed_col].iloc[0]) == str:
            full_df[embed_col] = full_df[embed_col].apply(lambda x: ast.literal_eval(x))
            full_df[embed_col] = full_df[embed_col].apply(lambda x: np.array(x))

    race_df = get_race_data(full_df).reset_index()
    gender_df = get_gender_data(full_df).reset_index()
    sexual_orientation_df = get_sexual_orientation_data(full_df).reset_index()
    return race_df, gender_df, sexual_orientation_df

'''Identity helper functions for clustering / other analyses'''

def multiracial_category_hispa_not(identity):
    '''
    If an individual selects one option in the race/ethnicity question, they are categorized as that option.
    • If an individual selects multiple options:
    • If they select "Hispanic" and "White", they are categorized as "Hispanic" [1].
    • If they select a minority label and "White", they are categorized as "Two or More Races".
    • If they select two or more minority labels, they are categorized as "Two or More Races".
    
    '''
    if ',' in identity:
        # Check if the identity contains both 'Hispanic or Latino' and 'White'
        if 'Hispanic or Latino' in identity and 'White' in identity and len(identity.split(','))==2:
            return 'Hispanic or Latino'
        # Otherwise, categorize as "Two or More Races"
        else:
            return f'Two or More Races'
    # If no comma, return the original identity
    else:
        return identity
    
def crosstab_gender_fine(row):
    '''
    Cross tabulate describe_gender and gender_trans
    describe_gender --> Man, Woman, Some other way
    gender_trans --> Yes, No, I prefer not to answer
    '''

    if row['describe_gender'] == "Some other way":
        row['describe_gender'] = "Other"

    if row['gender_trans'] == 'Yes':
        return f"Transgender {row['describe_gender']}"
    elif row['gender_trans'] == 'No':
        return f"Cisgender {row['describe_gender']}"
    elif row['gender_trans'] == 'I prefer not to answer':
        return f"Prefer not to answer {row['describe_gender']}"
    else:
        return "Unspecified"
    
def crosstab_gender(row):
    '''
    Cross tabulate describe_gender and gender_trans
    describe_gender --> Man, Woman, Some other way
    gender_trans --> Yes, No, I prefer not to answer
    '''

    if row['describe_gender'] == "Some other way":
        row['describe_gender'] = "(Other)"

    if row['gender_trans'] == 'Yes':
        return f"Transgender {row['describe_gender']}"
    elif row['gender_trans'] == 'No':
        return f"Cisgender {row['describe_gender']}"
    elif row['gender_trans'] == 'I prefer not to answer':
        return f"Prefer not to answer"
    else:
        return "Unspecified"

def group_bi_pan_gay_lesb_queer(identity):
    """
    Group 'Bisexual' and 'Pansexual' into 'Bisexual and/or Pansexual'
    Group 'Gay' and 'Lesbian' into 'Gay or lesbian'
    Group 'Straight' and Minority group into Minority group
    Group 'Queer' and Minority group into Minority group
    and handle multiple identities.
    """
    if ',' in identity:
        identities = identity.split(",")
        # Group Bisexual and Pansexual
        identities = ['Bisexual and/or Pansexual' if val in ['Bisexual', 'Pansexual'] else val for val in identities]
        # Group Gay and Lesbian
        identities = ['Gay or Lesbian' if val in ['Gay', 'Lesbian'] else val for val in identities]
        
        # remove duplicates
        identities = list(set(identities))

        if 'Queer' in identities and len(identities)==2:
            # print(identities)
            identities.remove('Queer')
            return identities[0] if "Other sexual identity or orientation" not in identity else "Other"

        return 'Multiple Identities' if len(identities) > 1 else ",".join(identities)
    elif identity in ['Bisexual', 'Pansexual']:
        return 'Bisexual and/or Pansexual'
    elif identity in ['Gay', 'Lesbian']:
        return 'Gay or Lesbian'
    return identity if "Other sexual identity or orientation" not in identity else "Other"

def get_race_data(df):
    start_id = "R_5eeheGDAbEjj86L"
    end_id = "R_6orr0seGXPVaOSl"

    start_idx = df.index[df['ResponseId'] == start_id][0]
    end_idx = df.index[df['ResponseId'] == end_id][0]
    df1 = df.iloc[min(start_idx, end_idx):max(start_idx, end_idx) + 1]

    if len(df1) != 302:
        raise ValueError("The size of the dataset should be 302 but is {}".format(len(df1)))
    
    start_id2 = "R_6pGQ8DszBcBHvr6"
    end_id2 = "R_1a9UwEfkraTJWMQ"
    start_idx2 = df.index[df['ResponseId'] == start_id2][0]
    end_idx2 = df.index[df['ResponseId'] == end_id2][0]
    df2 = df.iloc[min(start_idx2, end_idx2):max(start_idx2, end_idx2) + 1]
    filtered_df = pd.concat([df1,df2])
    
    if len(df2) != 100:
        raise ValueError("The size of the dataset should be 100 but is {}".format(len(df1)))
    
    return filtered_df

def get_sexual_orientation_data(df):
    start_id = "R_7ipPe3fvHXscZZs"
    end_id = "R_1wBOum0S4OjQ2tj"
    start_idx = df.index[df['ResponseId'] == start_id][0]
    end_idx = df.index[df['ResponseId'] == end_id][0]
    df1 = df.iloc[min(start_idx, end_idx):max(start_idx, end_idx) + 1]

    start_id2 = "R_5H9p1eabpPzbhBv"
    end_id2 = "R_11b20i52XPh40XO"
    start_idx2 = df.index[df['ResponseId'] == start_id2][0]
    end_idx2 = df.index[df['ResponseId'] == end_id2][0]
    df2 = df.iloc[min(start_idx2, end_idx2):max(start_idx2, end_idx2) + 1]

    filtered_df = pd.concat([df1,df2])
    if len(filtered_df) != 302:
        raise ValueError("The size of the dataset should be 302 but is {}".format(len(filtered_df)))

    return filtered_df

def get_gender_data(df):
    start_id = "R_7CftdMgTpNudzO4"
    end_id = "R_1y3CmieK1ACcwFq"
    start_idx = df.index[df['ResponseId'] == start_id][0]
    end_idx = df.index[df['ResponseId'] == end_id][0]
    df1 = df.iloc[min(start_idx, end_idx):max(start_idx, end_idx) + 1]

    start_id2 = "R_1C613TnnVYcOiwp"
    end_id2 = "R_1pQpi7N87PPxHlU"
    start_idx2 = df.index[df['ResponseId'] == start_id2][0]
    end_idx2 = df.index[df['ResponseId'] == end_id2][0]
    df2 = df.iloc[min(start_idx2, end_idx2):max(start_idx2, end_idx2) + 1]

    filtered_df = pd.concat([df1,df2])
    if len(filtered_df) != 300:
        raise ValueError("The size of the dataset should be 300 but is {}".format(len(filtered_df)))

    return filtered_df
