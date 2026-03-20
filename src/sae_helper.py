import pandas as pd
import numpy as np

def filter_interpretations(annotations_df, interpretation_fidelity_df, threshold, IDENTITY, M=32):
    to_remove_cols = []
    to_remove_cols += _low_fidelity_interpretations_to_remove(interpretation_fidelity_df, threshold)
    if M == 32:
        interpretation_map = _interpretations_map_M_32()
        to_remove_cols += _prompt_shaped_interpretations_to_remove(annotations_df, IDENTITY, interpretation_map)
    elif M == 16:
        interpretation_map = _interpretations_map_M_16()
        to_remove_cols += _prompt_shaped_interpretations_to_remove(annotations_df, IDENTITY, interpretation_map)
    elif M == 48:
        interpretation_map = _interpretations_map_M_48()
        to_remove_cols += _prompt_shaped_interpretations_to_remove(annotations_df, IDENTITY, interpretation_map)
    elif M == 64:
        interpretation_map = _interpretations_map_M_64()
        to_remove_cols += _prompt_shaped_interpretations_to_remove(annotations_df, IDENTITY, interpretation_map)
    else:
        raise ValueError(f"Unknown M value: {M}. Expected 16 or 32 or 48.")


    print(IDENTITY, to_remove_cols)

    return annotations_df.drop(columns=set(to_remove_cols))


def _low_fidelity_interpretations_to_remove(interpretation_fidelity_df, threshold):
    rows = interpretation_fidelity_df[interpretation_fidelity_df["f1_fidelity_score"] < threshold]
    interpretations_to_remove = rows["interpretation"].tolist()
    print(f"Removing {len(interpretations_to_remove)} low fidelity interpretations...")

    return interpretations_to_remove 

def _prompt_shaped_interpretations_to_remove(annotation_df, IDENTITY, interpretation_map):
    """
    Remove prompt-shaped or low-informative interpretations from the annotation DataFrame.

    Parameters:
    - annotation_df (pd.DataFrame): DataFrame with interpretation columns
    - IDENTITY (str): 'race', 'gender', or 'sexual_orientation'
    - interpretation_map (dict): manual keys to remove
    Returns:
    - pd.DataFrame: DataFrame with specified interpretation columns removed
    """
    if IDENTITY not in interpretation_map:
        raise ValueError(f"Unknown IDENTITY type: {IDENTITY}")

    to_remove = interpretation_map[IDENTITY]

    # Only keep interpretations that exist in the DataFrame
    existing_cols = [col for col in to_remove if col in annotation_df.columns]
    missing_cols = [col for col in to_remove if col not in annotation_df.columns]

    if missing_cols:
        print(f"Warning: The following interpretations were not found in the DataFrame:\n{missing_cols}")
        # raise ValueError("Some interpretations were not found in the DataFrame.")

    print(f"Removing {len(existing_cols)} prompt related interpretations...")
    return existing_cols

##################################################################################################

def _interpretations_map_M_32():
    interpretation_map = {
        "race": [
                "uses the phrase 'I would describe my race' or a close variation of it",
                "explicitly uses the phrase 'identify as' or 'identify with' to describe their race or ethnicity",
                "uses the phrase 'I am' followed by a race or ethnicity descriptor without elaboration",
                "uses single-word or very short descriptions of race/ethnicity (1-3 words)"
            ],
        "gender":  [
                "single-word self-description of gender",
                "uses the phrase 'I would describe my gender identity as ...'"
            ],
        "sexual_orientation": [
                "uses the phrase 'I would describe'",
                "uses a single word or very brief phrase to describe sexual orientation without elaboration"
            ]
    }

    return interpretation_map

def _interpretations_map_M_16(): 
    interpretation_map = {
        "race": [
                "uses single-word or extremely brief responses to describe race/ethnicity",
                "explicitly self-identifies race and ethnicity in distinct terms (e.g., 'My race is... and my ethnicity is...')"
            ],
        "gender":[
                "is a single word response",
                "uses single words or very short phrases to describe gender identity"
            ],
        "sexual_orientation": [
                "responses are extremely brief, often one or two words, and lack elaboration",
                "responses are single words or very short phrases"
            ]
    }

    return interpretation_map

def _interpretations_map_M_48(): 
    interpretation_map = {
        "race": [
                "explicitly states 'my race is' or 'my race believes'",
                "uses single words or very brief phrases to describe race/ethnicity"
            ],
        "gender":[
                "uses a single word to describe gender identity",
                "uses the phrase 'describe myself' or 'describe my gender'",
                "uses single-word or very short phrases to describe gender identity",
                "explicitly states 'identify as' followed by their gender",

            ],
        "sexual_orientation": [
                "uses the phrase 'My sexual orientation is ...'",
                "uses single-word responses or very short phrases",
                "uses the phrase 'I identify as ...'"
            ]
    }

    return interpretation_map

def _interpretations_map_M_64(): 
    interpretation_map = {
        "race": [
                "self-describes using a single concise phrase or label without additional context or elaboration",
                # "uses abstract or vague language to describe race/ethnicity or identity" # i think this is not prompt / syntax related?
            ],
        "gender":[
                "uses concise statements to describe their gender without elaboration or context",
                "single-word responses",
                "single-word responses referring only to gender",
                "explicitly states 'I identify as [a gender]'"
            ],
        "sexual_orientation": [
                "uses the phrase 'I would describe my sexual orientation as ...'",
                "explicitly uses the phrase 'My sexual orientation is ...'",
                "contains single-word responses or extremely brief descriptions",
                "uses the phrase 'identify as' to describe their orientation",
                "uses a brief, single-sentence response to describe sexual orientation",
                # "uses the phrase 'always been' to describe their sexual orientation or attraction",
            ]
    }

    return interpretation_map

##################################################################################################

def calculate_proportions_identity(annotations_df, df, IDENTITY): 
    """
    Calculate the proportion of each interpretation for each identity group.

    Parameters:
        IDENTITY (str): The identity column to group by (e.g. 'race', 'gender', 'sexual_orientation')

    Returns:
        pd.DataFrame: A DataFrame where rows are identity groups and columns are interpretations,
                        containing the proportion of each interpretation within each identity group
    """
    combined_df = pd.concat([annotations_df, df[[IDENTITY]]], axis=1)
    group_counts = combined_df.groupby(IDENTITY).sum()
    # print(group_counts)
    proportions = group_counts.div(group_counts.sum(axis=0), axis=1)
    
    return proportions

def count_activated_themes(annotations_df):
    """
    Count the number of times each theme appears in the annotations DataFrame.
    
    This function counts how many respondents (rows) have each theme (column) activated (value=1).
    For example, if a theme column has 50 1s, it means 50 respondents mentioned that theme.

    Parameters:
        annotations_df (pd.DataFrame): DataFrame containing binary columns (0s and 1s) representing theme activations

    Returns:
        pd.Series: A Series containing the count of activations for each theme, where:
            - Index: theme names (column names from annotations_df)
            - Values: number of times each theme appears (sum of 1s in each column)
    """
    # Validate that all values are binary (0 or 1)
    if not annotations_df.map(lambda x: x in [0, 1]).all().all():
        raise ValueError("annotations_df must contain only binary values (0 or 1)")

    return annotations_df.sum()