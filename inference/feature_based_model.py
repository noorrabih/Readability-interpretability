import pandas as pd
import numpy as np

def analyze_features_vs_readability(
    data: pd.DataFrame,
    excel_path: str,
    output_path: str,
    relevant_columns: list,
    filter_rl: bool = True,
):
    """    
    Analyzes a dataset of sentence features against gold readability labels (RL_num_19).
    Saves per-sentence predictions and prints accuracy statistics.

    """

    data_sorted = data.sort_values(by='RL_num_19').reset_index(drop=True)

    df = pd.read_excel(excel_path, header=0)
    feature_barec = {
        row['Feature']: row['BAREC Level']
        for _, row in df.iterrows()
        if pd.notna(row['BAREC Level'])
    }

    columns = [col for col in relevant_columns if col not in ['ID', 'Clean_Text', 'pattern_abstract', 'exception_lex']]

    ranges_unique_words = {
        range(1, 2): 1, range(2, 3): 2, range(3, 5): 3, range(5, 7): 4,
        range(7, 9): 5, range(9, 10): 6, range(10, 11): 7, range(11, 12): 8,
        range(12, 13): 9, range(13, 16): 10, range(16, 21): 11, range(21, 1000): 12
    }

    ranges_syllables = {
        range(1, 3): 1, range(3, 4): 2, range(4, 5): 5, range(5, 6): 6, range(6, 7): 7
    }

    sentence_results = {}
    rl_barec_counts = {}

    for index, row in data_sorted.iterrows():
        sentence = row['Clean_Sentnece']
        features = row[columns]
        rl_num = row['RL_num_19']
        ID = row['ID']
        max_barec = 0

        for feature, value in features.items():
            if value != 0:
                if feature == 'word count unique':
                    level = 12 if value >= 21 else next(v for r, v in ranges_unique_words.items() if value in r)
                    barec = level
                elif feature == 'syllables':
                    level = 6 if value >= 6 else next(v for r, v in ranges_syllables.items() if value in r)
                    barec = level
                else:
                    barec = feature_barec.get(feature, 0)

                if barec > max_barec:
                    max_barec = barec

        if rl_num not in rl_barec_counts:
            rl_barec_counts[rl_num] = {}
        rl_barec_counts[rl_num][max_barec] = rl_barec_counts[rl_num].get(max_barec, 0) + 1

        sentence_results[index] = {
            'ID': ID,
            'Text': sentence,
            'RL_num': rl_num,
            'Max BAREC': max_barec,
            'Match': max_barec == rl_num,
            'Higher/lower': 'Higher' if max_barec > rl_num else 'Lower'
        }

    if filter_rl:
        sentence_results = {k: v for k, v in sentence_results.items() if v['RL_num'] <= 11}
        # print(f'Match{rl_cutoff}')
        value = pd.DataFrame(sentence_results).T['Match'].value_counts()
        print(value)
        print(value / len(sentence_results))
    else:
        print('Match')
        print(pd.DataFrame(sentence_results).T['Match'].value_counts())

    sentence_results_df = pd.DataFrame(sentence_results).T
    sentence_results_df = sentence_results_df[['ID', 'Text', 'RL_num', 'Max BAREC']]
    sentence_results_df.rename(columns={'Max BAREC':  'feature based model prediction'}, inplace=True)
    sentence_results_df = sentence_results_df.sort_values(by='ID').reset_index(drop=True)

    sentence_results_df.to_csv(output_path, index=False)
    print(f'Saved to {output_path}')


def predict_readability_level_single(instance: dict, feature_excel_path: str, level: int) -> int:
    """
    Predicts the readability level (BAREC) of a single sentence instance based on its features.

    Args:
        instance (dict): Dictionary containing feature values for the sentence (including 'syllables' and 'word count unique').
        feature_excel_path (str): Path to the Excel file that maps feature names to BAREC levels.
        level (int): Readability level granularity: either 11 or 19.
                     - If 11, will use relevant_cols_11.
                     - If 19, will use relevant_columns_19.

    Returns:
        int: Predicted max BAREC level based on the features.

    Raises:
        ValueError: If level is not 11 or 19.
    """
    # 19 levels
    relevant_columns_19 = ['exception', 'vocab_12.0', 'vocab_14.0', 'vocab_15.0', 'vocab_16.0',
    'vocab_17.0', 'vocab_18.0',  'content_level_1', 'content_level_2', 'content_level_3', 'content_level_4',
    'content_level_5', 'content_level_6', 'content_level_7', 'word count unique', 'samerMSA_2', 'samerMSA_2', 'samerMSA_3', 'samerMSA_4', 'samerMSA_5']


    # 11 levels 
    relevant_cols_11 = ['noun_adj', 'pronoun_proper_noun',  'demonstrative_pronoun_singular', 'preposition', 'demonstrative_pronoun_plural_dual', 'negation_particle',
    'relative_pronoun_singular', 'relative_pronoun_dual_plural', 'syllables', 'imperfective_singular', 'prc_Al_det', 'suf_1s_pron', 'prc_waw',
    'verb_present_plural', 'prc_prep', 'suf_pron', 'dual_noun_adj', 'plural_fem_noun_adj', 'verb_past_s_p', 'plural_masc', 'verb_past_present_dual',
    'verb_command', 'suf_dual_pron', 'broken_plural', 'waw_alqassam', 'verb_command_plural', 'amma_lakin', 'verb_command_dual', 'ba_alqassam',
    'passive_voice', 'pronoun', 'proper_noun', 'no_obj', 'jar_majroor',  'verbal_present_sentence_with_an_almasdariya',  
    'verbal_sentence_with_two_objects',  'vocative','kana_wa_akhawataha', 'nominal_sentence', 'inna_wa_akhawataha','exception',  'word count unique', 
    'samerMSA_1', 'content_level_0',  'content_level_1', 'content_level_2', 'content_level_4', 'content_level_3',
    ]
    if level == 11:
        relevant_columns = relevant_cols_11
    elif level == 19:
        relevant_columns = relevant_columns_19
    else:
        raise ValueError("level must be either 11 or 19")

    columns = [col for col in relevant_columns]

    df = pd.read_excel(feature_excel_path, header=0)
    feature_barec = {
        row['Feature']: row['BAREC Level']
        for _, row in df.iterrows()
        if pd.notna(row['BAREC Level'])
    }

    max_barec = 0

    # Special handling ranges
    ranges_unique_words = {
        range(1, 2): 1, range(2, 3): 2, range(3, 5): 3, range(5, 7): 4,
        range(7, 9): 5, range(9, 10): 6, range(10, 11): 7, range(11, 12): 8,
        range(12, 13): 9, range(13, 16): 10, range(16, 21): 11, range(21, 1000): 12
    }

    ranges_syllables = {
        range(1, 3): 1, range(3, 4): 2, range(4, 5): 5, range(5, 6): 6, range(6, 7): 7
    }

    for feature in columns:
        value = instance.get(feature, 0)
        if value != 0:
            if feature == 'word count unique':
                level_value = 12 if value >= 21 else next(v for r, v in ranges_unique_words.items() if value in r)
                barec = level_value
            elif feature == 'syllables':
                level_value = 6 if value >= 6 else next(v for r, v in ranges_syllables.items() if value in r)
                barec = level_value
            else:
                barec = feature_barec.get(feature, 0)

            if barec > max_barec:
                max_barec = barec

    return max_barec


#     # 19 levels
# relevant_columns_19 = ['exception', 'vocab_12.0', 'vocab_14.0', 'vocab_15.0', 'vocab_16.0',
# 'vocab_17.0', 'vocab_18.0',  'content_level_1', 'content_level_2', 'content_level_3', 'content_level_4',
# 'content_level_5', 'content_level_6', 'content_level_7', 'word count unique', 'samerMSA_2', 'samerMSA_2', 'samerMSA_3', 'samerMSA_4', 'samerMSA_5']


# # 11 levels 
# relevant_cols_11 = ['noun_adj', 'pronoun_proper_noun',  'demonstrative_pronoun_singular', 'preposition', 'demonstrative_pronoun_plural_dual', 'negation_particle',
# 'relative_pronoun_singular', 'relative_pronoun_dual_plural', 'syllables', 'imperfective_singular', 'prc_Al_det', 'suf_1s_pron', 'prc_waw',
# 'verb_present_plural', 'prc_prep', 'suf_pron', 'dual_noun_adj', 'plural_fem_noun_adj', 'verb_past_s_p', 'plural_masc', 'verb_past_present_dual',
# 'verb_command', 'suf_dual_pron', 'broken_plural', 'waw_alqassam', 'verb_command_plural', 'amma_lakin', 'verb_command_dual', 'ba_alqassam',
# 'passive_voice', 'pronoun', 'proper_noun', 'no_obj', 'jar_majroor',  'verbal_present_sentence_with_an_almasdariya',  
# 'verbal_sentence_with_two_objects',  'vocative','kana_wa_akhawataha', 'nominal_sentence', 'inna_wa_akhawataha','exception',  'word count unique', 
# 'samerMSA_1', 'content_level_0',  'content_level_1', 'content_level_2', 'content_level_4', 'content_level_3',
# ]

# 'coordinated_verbs',

# s31
# relevant_columns = [ 'suf_pron', 'demonstrative_pronoun_singular', 'vocative', 'verb_past_s_p', 'kana_wa_akhawataha', 'amma_lakin', 'relative_pronoun_dual_plural',
#  'verb_command', 'plural_masc', 'dual_noun_adj', 'verb_present_plural', 'word count unique', 'relative_pronoun_singular', 'jar_majroor', 'negation_particle', 'verb_past_present_dual',
#  'inna_wa_akhawataha', 'broken_plural', 'syllables', 'verbal_present_sentence_with_an_almasdariya',   'prc_Al_det',
#  'pronoun_verb_present', 'noun_adj', 'pronoun_proper_noun', 'separate_pronoun_singular', 'preposition', 'demonstrative_pronoun_plural_dual', 'imperfective_singular',
#  'suf_1s_pron', 'prc_waw', 'prc_prep', 'plural_fem_noun_adj', 'suf_dual_pron', 'waw_alqassam', 'verb_command_plural', 'verb_command_dual', 'ba_alqassam', 'pronoun', 'proper_noun', 'no_obj',  'verbal_sentence_with_two_objects', 'nominal_sentence', 'coordinated_verbs',
# 'passive_voice', 'samerMSA_1', 'content_level_0',  'content_level_1', 'content_level_2', 'content_level_4', 'content_level_3',

# ]

# 'multiple_verbs', 'exception_lex', 'syllable_word', 'ta_alqassam', 'obj', 'advanced_khabar', 'nominal_with_tpc', 'idafa_lafzia'

# test_data = pd.read_csv('../features/full_features_Dev.csv') # '/Users/noor/Desktop/Thesis/Thesis/thesis_data/s31/dev_data_with_allfeatures_final.csv') #
# test_data_11 = test_data[test_data['RL_num_19'] <= 11]
# test_output_path_11 = 'predictions/feature based model/dev_predictions_11.csv'

# analyze_features_vs_readability(
#     data=test_data_11,
#     excel_path='../features/interpretability.xlsx',
#     output_path=test_output_path_11,
#     relevant_columns=relevant_cols_11,
#     filter_rl=True,
# )


# test_data_19 = test_data[(test_data['RL_num_19'] <= 19) & (test_data['RL_num_19'] > 11)]
# test_output_path_19 = 'predictions/feature based model/dev_predictions_19.csv'

# analyze_features_vs_readability(
#     data=test_data_19,
#     excel_path='../features/interpretability.xlsx',
#     output_path=test_output_path_19,
#     relevant_columns=relevant_columns_19,
#     filter_rl=False,
# )
