# given a sentence
# get all 50 feats
# return them in a dictionary
# return the highest level from intrepretability sheet

import pandas as pd

df = pd.read_excel('/Users/noor/Desktop/Thesis/Thesis/git/features/interpretability.xlsx', header=0)

# for each 'Feature' get 'BAREC Level' and save it in a dictionary
feature_barec = {}
for index, row in df.iterrows():
    feature_barec[row['Feature']] = row['BAREC Level']

# remove the keys that are nan
feature_barec = {k: v for k, v in feature_barec.items() if pd.notna(v)}

ranges_unique_words = {
    range(1, 2): 1,
    range(2, 3): 2,
    range(3, 5): 3,
    range(5, 7): 4,
    range(7, 9): 5,
    range(9, 10): 6,
    range(10, 11): 7,
    range(11, 12): 8,
    range(12, 13): 9,
    range(13, 16): 10,
    range(16, 21): 11,
    range(21, 1000): 12
}

ranges_syllables = {
    range(1, 3): 1,
    range(3, 4): 2,
    range(4, 5): 5,
    range(5, 6): 6,
    range(6, 7): 7,
}

# given a row and a level, return the features that have the same level from feature_barec, for only the features in the row

# for feature in row
# if value !=0 or True
# get the level from feature_barec
# get all the features that have the same level as RL_num
# return the features


def get_features(row, level):
    features = []
    if level <12:
        relevant_columns =  [
                            'noun_adj', 'pronoun_proper_noun',  'demonstrative_pronoun_singular', 'preposition', 'demonstrative_pronoun_plural_dual', 'negation_particle',
                            'relative_pronoun_singular', 'relative_pronoun_dual_plural', 'syllables', 'imperfective_singular', 'prc_Al_det', 'suf_1s_pron', 'prc_waw',
                            'verb_present_plural', 'prc_prep', 'suf_pron', 'dual_noun_adj', 'plural_fem_noun_adj', 'verb_past_s_p', 'plural_masc', 'verb_past_present_dual',
                            'verb_command', 'suf_dual_pron', 'broken_plural', 'waw_alqassam', 'verb_command_plural', 'amma_lakin', 'verb_command_dual', 'ba_alqassam',
                            'passive_voice', 'pronoun', 'proper_noun', 'no_obj', 'jar_majroor',  'verbal_present_sentence_with_an_almasdariya',  
                            'verbal_sentence_with_two_objects',  'vocative','kana_wa_akhawataha', 'nominal_sentence', 'inna_wa_akhawataha','exception',  'word count unique', 
                            'samerMSA_1', 'content_level_0',  'content_level_1', 'content_level_2', 'content_level_4', 'content_level_3',
                            ]
    else:
        relevant_columns = [
                            'exception', 'vocab_12.0', 'vocab_14.0', 'vocab_15.0', 'vocab_16.0',
                            'vocab_17.0', 'vocab_18.0',  'content_level_1', 'content_level_2', 'content_level_3', 'content_level_4',
                            'content_level_5', 'content_level_6', 'content_level_7', 'word count unique', 'samerMSA_2', 'samerMSA_2', 'samerMSA_3', 'samerMSA_4', 'samerMSA_5'
                            ]

    for feature in relevant_columns:
        feature_level = 0
        value = row[feature]
        if feature == 'word count unique':           
            # Find the correct value using the mapping
            if value >= 21:
                feature_level = 12
            else:
                # Find the correct value using the mapping
                feature_level = next(v for r, v in ranges_unique_words.items() if value in r)
        elif feature == 'syllables':
            if value >= 6:
                feature_level = 6
            else:
                feature_level = next((v for r, v in ranges_syllables.items() if value in r), None)
        else:
            if value != 0 or value != False:
                feature_level = feature_barec.get(feature)

        if feature_level == level:
            features.append(feature)
    return features