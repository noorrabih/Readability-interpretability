import pandas as pd
from collections import defaultdict
import ast
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

# df = pd.read_csv( "/Users/noor/Desktop/Thesis/Thesis/thesis_data/s31/dev_data_with_disam_pairs.csv")
# split = 'Dev'
# data = df[df['Split'].isin([split])][:100]

# df_counts = pd.read_csv("../../thesis_data/s31/vocab_counts.csv")

# for each sentence in data, get the max level of its words
# for each word, disambiguate it, get the lex and pos, get the level of the word from df_counts
# get the max level of the sentence
def get_max_level_vocab(pairs, df_counts):
    # lemma_pos_pairs =  ast.literal_eval(pairs) #disambiguate_sentence(sentence)
    lemma_pos_pairs = ast.literal_eval(pairs) if isinstance(pairs, str) else pairs

    levels_margin = []
    levels_margin_2 = []
    levels = []
    for lemma, pos in lemma_pos_pairs:
        # try:
        #     level_margin_2 = df_counts[(df_counts['Lemma'] == lemma) & (df_counts['POS'] == pos)]['First_Level_Margin_0.02'].values[0]
        #     levels_margin_2.append(level_margin_2)
        # except IndexError:
        #     levels_margin_2.append(-1)

        try:
            level_margin = df_counts[(df_counts['Lemma'] == lemma) & (df_counts['POS'] == pos)]['First_Level_Margin_0.01'].values[0]
            levels_margin.append(level_margin)
        except IndexError:
            levels_margin.append(-1)
        # try:
        #     level = df_counts[(df_counts['Lemma'] == lemma) & (df_counts['POS'] == pos)]['First_Level'].values[0]
        #     levels.append(level)
        # except IndexError:
        #     levels.append(-1)

    return { 'levels_margin_0.01': max(levels_margin)} #'levels_margin_0.02' : max(levels_margin_2), 'levels': max(levels)

# for each sentence in data, get the max level of its words
# for each word, disambiguate it, get the lex and pos, get the level of the word from df_counts
# get the max level of the sentence
def get_max_level_samer(pairs, samer_vocab):
    # lemma_pos_pairs =  ast.literal_eval(pairs) #disambiguate_sentence(sentence)
    lemma_pos_pairs = ast.literal_eval(pairs) if isinstance(pairs, str) else pairs

    levels = []
    for lemma, pos in lemma_pos_pairs:
        try:
            level = samer_vocab[(samer_vocab['lemma'] == lemma) & (samer_vocab['pos'] == pos)]['samerMSADA'].values[0]
            levels.append(level)
        except IndexError:
            levels.append(-1)
    return max(levels)


def extract_vocab_feats(data,  vocab_df, samer_vocab, samerMSA = True, vocab = True):
    if samerMSA:
        data['samerMSA'] = data['disam'].apply(lambda x: get_max_level_samer(x, samer_vocab))
        # Apply the function to each row in the DataFrame
    if vocab:
        results = data['disam'].apply(lambda x: get_max_level_vocab(x, vocab_df))
        
        data['vocab'] = results.apply(lambda x: x['levels_margin_0.01'])

    columns_to_return = ['ID']
    if vocab:
        columns_to_return.append('vocab')
    if samerMSA:
        columns_to_return.append('samerMSA')

    return data[columns_to_return]    

def extract_vocab_feats_single(disam, vocab_df, samer_vocab):
    """
    Extract vocabulary readability features for a single sentence.

    Args:
        disam (str or list): Disambiguated (lemma, pos) pairs as a string or list.
        vocab_df (pd.DataFrame): Vocabulary dataframe (BAREC) with levels.
        samer_vocab (pd.DataFrame): SAMER vocabulary dataframe.

    Returns:
        dict: Dictionary with 'vocab' and 'samerMSA' keys.
    """
    import ast

    # Convert string representation to list if needed
    if isinstance(disam, str):
        disam = ast.literal_eval(disam)

    samer_level = get_max_level_samer(disam, samer_vocab)
    vocab_levels = get_max_level_vocab(disam, vocab_df)
    vocab_level = vocab_levels['levels_margin_0.01']

    return {
        'vocab': vocab_level,
        'samerMSA': samer_level
    }