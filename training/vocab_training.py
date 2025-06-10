# import pandas as pd
# from collections import defaultdict
# import ast
# from camel_tools.tokenizers.word import simple_word_tokenize
# from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
# from camel_tools.morphology.database import MorphologyDB
# from camel_tools.morphology.analyzer import Analyzer


# df = pd.read_csv("../../thesis_data/1M_features/All_data_1M_morph_clean.csv")
# split = 'Train'
# data = df[df['Split'].isin([split])][:199]

# db = MorphologyDB("../../../CAMeLBERT_morphosyntactic_tagger/calima-msa-s31.db", "a")
# # Initialize the analyzer
# analyzer = Analyzer(db, 'NONE', cache_size=100000)

# # Load the pretrained BERT model
# bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False)
# bert._analyzer = analyzer  

# def disambiguate_sentence(sentence_text):
#     """Extract lemma and POS for each word in a sentence."""
#     sentence = simple_word_tokenize(sentence_text)
#     disambig = bert.tag_sentence(sentence)
#     # return a list of tuples, each tuple contains the lex, pos
#     return [(disambig[i]['lex'], disambig[i]['pos']) for i in range(len(disambig))]


# # if theres a row called disam, make it True
# is_disambiguated = False

# # Dictionary to store occurrences
# lemma_pos_counts = defaultdict(lambda: defaultdict(int))

# # Process each row
# for _, row in data.iterrows():
#     sentence = row['Clean_Sentnece']
#     level = row['RL_num_19']
#     if is_disambiguated:
#         ast.literal_eval(row['disam']) 
#     else:
#         lemma_pos_pairs = disambiguate_sentence(sentence)  

#     for lemma, pos in lemma_pos_pairs:
#         lemma_pos_counts[(lemma, pos)][level] += 1  # Count occurrences per level

# # Convert to DataFrame
# df_counts = pd.DataFrame.from_dict(lemma_pos_counts, orient='index').fillna(0).astype(int)

# # Reset index to convert tuple (lemma, POS) into columns
# # sort the columns names by the number of occurrences
# df_counts.index = pd.MultiIndex.from_tuples(df_counts.index, names=["Lemma", "POS"])
# df_counts.reset_index(inplace=True)

# # get the first non-zero row for each column
# def get_first_non_zero(row):
#     for i in range(1, 19):
#         if row[i] > 0:
#             return i
#     return 0

# df_counts['First_Level'] = df_counts.apply(get_first_non_zero, axis=1)

# # for each word in the df, count all its occurrences in all levels
# # for each word, count the number of levels it appears in
# for i, row in df_counts.iterrows():
#     word = row['Lemma']
#     levels = [row[i] for i in range(1, 19)]
#     df_counts.at[i, 'Total'] = sum(levels)
#     df_counts.at[i, 'Levels'] = sum([1 for level in levels if level > 0])

# # get the first non-zero row for each column, but give it a 1% margin
# def get_first_non_zero_margin(row):
#     margin = row['Total'] * 0.01
#     total = 0
#     for i in range(1, 19):
#         if total > margin or row[i] > margin:
#             return i
#         else:
#             total += row[i]
        
# df_counts['First_Level_Margin_0.01'] = df_counts.apply(get_first_non_zero_margin, axis=1)


# # get the first non-zero row for each column, but give it a 2% margin
# def get_first_non_zero_margin(row):
#     margin = row['Total'] * 0.02
#     total = 0
#     for i in range(1, 19):
#         if total > margin or row[i] > margin:
#             return i
#         else:
#             total += row[i]

# df_counts['First_Level_Margin_0.02'] = df_counts.apply(get_first_non_zero_margin, axis=1)


# df_counts.to_csv("vocab_counts.csv")


# # adding SAMER
# samer = pd.read_csv('/Users/noor/Desktop/Thesis/samer-readability-lexicon/SAMER-Readability-Lexicon.tsv', sep='\t')
# # separate lemma#pos into lemma and pos
# samer[['lemma','pos']] = samer['lemma#pos'].str.split("#",expand=True)

# # vocab = pd.read_csv('/Users/noor/Desktop/Thesis/Thesis/thesis_data/s31/vocab_counts.csv')

# # rename POS to pos
# df_counts.rename(columns={'POS':'pos'}, inplace=True)
# df_counts.rename(columns={'Lemma':'lemma'}, inplace=True)

# # get the number of matching (lemma, POS) in samer and vocab
# samer_vocab = samer.merge(df_counts, on=['lemma','pos'], how='inner')

# # # samer_vocab.to_csv('/Users/noor/Desktop/Thesis/Thesis/thesis_data/s31/samer_vocab.csv', index=False)



# # add BAREC dialectal info 

# barec = pd.read_csv('/Users/noor/Desktop/Thesis/Thesis/thesis_data/s31/BAREC-Lexicon-updated.csv')
# barec.rename(columns={'POS':'pos'}, inplace=True)
# barec.rename(columns={'Lemma':'lemma'}, inplace=True)

# barec_samer_vocab = barec.merge(samer_vocab, on=['lemma', 'pos'], how='inner')

# samer_vocab['samerMSADA'] = samer_vocab['readability (rounded average)']

# samer_vocab.loc[samer_vocab['lemma#pos'].isin(barec_samer_vocab[barec_samer_vocab['readability (rounded average)'] >3]['lemma#pos']), 'samerMSADA'] = samer_vocab['samerMSADA'] - 1

# samer_vocab.to_csv('samerMSA_barec_vocab.csv', index=False)


# vocab_training.py

import pandas as pd
from collections import defaultdict
import ast
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

# Initialize BERT + Analyzer (only once)
db = MorphologyDB("../CAMeLBERT_morphosyntactic_tagger/calima-msa-s31.db", "a")
analyzer = Analyzer(db, 'NONE', cache_size=100000)
bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False)
bert._analyzer = analyzer  

def disambiguate_sentence(sentence_text):
    sentence = simple_word_tokenize(sentence_text)
    disambig = bert.tag_sentence(sentence)
    return [(disambig[i]['lex'], disambig[i]['pos']) for i in range(len(disambig))]

def build_vocab_dicts(df, is_disambiguated=False):
    """
    Builds a vocabulary dictionary with (lemma, POS) keys and their counts per level.
    
    Args:
        df (pd.DataFrame): Input data with 'Clean_Sentence', 'RL_num_19', 'ID'.
        is_disambiguated (bool): If True, uses pre-disambiguated column 'disam'.

    Returns:
        pd.DataFrame: Vocab dictionary dataframe with counts, levels, and first-level thresholds.
    """
    lemma_pos_counts = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        sentence = row['Clean_Sentnece']
        level = row['RL_num_19']
        if is_disambiguated:
            lemma_pos_pairs = ast.literal_eval(row['disam']) if isinstance(row['disam'], str) else row['disam']

            # lemma_pos_pairs = ast.literal_eval(row['disam']) 
        else:
            lemma_pos_pairs = disambiguate_sentence(sentence)  

        for lemma, pos in lemma_pos_pairs:
            lemma_pos_counts[(lemma, pos)][level] += 1

    # Convert to DataFrame
    df_counts = pd.DataFrame.from_dict(lemma_pos_counts, orient='index').fillna(0).astype(int)
    df_counts.index = pd.MultiIndex.from_tuples(df_counts.index, names=["Lemma", "POS"])
    df_counts.reset_index(inplace=True)

    # First non-zero level
    def get_first_non_zero(row):
        for i in range(1, 19):
            # print(row)
            if row[i] > 0:
                return i
        return 0
    df_counts['First_Level'] = df_counts.apply(get_first_non_zero, axis=1)

    # Total and number of levels
    for i, row in df_counts.iterrows():
        levels = [row[i] for i in range(1, 19)]
        df_counts.at[i, 'Total'] = sum(levels)
        df_counts.at[i, 'Levels'] = sum([1 for level in levels if level > 0])

    # First level with 1% margin
    def get_first_non_zero_margin_01(row):
        margin = row['Total'] * 0.01
        total = 0
        for i in range(1, 19):
            if total > margin or row[i] > margin:
                return i
            total += row[i]
    df_counts['First_Level_Margin_0.01'] = df_counts.apply(get_first_non_zero_margin_01, axis=1)

    # First level with 2% margin
    def get_first_non_zero_margin_02(row):
        margin = row['Total'] * 0.02
        total = 0
        for i in range(1, 19):
            if total > margin or row[i] > margin:
                return i
            total += row[i]
    df_counts['First_Level_Margin_0.02'] = df_counts.apply(get_first_non_zero_margin_02, axis=1)

    return df_counts

def enrich_with_samer_and_barec(df_counts, samer_path, barec_path):
    """
    Merges vocab dict with SAMER and BAREC lexical data.

    Args:
        df_counts (pd.DataFrame): Output from build_vocab_dicts
        samer_path (str): Path to SAMER .tsv file
        barec_path (str): Path to BAREC .csv file

    Returns:
        pd.DataFrame: Merged dataframe with readability values.
    """
    # Load SAMER and process
    samer = pd.read_csv(samer_path, sep='\t')
    samer[['lemma','pos']] = samer['lemma#pos'].str.split("#", expand=True)

    df_counts = df_counts.rename(columns={'POS': 'pos', 'Lemma': 'lemma'})
    samer_vocab = samer.merge(df_counts, on=['lemma', 'pos'], how='inner')
    samer_vocab['samerMSADA'] = samer_vocab['readability (rounded average)']

    # Load and merge BAREC
    barec = pd.read_csv(barec_path)
    barec.rename(columns={'POS': 'pos', 'Lemma': 'lemma'}, inplace=True)
    barec_samer_vocab = barec.merge(samer_vocab, on=['lemma', 'pos'], how='inner')

    # Adjust readability if BAREC dialectal and readability > 3
    samer_vocab.loc[
        samer_vocab['lemma#pos'].isin(
            barec_samer_vocab[barec_samer_vocab['readability (rounded average)'] > 3]['lemma#pos']
        ), 'samerMSADA'
    ] -= 1

    return samer_vocab

# Optional CLI usage
if __name__ == "__main__":
    df = pd.read_csv("../../thesis_data/1M_features/All_data_1M_morph_clean.csv")[:200]
    # df = df[df['Split'] == 'Train'][:199]
    # df = df.rename(columns={"Clean_Sentnece": "Clean_Sentence"})

    vocab_df = build_vocab_dicts(df)
    vocab_df.to_csv("vocab_counts.csv", index=False)

    enriched_vocab = enrich_with_samer_and_barec(
        vocab_df,
        samer_path="/Users/noor/Desktop/Thesis/samer-readability-lexicon/SAMER-Readability-Lexicon.tsv",
        barec_path="/Users/noor/Desktop/Thesis/Thesis/thesis_data/s31/BAREC-Lexicon-updated.csv"
    )

    enriched_vocab.to_csv("samerMSA_barec_vocab.csv", index=False)
    print("✅ vocab dict + samer + barec ready.")



