import pandas as pd
import sys
import os

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'train': 'train.csv', 'validation': 'dev.csv', 'test': 'test.csv'}
df_train = pd.read_csv("hf://datasets/CAMeL-Lab/BAREC-Shared-Task-2025-sent/" + splits["train"])


# df_train = pd.read_csv( "../data/mini_data.csv")  # Replace with your actual data file path
# df_train = df_train[df_train['Split'] == 'Train']
sys.path.append(os.path.abspath("../training"))

from disambig_features import extract_disambig_feats
from tree_parsing import parse_data, batch_data
from parser_features import process_conllx_directory
from training.vocab_training import build_vocab_dicts, enrich_with_samer_and_barec, disambiguate_sentence
from vocab import extract_vocab_feats
# from content_inference import extract_content_feats

# 3. Step-by-step feature extraction

# 3.1 Disambiguation features
# disambig_feats = extract_disambig_feats(df_train)

# disambig_feats = pd.read_csv('../data/mini_disambig_feats.csv')
# TO DO add word count unique from df_train 
# disambig_feats['word count unique'] = df_train['Word_Count']

# disambig_feats.to_csv('/l/users/nour.rabih/readability_data/disambig_feats.csv')
# 3.2 data batching and Tree parsing
batch_output_dir = "/l/users/nour.rabih/readability_data/batches_Text_100"
# batch_data(df_train, batch_output_dir)
parsed_output_dir = '/l/users/nour.rabih/readability_data/parsed_Text_100'
# Parse all sentences
# parsed_trees = parse_data(batch_output_dir, parsed_output_dir)

# # 3.3 Parser features (needs parsed trees)
parser_output_path = '/l/users/nour.rabih/readability_data/train_parser_features.csv'
parser_feats = process_conllx_directory(parsed_output_dir, parser_output_path)

# parser_feats = pd.read_csv(parser_output_path)

df_train['disam'] = df_train['Sentence'].apply(lambda x: disambiguate_sentence(x))
# df_train = pd.read_csv('/Users/noor/Desktop/Thesis/Thesis/thesis_data/s31/dev_data_with_disam_pairs.csv')
# 3.4 Vocab training (builds vocab dictionaries from df_train)
vocab_df = build_vocab_dicts(df_train, is_disambiguated=False)
vocab_df.to_csv('vocab.csv')
df_train.to_csv('/l/users/nour.rabih/readability_data/train_with_disam.csv')
# vocab_df = pd.read_csv('vocab.csv')
print('done vocab')
enriched_vocab = enrich_with_samer_and_barec(
    vocab_df,
    samer_path="../data/SAMER-Readability-Lexicon.tsv",
    barec_path="../data/BAREC-Lexicon-updated.csv"
)

enriched_vocab.to_csv('/l/users/nour.rabih/readability_data/enriched.csv')

# enriched_vocab = pd.read_csv('enriched.csv')
# 3.5 Vocab features (uses vocab dictionaries)
from vocab import extract_vocab_feats


# # Make sure df_train includes a 'disam' column with list-of-tuples format
vocab_feats = extract_vocab_feats(df_train, vocab_df, enriched_vocab)
vocab_feats.to_csv('/l/users/nour.rabih/readability_data/train_vocab_feats.csv')
# vocab_feats = pd.read_csv('train_vocab_feats.csv')

# 3.6 Content inference features
# content_feats = extract_content_feats(df_train)
# content_feats = pd.read_csv('train_content_feats.csv')
# df train add content group based on RL_num_19
df_train['content_group'] = df_train['Readability_Level_19'].map({
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    10: 4,
    11: 5,
    12: 5,
    13: 6,
    14: 6,
    15: 7,
    16: 7,
    17: 7,
    18: 7,
    19: 7
})

# # 4. Merge everything into one dataframe on ID
full_features = disambig_feats.merge(parser_feats, on='ID', how='left')
full_features = full_features.merge(vocab_feats, on='ID', how='left')
# add RL_num_19 and content group
full_features = full_features.merge(df_train[['ID', 'content_group', 'Readability_Level_19']], on='ID', how='left') # , 'content_group'
# # full_features = full_features.merge(content_feats, on='ID', how='left')


# # one hot encode 'vocab' and 'samerMSA' columns
full_features = pd.get_dummies(full_features, columns=['vocab', 'samerMSA']) # , 'content_group'

# # # 5. Save the final dataset
full_features.to_csv('/l/users/nour.rabih/readability_data/full_features_train.csv', index=False)

print("✅ Full features dataset created and saved as 'full_features_dataset.csv'")
