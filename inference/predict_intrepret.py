import sys
import os
sys.path.append(os.path.abspath(".."))

import tempfile
from camel_parser.src.classes import TextParams
from camel_parser.src.conll_output import text_tuples_to_string
from camel_parser.src.data_preparation import get_tagset, parse_text
from camel_parser.src.initialize_disambiguator.disambiguator_interface import get_disambiguator
from camel_tools.utils.charmap import CharMapper
from conllx_df.src.conllx_df import ConllxDf
# from conllx_df.src.conll_utils import get_token_details, add_parent_details, add_direction
from typing import List
from feature_extraction.disambig_features import extract_features_from_sentence


from feature_based_model import predict_readability_level_single
from interpretability import get_features
# from content_inference import extract_content_from_sentence
from feature_extraction.vocab import extract_vocab_feats_single
from feature_extraction.parser_features import extract_parser_features_from_sentence
import pandas as pd
from pathlib import Path
import os

sentence = "الصياد المسكين والمارد اللعين"
label_level = 2
parser_results = extract_parser_features_from_sentence(sentence)

disambig_feats, pairs = extract_features_from_sentence(sentence)
content_level = 0  # my machine doesnt run the line below, so this is just a place holder
# content_level = extract_content_from_sentence(sentence)

vocab_feats = extract_vocab_feats_single(pairs, vocab_df=pd.read_csv('vocab.csv'), samer_vocab=pd.read_csv('enriched.csv'))


# if label_level<12:
#     all_feats = {**disambig_feats,  **parser_results, 'content_level': content_level}  
# else:

all_feats = {**disambig_feats,  **parser_results, **vocab_feats, 'content_level': content_level}
for i in range(8):
    all_feats[f'content_level_{i}'] = all_feats['content_level'] == i

# samerMSA samerMSA_1 - samerMSA_5
for i in range(1, 6):
    all_feats[f'samerMSA_{i}'] = all_feats['samerMSA'] == i

# vocab
for i in range(1, 19):
    all_feats[f'vocab_{i}.0'] = all_feats['vocab'] == float(i)

# TO DO add word count

# get the readability level
# feature based model
level = predict_readability_level_single(all_feats, feature_excel_path='interpretability.xlsx')
print(level)
# bert

# bert with features



# get the corresponding feats
# the level can either be the predicted level or the label level
features = get_features(all_feats, level)
print(features)
