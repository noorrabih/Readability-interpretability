"""
Arabic Feature Extraction Script for Readability Analysis

This script extracts morphological features from Modern Standard Arabic sentences
using CAMeL Tools and a pretrained CAMeLBERT model. It processes sentences in batches, disambiguates
them, and identifies both word-level and sentence-level features. The output is saved as a CSV for use in readability prediction or linguistic analysis.

"""


import re
import string
from functools import lru_cache

import pandas as pd
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.charsets import (AR_DIAC_CHARSET, AR_LETTERS_CHARSET,
                                        UNICODE_PUNCT_CHARSET)
from camel_tools.utils.transliterate import Transliterator
from tqdm import tqdm

# Initialize tqdm for pandas
tqdm.pandas()


qassam_lex = ['حَقّ', 'رَبّ' , 'شَمْس', 'قَمَر', 'تِين', 'اللّٰه', 'حَياة','مَوْت', 'دِين', 'قُرْآن', 'كَعْبَة', 'سَماء', 'عَصْر', 'فَجْر', 'لَيْل', 'أَرْض', 'كَعْبَة', 'جَبَل', 'مَلْأَك', 'جَبْرَئِيل', 'ميكائيل', 'مُحَمَّد', 'مُوسَى', 'عِيسَى', 'إِبْرَاهِيم', 'نُوح', 'يُونُس', 'يُوسُف', 'يَعْقُوب', 'إِسْمَاعِيل', 'إِسْحَاق', 'إِدْرِيس', 'نَبِيّ'  ]

tanween = ['ً', 'ٍ', 'ٌ']

# Define vowels
vowels = ['a', 'e', 'i', 'o', 'u', 'aa', 'ee', 'ii', 'oo', 'uu']
arabic_vowels = ['ي', 'و', 'ا', 'ى', 'ؤ', 'ئ', 'إ', 'أ', 'آ', 'ة']
AR_LETTERS_PATTERN = "[" + "".join(AR_LETTERS_CHARSET) + "]"

db = MorphologyDB("../CAMeLBERT_morphosyntactic_tagger/calima-msa-s31.db", "a")

# Initialize the analyzer
analyzer = Analyzer(db, 'NONE', cache_size=100000)

# Load the pretrained BERT model
bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False)

# Set the analyzer
bert._analyzer = analyzer


@lru_cache(maxsize=1024)  # Caches results of disambiguation
def disambiguate_sentence(sentence_text):
    # Tokenize the sentence
    sentence = simple_word_tokenize(sentence_text)
    
    # Tag and disambiguate the sentence
    disambig = bert.tag_sentence(sentence)
    
    return disambig

def extract_word_features(word_analysis):
    """
    Extract morphological features from a single word analysis.
    Args:
        word_analysis (dict): A dictionary containing morphological analysis of a word.
    Returns:
        features (dict): A dictionary containing extracted features for the word.
    """
    # Unpack values from analysis with default values
    pos = word_analysis.get('pos', '')
    num = word_analysis.get('num', '')
    asp = word_analysis.get('asp', '')
    gen = word_analysis.get('gen', '')
    form_gen = word_analysis.get('form_gen', '')
    form_num = word_analysis.get('form_num', '')
    enc0 = word_analysis.get('enc0', '')
    prc0 = word_analysis.get('prc0', '')
    prc1 = word_analysis.get('prc1', '')
    prc2 = word_analysis.get('prc2', '')
    prc3 = word_analysis.get('prc3', '')
    bw = word_analysis.get('bw', '')
    lex = word_analysis.get('lex', '')
    vox = word_analysis.get('vox', '')
    diac = word_analysis.get('diac', '')
    per = word_analysis.get('per', '')

    # مفرد مضارع
    def imperfective_singular():
        return num == 's' and asp == 'i' and pos == 'verb'
    
    # سوابق: ال التعريف
    def prc_Al_det():
        return prc0 == 'Al_det'

    #  حرف  العطف  واو (سابقة)
    def prc_waw():
        return prc2 == 'wa_conj'
    
    # لواحق: ضمير المتكلم المفرد المتصل
    def suf_1s_pron():
        suf = True if (enc0 in ['1s_pron', '1s_poss', '1s_dobj']) else False
        return suf

    #  الفعل المضارع الجمع
    def verb_present_plural():
        return pos == 'verb' and asp == 'i' and num == 'p'

    # سوابق حروف جر متصلة (ب+ ل+ ك+)
    def prc_prep():
        return prc1 in ('bi_prep', 'li_prep', 'ka_prep')

    # لواحق: ضمير  متصل مفرد أو جمع
    def suf_pron():
        # Define the list of pronoun types
        prefixes = [
            "1p", "2ms", "2fs", "2mp", "2fp", "2p", "3ms", "3fs", "3mp", "3fp", "3p"
        ]
        suffixes = [
            "dobj", "poss", "pron"
        ]

        # Combine prefixes and suffixes to form the full list of pronouns
        valid_enc0_values = [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]

        return enc0 in valid_enc0_values

    # المثنى (في الأسماء والصفات)
    def dual_noun_adj():
        return num == 'd' and pos in ['noun', 'adj', 'noun_quant', 'adj_comp']

    # جمع المؤنث السالم في الأسماء فقط والصفات للكلمات الشائعة (سيارات، معلّمات)
    def plural_fem_noun_adj():
        return form_num == 'p' and form_gen == 'f' # and pos in ['noun', 'adj']

    # الفعل الماضي المفرد والجمع
    def verb_past_s_p():
        return pos == 'verb' and asp == 'p' and num in ['s', 'p']

    # جمع مذكر سالم
    def plural_masc():
        return form_gen == 'm' and form_num == 'p' and pos in ['noun', 'adj']

    #  الفعل الماضي المثنى  والمضارع المثنى 
    def verb_past_present_dual():
        return pos == 'verb' and asp in ['p', 'i'] and num == 'd'

    # فعل الأمر المفرد
    def verb_command():
        return pos == 'verb' and asp == 'c' and num == 's'

    #  لواحق: ضمير المثنى المتصل
    def suf_dual_pron():
        prefixes = ["2d", "3d"]
        suffixes = ["dobj", "poss", "pron"]
        valid_enc0_values = [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]
        return enc0 in valid_enc0_values

    # جمع التكسير غير الشائع: أقدام، خضروات
    def broken_plural():
        return pos in ['noun', 'adj'] and form_num == 's' and num == 'p'

    # واو القسم (والله)
    def waw_alqassam():
        return prc2 in [ 'wa_prep'] and lex in qassam_lex

    # فعل الأمر الجمع
    def verb_command_plural():
        return asp == 'c' and num == 'p' and pos == 'verb'

    lex_list = ['ثُمَّ', 'حَتَّى', 'أَوْ', 'أَمْ', 'لٰكِنَّ', 'لٰكِنْ', 'أَمّا'] # لٰكِنْ

    #أدوات ربط
    def amma_lakin():
        return lex in lex_list
    
    # فعل الأمر للمثنى
    def verb_command_dual():
        return asp == 'c' and num == 'd' and pos == 'verb'

    # أداة الاستفهام أ مثل: أسمعتَ.
    # def interrogative_alif():
    #     return prc3 in ['>a_ques', 'A_ques']

    # باء القسم
    def ba_alqassam():
        return prc1 in ['bi_prep'] and lex in qassam_lex 

    # المبني للمجهول
    def passive_voice():
        return vox == 'p' and pos == 'verb'

    # تاء القسم
    def ta_alqassam():
        return prc1 in ['ta_prep'] and lex in qassam_lex

    # ضمير منفصل 
    def pronoun():
        return pos == 'pron' # , 'pron_exclam', 'pron_interrog', 'pron_rel'

    # اسم علم
    def proper_noun():
        return pos == 'noun_prop'

    # Return the extracted features as a dictionary
    return {
        'imperfective_singular': imperfective_singular(),
        'prc_Al_det': prc_Al_det(),
        'suf_1s_pron': suf_1s_pron(),
        'prc_waw': prc_waw(),
        'verb_present_plural': verb_present_plural(),
        'prc_prep': prc_prep(),
        'suf_pron': suf_pron(),
        'dual_noun_adj': dual_noun_adj(),
        'plural_fem_noun_adj': plural_fem_noun_adj(),
        'verb_past_s_p': verb_past_s_p(),
        'plural_masc': plural_masc(),
        'verb_past_present_dual': verb_past_present_dual(),
        'verb_command': verb_command(),
        'suf_dual_pron': suf_dual_pron(),
        'broken_plural': broken_plural(),
        'waw_alqassam': waw_alqassam(),
        'verb_command_plural': verb_command_plural(),
        'amma_lakin' : amma_lakin(),
        'verb_command_dual': verb_command_dual(),
        # 'interrogative_alif': interrogative_alif(),
        'ba_alqassam': ba_alqassam(), 
        'passive_voice': passive_voice(),
        'ta_alqassam' : ta_alqassam(),
        'pronoun': pronoun(),
        'proper_noun': proper_noun()
    }


def pronoun(word_analysis):
    return word_analysis.get('pos') == 'pron'

def imperfective_singular(word_analysis):
    return word_analysis.get('num') == 's' and word_analysis.get('asp') == 'i' and word_analysis.get('pos') == 'verb'

def proper_noun(word_analysis):
    return word_analysis.get('pos') == 'noun_prop'

def count_syllables(caphi, diac, prc0, prc2):
    syllable_count = 0
    if caphi is not None:
        caphi_parts = caphi.split('_')  # Split caphi into individual components
        # Iterate over caphi parts to count syllables
        for i, part in enumerate(caphi_parts):
            if part in vowels:
                if i == len(caphi_parts) - 1:  # If it's the last part (vowel)
                    if diac and diac[-1] in AR_DIAC_CHARSET:  # Check if the last diac char is in AR_DIAC_CHARSET
                        continue  # Don't count this syllable
                syllable_count += 1
            
        if len(caphi_parts) > 2:
            # tanween 
            if caphi_parts[-1] == 'n' and caphi_parts[-2] in vowels and diac[-1] in tanween :
                syllable_count -= 1
        # exclude al
        if prc0 == 'Al_det':
            syllable_count -= 1
        if prc2 == 'wa_conj':
            syllable_count -= 1

    else: # If caphi is None, count syllables based on vowels
        for i in range(len(diac)):
                if diac[i] in arabic_vowels:
                    syllable_count += 1

    return syllable_count

def analyze_sentence(sentence_analysis, raw_sentence):
    """
    Analyze a sentence to extract various linguistic features.
    Args:
        sentence_analysis (list): List of dictionaries containing morphological analysis of each word.
        raw_sentence (str): The original sentence as a string.
    Returns:
        results (dict): A dictionary containing various linguistic features extracted from the sentence.
    """
    syllable_word = ''
    tokenized_raw_sentence = simple_word_tokenize(raw_sentence)
    results = {
        'noun_adj': False,
        'pronoun_proper_noun': False,
        'multiple_verbs': False,
        'exception': False,
        'exception_lex': False,
        'demonstrative_pronoun_singular': False,
        'preposition': False,
        'demonstrative_pronoun_plural_dual': False,
        'negation_particle': False,
        'relative_pronoun_singular': False,
        'relative_pronoun_dual_plural': False,
        'syllables': 0,
        'syllable_word': '',
    }

    verb_count = 0
    lexs = []
    question_mark = False
    interrog_word = False
    tanween = ['ً', 'ٍ', 'ٌ']
    for i, (word, raw_word) in enumerate(zip(sentence_analysis,tokenized_raw_sentence)):
        caphi = word.get('caphi')
        diac = word.get('diac')
        pos = word.get('pos')
        lex = word.get('lex')
        num = word.get('num')
        prc0 = word.get('prc0', '')
        prc2 = word.get('prc2', '')


        # <ضمير> + <فعل مضارع>
        # if i < len(sentence_analysis) - 1 and pronoun(word) and imperfective_singular(sentence_analysis[i + 1]):
        #     results['pronoun_verb_present'] = True

        # <اسم> + <صفة>
        if i < len(sentence_analysis) - 1 and pos == 'noun' and sentence_analysis[i + 1].get('pos') == 'adj':
            results['noun_adj'] = True

        # <ضمير> + <اسم علم>
        if i < len(sentence_analysis) - 1 and pronoun(word) and proper_noun(sentence_analysis[i + 1]):
            results['pronoun_proper_noun'] = True

        # جمل فعلية معطوفة
        # if i == 0 and pos == 'verb':
        #     for word in sentence_analysis[1:]:
        #         if word.get('pos') == 'verb' and word.get('prc2') == 'wa_conj':
        #             results['coordinated_verbs'] = True            

        # فعلين أو أكثر في الجملة الواحدة
        if pos == 'verb' and lex not in lexs : # and diac != 'صَحَّ'
            verb_count += 1
            lexs.append(lex)

        # استثناء
        if pos == 'part_restrict' :
            results['exception'] = True

        if lex in ['إِلّا', 'غَيْر', 'سِوَى', 'عَدا', 'خَلا', 'حاش', 'حاشا']:
            results['exception_lex'] = True

        # ضمير منفصل مفرد
        # if pos == 'pron' and num == 's':
        #     results['separate_pronoun_singular'] = True

        # أسماء الإشارة المفردة
        if pos == 'pron_dem' and num == 's':
            results['demonstrative_pronoun_singular'] = True

        # حروف الجر المنفصلة والمتصلة
        if pos == 'prep' or word.get('prc1') in ['bi_prep', 'li_prep', 'ka_prep', 'wa_prep', 'ta_prep'] :
            results['preposition'] = True

        #  اسم اشارة مثنى، جمع"
        if pos == 'pron_dem' and num in ('p', 'd'):
            results['demonstrative_pronoun_plural_dual'] = True

        # أحرف النفي
        if pos == 'part_neg' : # or lex in ['لا', 'لَيْسَ', 'غَيْر', 'لَمْ', 'لمّا', 'لَنْ', 'ما', 'لات']
            results['negation_particle'] = True

        # أسماء الوصل المفردة
        if pos == 'pron_rel' and num == 's':
            results['relative_pronoun_singular'] = True

        # أسماء الوصل المثنى والجمع
        if pos == 'pron_rel' and num in ['d', 'p']:
            results['relative_pronoun_dual_plural'] = True

        # # الضمائر المنفصلة الجمع
        # if pos == 'pron' and num == 'p':
        #     results['separate_pronoun_plural'] = True

        # Count syllables
        syllable =  count_syllables(caphi, diac, prc0, prc2)
        if syllable > results['syllables']:
            results['syllables'] = syllable
            syllable_word = raw_word


    # Handle post-loop conditions:
    results['multiple_verbs'] = verb_count >= 2
    results['syllable_word'] = syllable_word


    # number of unique words
    # Remove punctuation
    # translator = str.maketrans('', '', punctuation)
    # cleaned_text = raw_sentence.translate(translator)
    # # Split text into words
    # words = cleaned_text.split()
    # # Get unique words
    # unique_words = set(words)
    # # Return the number of unique words
    # results['unique_words'] = len(unique_words)

    # Check for tanween and "Al" at the start of the raw sentence
    # try:
    #     if tokenized_raw_sentence[0][-1] in tanween and sentence_analysis[1].get('prc0') == 'Al_det':
    #         results['tanween_and_al'] = True
    # except:
    #     pass

    return results


# Define feature extraction function
def extract_features_from_row(row):
    """Extract features from a single row of the DataFrame.
    Args:
        row (pd.Series): A single row of the DataFrame containing 'Clean_Sentnece' and 'ID'.
    Returns:
        features (dict): A dictionary containing extracted features for the row.
        This includes both sentence-level and aggregated word-level features.
    """

    sentence = row['Clean_Sentnece']  # Replace 'Text' with the correct column name
    # Perform sentence-level analysis
    if len(sentence.split(" ")) == 1:
        sentence = '"' + sentence + '"'
    sentence_analysis = disambiguate_sentence(sentence)
    sentence_features = analyze_sentence(sentence_analysis, sentence)
    
    # Extract word-level features and aggregate
    word_features_list = []
    for word_analysis in sentence_analysis:
        word_features = extract_word_features(word_analysis)
        word_features_list.append(word_features)
    
    # Aggregate word-level features
    aggregated_word_features = {
        'imperfective_singular': sum(wf.get("imperfective_singular", 0) for wf in word_features_list),
        'prc_Al_det': sum(wf.get("prc_Al_det", 0) for wf in word_features_list),
        'suf_1s_pron': sum(wf.get("suf_1s_pron", 0) for wf in word_features_list),
        'prc_waw': sum(wf.get("prc_waw", 0) for wf in word_features_list),
        'verb_present_plural': sum(wf.get("verb_present_plural", 0) for wf in word_features_list),
        'prc_prep': sum(wf.get("prc_prep", 0) for wf in word_features_list),
        'suf_pron': sum(wf.get("suf_pron", 0) for wf in word_features_list),
        'dual_noun_adj': sum(wf.get("dual_noun_adj", 0) for wf in word_features_list),
        'plural_fem_noun_adj': sum(wf.get("plural_fem_noun_adj", 0) for wf in word_features_list),
        'verb_past_s_p': sum(wf.get("verb_past_s_p", 0) for wf in word_features_list),
        'plural_masc': sum(wf.get("plural_masc", 0) for wf in word_features_list),
        'verb_past_present_dual': sum(wf.get("verb_past_present_dual", 0) for wf in word_features_list),
        'verb_command': sum(wf.get("verb_command", 0) for wf in word_features_list),
        'suf_dual_pron': sum(wf.get("suf_dual_pron", 0) for wf in word_features_list),
        'broken_plural': sum(wf.get("broken_plural", 0) for wf in word_features_list),
        'waw_alqassam': sum(wf.get("waw_alqassam", 0) for wf in word_features_list),
        'verb_command_plural': sum(wf.get("verb_command_plural", 0) for wf in word_features_list),
        'amma_lakin': sum(wf.get("amma_lakin", 0) for wf in word_features_list),
        'verb_command_dual': sum(wf.get("verb_command_dual", 0) for wf in word_features_list),
        # 'interrogative_alif': sum(wf.get("interrogative_alif", 0) for wf in word_features_list),
        'ba_alqassam': sum(wf.get("ba_alqassam", 0) for wf in word_features_list),
        'passive_voice': sum(wf.get("passive_voice", 0) for wf in word_features_list),
        'ta_alqassam': sum(wf.get("ta_alqassam", 0) for wf in word_features_list),
        'pronoun': sum(wf.get("pronoun", 0) for wf in word_features_list),
        'proper_noun': sum(wf.get("proper_noun", 0) for wf in word_features_list)
    }

    # Combine sentence and word-level features and id
    features = {'ID': row['ID'], **sentence_features, **aggregated_word_features}
    
    return features


def extract_features_from_sentence(sentence):
    """Extract features from a single sentence(string).
        Args:
            sentence (str): A single Arabic sentence.
        Returns:
            features (dict): A dictionary containing extracted features.
            pairs (list): A list of tuples containing lemma and pos pairs. (required for further feature extraction)
    """
    # sentence = row['Clean_Sentnece']  # Replace 'Text' with the correct column name
    # Perform sentence-level analysis
    if len(sentence.split(" ")) == 1:
        sentence = '"' + sentence + '"'
    sentence_analysis = disambiguate_sentence(sentence)
    # from each sentence_analysis, extract lemma and pos pairs
    sentence_features = analyze_sentence(sentence_analysis, sentence)
    
    # Extract word-level features and aggregate
    word_features_list = []
    pairs = []

    for word_analysis in sentence_analysis:
        word_features = extract_word_features(word_analysis)
        word_features_list.append(word_features)
        if 'lex' in word_analysis and 'pos' in word_analysis:
            pairs.append((word_analysis['lex'], word_analysis['pos']))
    
    # Aggregate word-level features
    aggregated_word_features = {
        'imperfective_singular': sum(wf.get("imperfective_singular", 0) for wf in word_features_list),
        'prc_Al_det': sum(wf.get("prc_Al_det", 0) for wf in word_features_list),
        'suf_1s_pron': sum(wf.get("suf_1s_pron", 0) for wf in word_features_list),
        'prc_waw': sum(wf.get("prc_waw", 0) for wf in word_features_list),
        'verb_present_plural': sum(wf.get("verb_present_plural", 0) for wf in word_features_list),
        'prc_prep': sum(wf.get("prc_prep", 0) for wf in word_features_list),
        'suf_pron': sum(wf.get("suf_pron", 0) for wf in word_features_list),
        'dual_noun_adj': sum(wf.get("dual_noun_adj", 0) for wf in word_features_list),
        'plural_fem_noun_adj': sum(wf.get("plural_fem_noun_adj", 0) for wf in word_features_list),
        'verb_past_s_p': sum(wf.get("verb_past_s_p", 0) for wf in word_features_list),
        'plural_masc': sum(wf.get("plural_masc", 0) for wf in word_features_list),
        'verb_past_present_dual': sum(wf.get("verb_past_present_dual", 0) for wf in word_features_list),
        'verb_command': sum(wf.get("verb_command", 0) for wf in word_features_list),
        'suf_dual_pron': sum(wf.get("suf_dual_pron", 0) for wf in word_features_list),
        'broken_plural': sum(wf.get("broken_plural", 0) for wf in word_features_list),
        'waw_alqassam': sum(wf.get("waw_alqassam", 0) for wf in word_features_list),
        'verb_command_plural': sum(wf.get("verb_command_plural", 0) for wf in word_features_list),
        'amma_lakin': sum(wf.get("amma_lakin", 0) for wf in word_features_list),
        'verb_command_dual': sum(wf.get("verb_command_dual", 0) for wf in word_features_list),
        # 'interrogative_alif': sum(wf.get("interrogative_alif", 0) for wf in word_features_list),
        'ba_alqassam': sum(wf.get("ba_alqassam", 0) for wf in word_features_list),
        'passive_voice': sum(wf.get("passive_voice", 0) for wf in word_features_list),
        'ta_alqassam': sum(wf.get("ta_alqassam", 0) for wf in word_features_list),
        'pronoun': sum(wf.get("pronoun", 0) for wf in word_features_list),
        'proper_noun': sum(wf.get("proper_noun", 0) for wf in word_features_list)
    }

    # Combine sentence and word-level features and id
    features = { **sentence_features, **aggregated_word_features}
    
    return features, pairs

def load_existing_ids(output_file):
    """Load existing IDs from the output file to avoid duplication."""
    try:
        existing_data = pd.read_csv(output_file, usecols=['ID'])
        existing_ids = set(existing_data['ID'].tolist())
        return existing_ids
    except FileNotFoundError:
        # If the file does not exist, return an empty set
        return set()
 
# Main reusable function
def extract_disambig_feats(df):
    """
    Extract disambiguation features for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with at least 'Clean_Sentnece' and 'ID' columns.
        
    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    features = df.progress_apply(extract_features_from_row, axis=1)
    features_df = pd.DataFrame(features.tolist())
    return features_df

if __name__ == "__main__":
    data_file_path = "../data/All_data_1M_morph_clean.csv"
    batch_size = 200  
    output_file = "../data/disambig_features.csv"

    existing_ids = load_existing_ids(output_file)


    # Process data in batches
    reader = pd.read_csv(data_file_path, chunksize=batch_size)

    # Write header to the output file first
    column_headers =['ID',  'noun_adj', 'pronoun_proper_noun', 'multiple_verbs', 'exception','exception_lex', 'demonstrative_pronoun_singular', 'separate_pronoun_plural', 'preposition', 'demonstrative_pronoun_plural_dual','negation_particle', 'relative_pronoun_singular', 'relative_pronoun_dual_plural', 'syllables', 'syllable_word',      'imperfective_singular', 'prc_Al_det', 'suf_1s_pron', 'prc_waw', 'verb_present_plural', 'prc_prep', 'suf_pron', 'dual_noun_adj', 'plural_fem_noun_adj', 'verb_past_s_p', 'plural_masc', 'verb_past_present_dual', 'verb_command', 'suf_dual_pron', 'broken_plural', 'waw_alqassam', 'verb_command_plural', 'amma_lakin', 'verb_command_dual', 'ba_alqassam', 'passive_voice', 'ta_alqassam', 'pronoun', 'proper_noun']

    # Ensure the output file is created with headers if it doesn't exist
    if not existing_ids:
        pd.DataFrame(columns=column_headers).to_csv(output_file, index=False)

    for i, chunk in enumerate(reader):
        print(f"Processing batch {i+1}...")

        # Skip rows where ID already exists in the output file
        chunk = chunk[~chunk['ID'].isin(existing_ids)]
        if chunk.empty:
            print(f"Batch {i + 1} has no new data to process. Skipping...")
            continue
       # Apply the feature extraction to each row in the chunk
        features = chunk.progress_apply(extract_features_from_row, axis=1)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features.tolist())
        
        # Append the extracted features to the output CSV file
        features_df.to_csv(output_file, mode='a', header=False, index=False)


    print(f"Features extraction complete. Results saved to {output_file}")

# if __name__ == "__main__":
#     input_file = "../../thesis_data/1M_features/All_data_1M_morph_clean.csv"
#     output_file = "disambig_features.csv"

#     df = pd.read_csv(input_file)

#     print("✅ Extracting features...")
#     features_df = extract_disambig_feats(df)

#     features_df.to_csv(output_file, index=False)
#     print(f"✅ Done. Features saved to {output_file}")
