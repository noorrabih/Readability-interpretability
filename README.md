# Readability-interpretability

This repo consists of code for feature extraction, training the models and inference code for arabic readability and intrepretability.

# Training
contains models for content, vocab, and for the whole task:
1. content: content_training.py
    finetuning arabert to predict the content level.
2. vocab: vocab_training.py
    Builds a vocabulary dictionary and their counts per level

3. classification task: 
    1. bert_with_feats.ipynb
        finetuning arabert on the readability classification task with a features layer
    2. subset_training.ipynb 
        finetuning arabert on the readability classification task, using different subset sizes of the data
    3. models.ipynb
        training conventional models on the task.


# feature extraction
This consists of 4 components:
1. Morphological features: using disambig_features.py
    extracts morphological features from Arabic sentences using CAMeL Tools.
2. Syntactic features: using tree_parsing.py -> parser_features.py
    processes dependency trees to extract syntactic features from Arabic sentences.
3. Vocabulary features: using vocab.py
    gets the vocabulary difficulty level of an Arabic sentence, from the extracted dictionary.
4. content features: content_inference.py
    gets the content difficulty level of an Arabic sentence, from the finetuned model.

# inference
1. feature based mode;
    contains the feature based model.
2. intrepretability
    returns the intrepretable reasons of readability
3. readability models


# full pipeline
the full pipeline consists of feature extraction -> readbility level classification -> intrepretability

these are found in:
1. feature_extraction/full_pipeline.py -> extracts the features of the whole dataset:
        1. Morphological features
        2. data batching and Tree parsing, followed by Syntactic features
        3. Vocab training(add this for the training set only), followed by vocab features 
        4. conent features
2.  readbility level classification
    can be found in the corresponding files for each model:
    1. training/subset_training.ipynb
    2. training/bert_with_feats.ipynb
    3. inference/feature_based_model.py

3. inference/interpretability.py 

Functions from the above files are used in inference/predict_intrepret.py .

So for to get interpretations for a given sentence predict_intrepret.py can be used for guidance.
