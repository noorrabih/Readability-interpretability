# works on collab but not my machine.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import Trainer
from datasets import Dataset
import pandas as pd

# full_data = pd.read_csv( "../data/All_data_1M_morph_clean.csv")
# full_data= full_data[full_data['Split'] == 'Test']
# # rename word_sents in full data to text
# full_data.rename(columns={'word_sents': 'text'}, inplace=True)

# Load tokenizer and model
model_name = "Noorrabie/bert_content"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")

# Batch prediction function
def batch_predict(df, model, tokenizer):
    dataset = Dataset.from_pandas(df[['text']])
    dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    trainer = Trainer(model=model, tokenizer=tokenizer)
    outputs = trainer.predict(dataset)
    preds = outputs.predictions.argmax(axis=1)
    return preds


# full_data.to_csv('content_feature.csv')

def extract_content_feats(df):
    # Load the model and tokenizer
    model_name = "Noorrabie/bert_content"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")
    preds= batch_predict(full_data, model, tokenizer)
    df["content_level"] = preds

    return df

def extract_content_from_sentence(sentence):
    # Load the model and tokenizer
    model_name = "Noorrabie/bert_content"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")

    # Prepare the input
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding='max_length', max_length=512)

    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1).item()

    return preds

