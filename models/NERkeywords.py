import nltk 
import pandas as pd
import math

# n = 100 => All NER in full article
n = 100

# installed punkt
#           averaged_perceptron_tagger
#           maxent_ne_chunker
#           words

training_preprocessed = pd.read_csv(f"~/Research/fake_and_real_news/train.csv")
testing_preprocessed = pd.read_csv(f"~/Research/fake_and_real_news/test.csv")
validation_preprocessed = pd.read_csv(f"~/Research/fake_and_real_news/val.csv")

def extract_ner_entities(text):
    tokens = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(tokens)
    named_entities = nltk.ne_chunk(tagged_words)

    ner_words = []
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            ner_words.append(' '.join(c[0] for c in chunk.leaves()))
    total_words = len(tokens)
    num_ner_words = math.ceil(total_words * (n * 0.01))
    
    ner_words = list(set(ner_words))
    if num_ner_words > len(ner_words):
        num_ner_words = len(ner_words)
    return ' '.join(ner_words[:num_ner_words])

training_preprocessed['NER_words'] = training_preprocessed['body_text'].apply(extract_ner_entities)
testing_preprocessed['NER_words'] = testing_preprocessed['body_text'].apply(extract_ner_entities)
validation_preprocessed['NER_words'] = validation_preprocessed['body_text'].apply(extract_ner_entities)

training_preprocessed.head(2)

training_preprocessed.to_csv(f'~/Research/fake_and_real_news/train_NER{str(n)}keywords.csv',index=False)
testing_preprocessed.to_csv(f'~/Research/fake_and_real_news/test_NER{str(n)}keywords.csv',index=False)
validation_preprocessed.to_csv(f'~/Research/fake_and_real_news/val_NER{str(n)}keywords.csv',index=False)
