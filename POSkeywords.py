import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import math

# try 10, 20, and 30% for both
n = 25  

training   = pd.read_csv("~/Desktop/4thSem/DATALab/fake_and_real_news/train.csv").reset_index(drop=True)
testing    = pd.read_csv('~/Desktop/4thSem/DATALab/fake_and_real_news/test.csv').reset_index(drop=True)
validation = pd.read_csv('~/Desktop/4thSem/DATALab/fake_and_real_news/val.csv').reset_index(drop=True)

# Zhaoyang Code
# Preprocessing the contents text: such as remove all non alphabet characters, 
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess_text(text):
    # Replace characters that are not between a to z or A to Z with whitespace
    #text = re.sub('[^a-zA-Z0-9]', ' ', text)

    # Convert all characters into lowercase
    text = text.lower()

    # Remove inflectional morphemes like "ed", "est", "s", and "ing" from their token stem
    #text = [stemmer.stem(word) for word in text.split()]

    # Join the processed words back into a single string
    #text = ' '.join(text)

    return text


# Performs all the required data cleaning and preprocessing steps

################# Performs all the required data cleaning and preprocessing steps #################

def data_preprocessing(dataset):
    dataset['body_text'] = dataset['body_text'].astype(str).apply(preprocess_text)

    #Dealing with empty datapoints for metadata columns - subject, speaker, job, state,affiliation, context
    merged_info = []
    for i in range(len(dataset)):
        body_text    =  dataset['body_text'][i]
        #combining all the meta data columns into a single column
        merged_info.append(str(body_text)) 
      
    #Adding cleaned and combined metadata column to the dataset

    ##
#### MERGED INFO SAME AS BODY_TEXT, LEGACY CODE
    ##
    dataset['merged_info'] = merged_info
    # dataset = dataset.drop(columns=['statement_id', 'pre_label','body_text', 'subject', 'speaker','speaker_job_title', 'state_info',
    #                'party_affiliation', 'barely_true_count','false_count','half_true_count','mostly_true_count','pants_on_fire_count',
    #                'context','justification'])
  
    dataset.dropna() #Dropping if there are still any null values

    return dataset

training_preprocessed = data_preprocessing(training)
testing_preprocessed = data_preprocessing(testing)
validation_preprocessed = data_preprocessing(validation)

################# Check the Datasets ################# 
print(training_preprocessed.head(2))

## POS-Tagging Part
# Calculate keyword list length l as n percentage of document length
# Map document into list of tuples containing word and its POS.
# Optional: count percentages of adjectives compared to rest
# Filter document for words in tuples with desired POS(adj or adv)
# run TFIDF on filtered document to find l highest words
# write into csv as merged_info
# make XLNet .py file read csv and run XLNet on merged_info column to calculate doc truth value.
def extract_pos_words(text):
    tokens = word_tokenize(text)
    total_words = len(tokens)
    num_pos_words = math.ceil(total_words * (n / 100))
    tagged_words = pos_tag(tokens)
    pos_words = [word for word, tag in tagged_words if tag.startswith('JJ') or tag.startswith('RB')]
    # 如果提取的数量超过总词数，限制在总词数内
    if num_pos_words > len(pos_words):
        num_pos_words = len(pos_words)
    return ' '.join(pos_words[:num_pos_words])
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text      ####(I deleted all numbers here)
training_preprocessed['POS_words'] = training_preprocessed['merged_info'].apply(extract_pos_words).astype(str).apply(preprocess_text)
testing_preprocessed['POS_words'] = testing_preprocessed['merged_info'].apply(extract_pos_words).astype(str).apply(preprocess_text)
validation_preprocessed['POS_words'] = validation_preprocessed['merged_info'].apply(extract_pos_words).astype(str).apply(preprocess_text)

training_preprocessed.to_csv(f"~/Desktop/4thSem/DATALab/fake_and_real_news/POS{n}train.csv", index=False)
testing_preprocessed.to_csv(f"~/Desktop/4thSem/DATALab/fake_and_real_news/POS{n}test.csv", index=False)
validation_preprocessed.to_csv(f"~/Desktop/4thSem/DATALab/fake_and_real_news/POS{n}val.csv", index=False)