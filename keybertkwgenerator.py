# %%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# %%
############################ Load the datasets ############################
training   = pd.read_csv('~/Desktop/4thSem/DATALab/fake_and_real_news/train.csv').reset_index(drop=True)
testing    = pd.read_csv('~/Desktop/4thSem/DATALab/fake_and_real_news/test.csv').reset_index(drop=True)
validation = pd.read_csv('~/Desktop/4thSem/DATALab/fake_and_real_news/val.csv').reset_index(drop=True)
'''
training.columns = ['statement_id', 'pre_label','body_text', 'subject', 'speaker','speaker_job_title', 'state_info',
                    'party_affiliation', 'barely_true_count','false_count','half_true_count','mostly_true_count','pants_on_fire_count',
                    'context','justification']
testing.columns = ['statement_id', 'pre_label','body_text', 'subject', 'speaker','speaker_job_title', 'state_info',
                    'party_affiliation', 'barely_true_count','false_count','half_true_count','mostly_true_count','pants_on_fire_count',
                    'context','justification']
validation.columns = ['statement_id', 'pre_label','body_text', 'subject', 'speaker','speaker_job_title', 'state_info',
                    'party_affiliation', 'barely_true_count','false_count','half_true_count','mostly_true_count','pants_on_fire_count',
                    'context','justification']
training['label'] = 1
testing['label'] = 1
validation['label'] = 1

for i in range(len(training)):
        if training['pre_label'][i]=='true':
            training['label'][i] = 1
        elif training['pre_label'][i]=='mostly-true':
            training['label'][i] = 1
        elif training['pre_label'][i]=='half-true':
            training['label'][i] = 1
        elif training['pre_label'][i]=='barely-true':
            training['label'][i] = 0
        elif training['pre_label'][i]=='false':
            training['label'][i] = 0
        elif training['pre_label'][i]=='pants-fire':
            training['label'][i] = 0
        else:
            print('Incorrect label')
            
for i in range(len(testing)):
        if testing['pre_label'][i]=='true':
            testing['label'][i] = 1
        elif testing['pre_label'][i]=='mostly-true':
            testing['label'][i] = 1
        elif testing['pre_label'][i]=='half-true':
            testing['label'][i] = 1
        elif testing['pre_label'][i]=='barely-true':
            testing['label'][i] = 0
        elif testing['pre_label'][i]=='false':
            testing['label'][i] = 0
        elif testing['pre_label'][i]=='pants-fire':
            testing['label'][i] = 0
        else:
            print('Incorrect label')

for i in range(len(validation)):
        if validation['pre_label'][i]=='true':
            validation['label'][i] = 1
        elif validation['pre_label'][i]=='mostly-true':
            validation['label'][i] = 1
        elif validation['pre_label'][i]=='half-true':
            validation['label'][i] = 1
        elif validation['pre_label'][i]=='barely-true':
            validation['label'][i] = 0
        elif validation['pre_label'][i]=='false':
            validation['label'][i] = 0
        elif validation['pre_label'][i]=='pants-fire':
            validation['label'][i] = 0
        else:
            print('Incorrect label')
            '''

import numpy as np
training.replace('', np.nan, inplace=True)
training.replace(' ', np.nan, inplace=True)
training.dropna(inplace=True)
training = training.reset_index(drop=True)

testing.replace('', np.nan, inplace=True)
testing.replace(' ', np.nan, inplace=True)
testing.dropna(inplace=True)
testing = testing.reset_index(drop=True)

validation.replace('', np.nan, inplace=True)
validation.replace(' ', np.nan, inplace=True)
validation.dropna(inplace=True)
validation = validation.reset_index(drop=True)

################# Check the Datasets ################# 
training.head(2)
   

# %%
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

# %%
training_preprocessed = data_preprocessing(training)
testing_preprocessed = data_preprocessing(testing)
validation_preprocessed = data_preprocessing(validation)

################# Check the Datasets ################# 
training_preprocessed.head(2)


# %%
from keybert import KeyBERT

kw_model = KeyBERT(model='all-mpnet-base-v2')

def combine_strings(string_list):
    combined_string = ' '.join(string_list)
    return combined_string

def count_words(preprocessed_document: str) -> int:
    return len(preprocessed_document.strip().split(" "))
    # maybe need to also split dashes for combined words

def print_dataset(dataset):
    for i in range(len(dataset)):
        info = dataset["merged_info"][i]
        print(type(info))
        print(f'{i}th element = {info}')

def dataset_keyword_extraction(dataset, n: float):
    for i in range(len(dataset)):
        keywords = kw_model.extract_keywords(dataset['merged_info'][i], # top 5 is function default
                                             
                                             keyphrase_ngram_range=(1, 1), 

                                             stop_words='english', 

                                             highlight=False,

                                            # top n percent of total words in data set
                                             top_n = int(0.01 * n * count_words(dataset['merged_info'][i])),
                                     
                                             use_mmr=True, diversity=0.5)
        keywords_list_positive = [word[0] for word in keywords if word[1] > 0]
        dataset['merged_info'][i] = combine_strings(keywords_list_positive)
    return dataset

# %%
# 100% KW Test
n: int = 25 # percent
training_preprocessed = dataset_keyword_extraction(training_preprocessed, n)
testing_preprocessed = dataset_keyword_extraction(testing_preprocessed, n)
validation_preprocessed = dataset_keyword_extraction(validation_preprocessed, n)


################# Check the Datasets ################# 
training_preprocessed.head(2)

# %% [markdown]
# ## Save keyword dataset

# %%
training_preprocessed.to_csv(f'~/Desktop/4thSem/DATALab/fake_and_real_news/train_{str(n)}keywords.csv',index=False)
testing_preprocessed.to_csv(f'~/Desktop/4thSem/DATALab/fake_and_real_news/test_{str(n)}keywords.csv',index=False)
validation_preprocessed.to_csv(f'~/Desktop/4thSem/DATALab/fake_and_real_news/val_{str(n)}keywords.csv',index=False)


