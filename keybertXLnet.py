# %%
import sys
import pandas as pd
import random
import torch
import torch.nn as nn
import datetime
from sklearn.metrics import roc_auc_score,f1_score
from transformers import (
    AdamW,
    XLNetTokenizer,
    XLNetForSequenceClassification)
import re
import time
import warnings
import csv # To save results in output csv
warnings.filterwarnings('ignore')

# %% 
# n = 25 => Run model on csv with Top 25% Keywords.
n = 25

# Determines file name when saving model. Can be replaced with arbitrary value.
T = sys.argv[1]

# Replace csv files.
############################ Load the datasets ############################
training_preprocessed   = pd.read_csv(f'~/csvs/train_{n}keywords.csv')
testing_preprocessed    = pd.read_csv(f'~/csvs/test_{n}keywords.csv')
validation_preprocessed = pd.read_csv(f'~/csvs/val_{n}keywords.csv')


################# Check the Datasets ################# 
training_preprocessed.head(2)

# %%
training_preprocessed['merged_info'][0]

# %%
def count_words(input_string):
    words = input_string.split()
    return len(words)

word_count = count_words(training_preprocessed['merged_info'][0])
print("Number of words:", word_count)

# %%
print('training\n'   , training_preprocessed['label'].value_counts())
print('testing\n'    , testing_preprocessed['label'].value_counts())
print('validation\n' , validation_preprocessed['label'].value_counts())

# %% [markdown]
# # 2: Tokenize the dataset

# %%
# Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top. 
xlnet_model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

# Tell pytorch to run this model on the GPU.
xlnet_model.cuda()

# %%
# Print the original sentence.
print('Original: ', training_preprocessed["merged_info"])

# Split the sentence into tokens - XLNet
print('Tokenized XLNet: ', xlnet_tokenizer.tokenize(training_preprocessed["merged_info"][0]))

# Mapping tokens to token IDs - XLNet
print('Token IDs XLNet: ', xlnet_tokenizer.convert_tokens_to_ids(xlnet_tokenizer.tokenize(training_preprocessed["merged_info"][0])))

# %%
#assigning merged information and labels to separate variables

keywords = training_preprocessed["merged_info"].values
labels   = training_preprocessed["label"].values

# %%
# max_len = 0
# ind = [100,200,300,400,500,512]
# for i in ind:
#   count = 0
#   for merged in POS_words:
#       max_len = max(max_len, len(merged))
#       if len(merged)>i:
#         count+=1
#   print("Count of sentence length over {} is: ".format(i), count)
# print('Max sentence length: ', max_len)

# %%
import torch

# Below function performs tokenization process as required by XLNet models, for a given dataset
def xlnet_tokenization(dataset):
  keywords = dataset["merged_info"].values
  labels   = dataset["label"].values
  # max_length = 4096

  # Tokenize all of the sentences and map the tokens to thier word IDs.
  xlnet_input_ids = []
  xlnet_attention_masks = []

  selected_words_ids = []
  counter = 0

  # For every sentence...
  for sent in keywords:
      #encode_plus function will encode the sentences as required by model, including tokenization process and mapping token ids
      xlnet_encoded_dict = xlnet_tokenizer.encode_plus(
                          str(sent),        #sentence              
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]' tokens 
                          max_length = 512,     #Since we have seen from our analysis that majority of sentences have length less than 300.    
                          pad_to_max_length = True,    # Pad sentences to 256 length  if the length of sentence is less than max_length
                          return_attention_mask = True,   # Create attention mask
                          truncation = True,  # truncate sentences to 256 length  if the length of sentence is greater than max_length
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )
      
    
      
    
      # Add the encoded sentence to the list.    
      xlnet_input_ids.append(xlnet_encoded_dict['input_ids'])
      
      # Add attention mask to the list 
      xlnet_attention_masks.append(xlnet_encoded_dict['attention_mask'])

      
      # collecting sentence_ids
      selected_words_ids.append(counter)
      counter  = counter + 1
      
      
  # Convert the lists into tensors.
  xlnet_input_ids = torch.cat(xlnet_input_ids, dim=0)
  xlnet_attention_masks = torch.cat(xlnet_attention_masks, dim=0)

  labels = torch.tensor(labels)
  selected_words_ids = torch.tensor(selected_words_ids)

  return {"XLNet":[xlnet_input_ids, xlnet_attention_masks, labels]}


# %%
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import TensorDataset, random_split
# function to seed the script globally
torch.manual_seed(0)

#tokenizing train set
token_dict_train = xlnet_tokenization(training_preprocessed)

xlnet_input_ids,xlnet_attention_masks,labels = token_dict_train["XLNet"]

#tokenizing validation set
token_dict_valid = xlnet_tokenization(validation_preprocessed)

xlnet_input_ids_valid,xlnet_attention_masks_valid,labels_valid = token_dict_valid["XLNet"]

#tokenizing test set
token_dict_test = xlnet_tokenization(testing_preprocessed)

xlnet_input_ids_test,xlnet_attention_masks_test,labels_test = token_dict_test["XLNet"]

# %%
# Combine the training inputs into a TensorDataset.
xlnet_train_dataset = TensorDataset( xlnet_input_ids, xlnet_attention_masks, labels) 

# Combine the validation inputs into a TensorDataset.
xlnet_val_dataset = TensorDataset(xlnet_input_ids_valid,xlnet_attention_masks_valid,labels_valid)

# Combine the test inputs into a TensorDataset.
xlnet_test_dataset = TensorDataset(xlnet_input_ids_test,xlnet_attention_masks_test,labels_test)

# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 15

# Create the DataLoaders for our training - Loads the data randomly in batches of size 16
xlnet_train_dataloader = DataLoader(
            xlnet_train_dataset,  # The training samples.
            sampler = RandomSampler(xlnet_train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# Create the DataLoaders for our validation - Loads the data in batches of size 32
xlnet_validation_dataloader = DataLoader(
            xlnet_val_dataset, # The validation samples.
            sampler = SequentialSampler(xlnet_val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# %%
# optimizers - AdamW
# here, i have used default learning rate and epsilon values for both XLNet
xlnet_optimizer = AdamW(xlnet_model.parameters(),
                  lr = 5e-5, 
                  eps = 1e-8 
                )

# %%
from transformers import get_linear_schedule_with_warmup

# Number of training epochs
epochs = 15

# Total number of training steps is [number of batches] x [number of epochs]
total_steps = len(xlnet_train_dataloader) * epochs

# Create the learning rate scheduler.
xlnet_scheduler = get_linear_schedule_with_warmup(xlnet_optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

# %%
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# %%
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# %%
import torch

# tell pytorch to use the gpu if available
if torch.cuda.is_available():    
      
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# %%
import random
# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
loss_values = []

best_acc = -0.1

print(f"{n} Keyword Percentage")

for epoch_i in range(0, epochs):
    #Training 
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_loss = 0
    xlnet_model.train()
    # For each batch of training data...
    for step, batch in enumerate(xlnet_train_dataloader):
      #Report progress after every 30 epochs
        if step % 30 == 0 and not step == 0: 
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # print current training batch and elapsed time
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(xlnet_train_dataloader), elapsed))
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        xlnet_model.zero_grad()      
        outputs = xlnet_model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # model returns a tuple, extract loss value from that tuple
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xlnet_model.parameters(), 1.0)
        xlnet_optimizer.step()
        
        xlnet_scheduler.step()
    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(xlnet_train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        


    #Validation Part
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode    
    xlnet_model.eval()
    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in xlnet_validation_dataloader:  
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
           outputs = xlnet_model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        # Track the number of batches
        nb_eval_steps += 1
 

    # Report the final accuracy for this validation run.
    ACC = eval_accuracy/nb_eval_steps
    print("  Accuracy: {0:.4f}".format(ACC))
    if ACC > best_acc:
        best_acc = ACC;
        print("Save new best model")
        torch.save(xlnet_model.state_dict(),f'{T}best_model{n}.pth')
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("")
print("Training complete!")

# %% [markdown]
# ### Prediction

# %%
xlnet_prediction_sampler = SequentialSampler(xlnet_test_dataset)
xlnet_prediction_dataloader = DataLoader(xlnet_test_dataset, sampler=xlnet_prediction_sampler, batch_size=batch_size)

# load best model
xlnet_model.load_state_dict(torch.load(f'{T}best_model{n}.pth'))

# %%
# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(xlnet_input_ids_test)))

# Put model in evaluation mode
xlnet_model.eval()

# Tracking variables 
predictions , true_labels = [], []

# %%
# Predict 
for batch in xlnet_prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
 
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = xlnet_model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)
print('    DONE.')

# %%
predictions_labels = [item for subitem in predictions for item in subitem]

predictions_labels = np.argmax(predictions_labels, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = [item for sublist in true_labels for item in sublist]

from sklearn.metrics import classification_report, confusion_matrix

print (classification_report(predictions_labels, flat_true_labels,digits=4))
print(confusion_matrix(flat_true_labels, predictions_labels))
confusion_matrix = confusion_matrix(flat_true_labels, predictions_labels)

print('The final accuracy is ', (confusion_matrix[0,0]+confusion_matrix[1,1])/np.sum(confusion_matrix))

# %%
import tensorflow as tf 
# Calculate AUC
auc = tf.keras.metrics.AUC(num_thresholds=100)(flat_true_labels, predictions_labels).numpy()
print(f"The AUC for Trial {T} is ", '%.4f' % auc)

# open output file and write new row of data
trialnumber = T
area_under_curve = auc
final_accuracy = (confusion_matrix[0,0]+confusion_matrix[1,1])/np.sum(confusion_matrix)
f1macro = f1_score(predictions_labels, flat_true_labels, average="macro")

outputDF = pd.dataframe(np.array(trialnumber, area_under_curve, final_accuracy, f1macro),
                        columns = ["trialnumber", "area_under_curve", "final_accuracy", "f1macro"])

outputDF.to_csv(f"{n}output.csv", mode="a", index=False, header=False) 
                        

