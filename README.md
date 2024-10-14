# ``Good'' Less Can be More:\\Fake News Detection with Limited Information

# Limited Information 
KeyWords - Extracts keywords

POS     - Extracts adjectives and adverbs

NER     - Extracts Named Entities



# Files

{x}keywords.py => Input dataset as csv, outputs dataset with new column containing {x} keywords.

{x}SLIM.py    => Input dataset generated from {x}keywords.py, prints output of SLIM model on dataset.

shannon.py => Input dataset as csv, outputs dataset with new column containing shannon score of another column

# How to use

Modify n in a {x}keywords.py file to desired percentage and replace pd.read_csv(...) lines. 

```python {x}keywords.py```

Modify n in a {x}LEMMA.py to be the same as n in {x}keywords.py.

```python {x}SLIM.py```

Modify Fcsv in shannon.py to desired csv and change column names.

```python shannon.py```

# Notes
Column ["merged_info"] used in some files, this column contained purely article text. Replace with column name containing article text for other datasets.

Requirements.txt lists required packages for running SLIM files. Please refer to the github pages of DOCEMB, MisRoBÆRTa, and CapsNet respectively if using the other models.

## Citation

If you use this work, please cite:

Author(s). (Year). Title of the Paper. Journal Name, Volume(Issue), Page numbers. DOI: [DOI link]
