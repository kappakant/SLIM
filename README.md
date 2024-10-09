# **L**imited but **E**ffective Information w/o **M**etadata in Language **M**odel for F**A**ke News (LEMMA)


# Limited Information 
KeyWords - Extracts keywords

POS     - Extracts adjectives and adverbs

NER     - Extracts Named Entities



# Files

{x}keywords.py => Input dataset as csv, outputs dataset with new column containing {x} keywords.

{x}LEMMA.py    => Input dataset generated from {x}keywords.py, prints output of LEMMA model on dataset.

shannon.py => Input dataset as csv, outputs dataset with new column containing shannon score of another column

# How to use

Modify n in a {x}keywords.py file to desired percentage and replace pd.read_csv(...) lines. 

```python {x}keywords.py```

Modify n in a {x}LEMMA.py to be the same as n in {x}keywords.py.

```python {x}LEMMA.py```

Modify Fcsv in shannon.py to desired csv and change output destination.

```python shannon.py```

# Notes
Column ["merged_info"] used for in some files, this column contained purely article text. Replace with column name containing article text for other datasets.

Requirements.txt lists required packages for running LEMMA scripts. Please refer to the github pages of DOCEMB, MisRoBÆRTa, and CapsNet respectively if testing the other models.

## Citation

If you use this work, please cite:

Author(s). (Year). Title of the Paper. Journal Name, Volume(Issue), Page numbers. DOI: [DOI link]
