# LEMMAframework

{x}keywords.py => Input dataset as csv, outputs dataset with new column containing {x} keywords.

{x}LEMMA.py => Input dataset generated from {x}keywords.py, prints output of LEMMA model on dataset.

# How to use

Modify n in a {x}keywords.py file to desired percentage and replace pd.read_csv(...) lines. 

```python {x}keywords.py```

Modify n {x}LEMMA.py to be the same as n in {x}keywords.py and replace pd.read_csv(...) lines. 

```python {x}LEMMA.py```

# Notes
Column ["merged_info"] used for our testing, replace with column name containing article text.
