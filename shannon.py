import numpy as np
from collections import Counter
import re

text = """
becoming the world's most advanced and largest professional mitochondrial technology and manufacturing leader is the vision of most biotech companies in the field of mitochondria and mitochondrial research, and preventive medicine.  mitochondrial disorders are multi-systemic and commonplace, and it belongs to a prevalent category of inherited neurometabolic diseases (prevalence: :). such diseases are clinically, genetically and chemically heterogeneous so that recognition and diagnosis are frequently difficult, but thankfully, for most of the patients, recent advances in next-generation sequencing (ngs) technology has led to great improvements in genetic diagnosis. due to the variability of features and nature of the biochemical or genetic defects, early referral to a specialized center is recommended because improved diagnostics and ngs use have enabled an earlier diagnosis for many, as well as sometimes early instigation of effective therapy and precision medicine.  there is currently no cure available for the majority of such conditions, and clinical care is largely restricted to negating the complications of these diseases or assisting prevention through prenatal testing, mitochondrial donation therapy and preimplantation genetic diagnosis. as such, the limited options and lack of curative treatment options have created more opportunities and investments into individual therapy trials, focused on optimizing oxidative phosphorylation and improving antioxidant capacity. in essence, a combination of vitamins, including coenzyme q (ubiquinone), thiamine and riboflavin in addition to other regenerative medicinal approaches in the telomeres, stem cell and senescent cells is recommended. as of february , more than  trials are ongoing worldwide investigating potential therapeutic approaches.  mitochondrion application biomedicine inc. (mab or mitobiomed) is one such company conducting such trials and aims to fulfil this mission by forming a strong competitive team with our medical institutions and customers, as a long-term and trusted technology and capacity provider in the global mitochondrial /cell therapy industry. scientists are working night and day, racing to find a cure for the novel coronavirus. an east-meets-west combination of medicines is now suggested to be the recommended course.  with the advent of an ageing society, many studies in the past ten years have shown that mitochondria are closely related to ageing and degenerative diseases (such as neurodegenerative diseases such as parkinson's disease and alzheimer's disease, etc). this is because when mitochondrial function is abnormal, it will trigger a lack of energy supply and oxidative stress, and even induce cells to enter apoptosis or autophagy, which will cause disease.  mab also establishes a series of training and certification mechanisms for medical personnel (such as doctors) and cooperates with medical institutions and academic institutions around the world. this makes it a prime candidate for impact investing towards sustainability development, in alignment with the united nations and its  sustainable development goals (sdgs).  the company along with other biotech companies in the field have developed many product lines, including mitochondrial activated stem cells (mitocell), mitochondrial biologics (mitobio), mitochondrial activated immune cells (makcell), mitochondrial function tests (mitoscan), mitochondrial aesthetic medicine, mitochondrial cell bank(mitobank), mitochondria activated herbal extracts (dynamito), mitochondria activated health food (mitofood), mitochondria activation culture medium(stemoto), etc., mitocell, mitobio, and makcell are all ready for the next phase of clinical development. other products are cooperatively sold in the fields of preventive health care, medical beauty, cell culture, detection and analysis. these products have proven safety and effectiveness in preliminary studies.  furthermore, mitochondria for stem cell therapy is another frontier for regenerative medicine. stem cells are primitive and unspecialized cells, they are a type of cells that are not sufficiently differentiated and have the potential to regenerate various tissues and organs. in medicine, stem cells have long been considered to have potential for medical applications because of their strong potential for treating physical damage caused by disease, aging, genetic factors or trauma. in international research, mitochondria are an important factor affecting the quality of stem cells. by analyzing the function of mitochondria in stem cells as a way to identify the quality of cells, it provides a scientific way to identify cell quality. the company's research is to improve the quality and function of stem cells by screening and increasing mitochondrial function. stem cells activated by mitochondria have better therapeutic effects. our company's mitochondrial activated stem cells have also been proven to be useful in the treatment of parkinson's disease, osteoarthritis, multiple system atrophy  lastly, in addition to the aforementioned products in the market, mitochondrial replacement techniques (mrt) work to prevent transmitting mitochondrial dna (mtdna) diseases from mother to the next generation. such diseases vary in presentation and severity. the goal of mrt is to prevent the transmission of such diseases by creating an embryo with nuclear dna (ndna) from the intended mother and mtdna from a woman with non-pathogenic mtdna through modification of either an oocyte (egg) or zygote (fertilized egg). this will help alleviate common symptoms including developmental delays, seizures, weakness and fatigue, muscle weakness, vision loss, and heart problems, which in combination lead to an increase of morbidity and in some cases premature death. while research continues to be conducted, if effective, mrt could satisfy the desire of women seeking to have a genetically related child without the risk of passing on mtdna disease.  in the us alone, mitochondrial diseases affect around  in , individuals. it is also estimated that ,-, children are born with a mitochondrial disease every year therefore, there is a direct need to focus on building a world class professional mitochondrial technology and manufacturing leader, with a purpose in creating affordable healthcare and biotech leadership in preventative medicine. it also creates a space for the community of social impact investing to be more deeply involved in health and wellbeing
"""

words = re.findall(r"\b\w+\b|'\w+", text.lower())
#words = [word.lower() for word in text.split()]

# NER tagging words
ner_tagging_words = ['ngs', 'due', 'ngs', 'mab', 'mitobiomed', 'parkinson', 'alzheimer', 'mab', 'united', 'nations', 
 'sdgs', 'company', 'mitocell', 'mitobio', 'makcell', 'mitoscan', 'mitobank', 'dynamito', 'mitofood', 
 'stemoto', 'mitocell', 'mitobio', 'makcell', 'mitochondria', 'stem', 'stem', 'parkinson', 
 'osteoarthritis', 'multiple', 'system', 'atrophy', 'lastly', 'mrt', 'dna', 'mtdna', 'mrt', 
 'dna', 'ndna', 'mtdna', 'mrt', 'mtdna', 'us']

word_counts = Counter(words)

total_words = sum(word_counts.values())

# NER tagging words significance level Shannon entropy
def compute_significance_and_entropy(ner_words, word_counts, total_words):
    significance_entropy = {}
    for word in ner_words:
        # (TF)
        word_freq = word_counts[word] if word in word_counts else 1
        significance_level = word_freq / total_words
        # Shannon entropy
        entropy = -significance_level * np.log2(significance_level) if significance_level > 0 else 0
        # significance level / entropy 
        significance_entropy[word] = entropy/significance_level
    return significance_entropy

# 计
significance_entropy_dict = compute_significance_and_entropy(ner_tagging_words, word_counts, total_words)

# 计算这组词的总score
total_score = sum(significance_entropy_dict.values())

# 输出结果
print("NER tagging words' significance level * Shannon entropy:")
for word, score in significance_entropy_dict.items():
    print(f"Word: {word}, Score: {score:.4f}")

print(f"\nFinal score for this group of words: {total_score:.4f}")

text2 = "in: natural medicine we all know that it’s important to eat our vegetables. at least, that’s what most of us have heard since we were kids. what our mother’s told us as when we were young, our doctors tell us as we get older. sometimes though, it helps to have a more specific reason than high cholesterol, or even a motherly “because i said so.” especially for people who aren’t big fans of eating organic greens. according to a study conducted at oregon state university’s linus pauling micronutrient research institute confirms that sulforaphane, a phytochemical found in broccoli and related cruciferous vegetables, such as cauliflower and cabbage , have a natural ability to target and attack prostate cancer cells without harming neighboring cells [ 1 ] . unconnected studies suggest it may have similar promise for breast cancer. the active chemicals found in everyday foods – such as broccoli – are often much more potent than people would imagine. if fact, determining how to safely adapt these chemical ingredients for medical use is one of the biggest hurdles researchers face. even edible plants that are considered “rich” in a given nutritional substance, contain relatively low amounts of it by volume. the vast majority of these compounds may also become toxic to humans if taken in large enough concentrations. while a number of previous investigations have proven that sulforaphane is able to attack both benign and malignant cancer cells, the oregon state study is one of the first to prove that it is effective without disrupting otherwise healthy tissue. this gives researchers a tremendous tool for developing new, low-risk treatment options, and is likely to encourage additional research into the healing potential of other seemingly mundane edible plants. realistically, it could be some time before these findings are applied to any sort of drug development or cancer treatment in a traditional hospital setting. meanwhile though, the researchers behind the study recommend that we all eat more organic cruciferous vegetables. besides broccoli, a number of readily available cruciferous vegetables contain naturally large amounts of sulforaphane. some good examples of foods high in this important phytochemical include mild and spicy radishes, turnips, watercress, cabbage, arugula, kale, chard, and most other leafy greens. unrelated studies also suggest a variety of other cancer-fighting compounds may be present in other herbs and garden vegetables. celery and parsley , for instance, are especially rich in apigenin – a substance that has shown remarkable promise for fighting breast cancer. trace amounts of apigenin are also found in oranges, apples, and some tree nuts. the problem is, it’s very difficult for the body to effectively extract it from any of these foods on its own. references: oregon state university. study confirms safety, cancer-targeting ability of nutrient in broccoli . news & research communications. 2011 june 9. submit your review"
ner2 = [
"phytochemical", "sulforaphane", "herbs", "ingredients", "broccoli", "vegetables", "micronutrient", "parsley", 
"radishes", "kale", "cauliflower", "cholesterol", "cabbage", "nutritional", "arugula", "cancer", "celery", "plants",
"nutrient", "chemicals", "foods", "turnips", "malignant", "edible", "organic", "cells", "medicine", "chemical", "compounds",
"cruciferous", "greens", "leafy", "studies", "findings", "healing", "doctors", "apples", "medical", "treatment", "breast", 
"researchers", "potent", "concentrations", "healthy", "drug", "spicy", "substance", "benign", "toxic", "tissue", "oranges",
"garden", "eat", "watercress", "safely", "prostate", "proven", "eating", "harming", "natural", "extract", "variety", "apigenin", 
"chard", "helps", "motherly", "face", "research", "given", "effectively", "attack", "references", "humans", "safety", "prove", 
"relatively", "vast", "effective", "risk", "disrupting", "promise", "gives", "suggest", "contain", "tell", "submit", "adapt", 
"mild", "remarkable", "encourage", "mother", "study", "investigations", "tremendous", "naturally", "taken", "fact", "amounts", 
"seemingly", "know", "readily", "nuts", "use", "options", "told", "high", "biggest", "considered", "large", "tree", "review", "majority", 
"2011", "examples", "problem", "body", "shown", "news", "mundane", "people", "according", "active", "conducted", "hospital", "developing", 
"important", "aren", "target", "imagine", "additional", "good", "volume", "able", "especially", "big", "realistically", "heard", "university", 
"determining", "number", "everyday", "kids", "rich", "potential", "institute", "targeting", "said", "unrelated", "june", "confirms", 
"specific", "recommend"
]

words2 = re.findall(r"\b\w+\b|'\w+", text2.lower())
word2_counts = Counter(words2)
total_words2 = sum(word2_counts.values())

significance_entropy_dict2 = compute_significance_and_entropy(ner2, word2_counts, total_words2)

total_score = sum(significance_entropy_dict2.values())

# 输出结果
print("Second article")
print("NER tagging words' significance level * Shannon entropy:")
for word, score in significance_entropy_dict2.items():
    print(f"Word: {word}, Score: {score:.4f}")

print(f"\nFinal score for this group of words: {total_score:.4f}")

### Apply to PD dataframe
def compute_significance_and_entropy(ner_words, word_counts, total_words):
    significance_entropy = {}
    for word in ner_words:
        # (TF)
        # Changed else 0 to else 1 to avoid division by 0 from minor syntax differences
        # i.e. #dallaspoliceshooting turning into dallaspoliceshooting
        word_freq = word_counts[word] if word in word_counts else 1
        significance_level = word_freq / total_words
        # Shannon entropy
        entropy = -significance_level * np.log2(significance_level) if significance_level > 0 else 0
        # significance level / entropy 
        significance_entropy[word] = entropy/significance_level
    return significance_entropy

def sig_entropy(row, column):
    article = str(row[column]) + " " + row["body_text"] #row["title"] + " " + row["body_text"]
    partial = str(row[column])
    partialwords = re.findall(r"\b\w+\b|'\w+", partial.lower())
    words = re.findall(r"\b\w+\b|'\w+", article.lower())
    word_duplicates = Counter(words)
    words_sum = sum(word_duplicates.values())

    sig_entropy = compute_significance_and_entropy(partialwords, word_duplicates, words_sum)
    score = sum(sig_entropy.values())
    return score

import pandas as pd
# Keybert
#n = 35
#F = f"recovery/train_keybert_{n}"
#Fcsv = pd.read_csv(f"~/Desktop/DATALab/minimum/{F}.csv")

# POS
#n = 20
#F = f"recovery/train_NER"
#Fcsv = pd.read_csv(f"~/Desktop/DATALab/minimum/{F}.csv")

# Title
Fcsv = pd.read_csv("~/Desktop/DATALab/minimum/recovery/train_title.csv")
'''
Fcsv["KWS_sig_entropy_score"] = Fcsv.apply(lambda x: sig_entropy(x, "merged_info"), axis=1)
Fcsv["sig_entropy_score"] = Fcsv.apply(lambda x: sig_entropy(x, "body_text"), axis=1)
Fcsv["title_sig_entropy_score"] = Fcsv.apply(lambda x: sig_entropy(x, "title"), axis=1)
print(Fcsv["KWS_sig_entropy_score"].head())'''

print(Fcsv.head())

Fcsv["sig_entropy_score"] = Fcsv.apply(lambda x: sig_entropy(x, "body_text"), axis=1)

Fcsv.to_csv(f"~/Desktop/DATALab/minimum/Rtrain_fulltext.csv", index=False)

#calculate average shannon score
average_shannonKW = Fcsv["sig_entropy_score"].sum() / len(Fcsv)
print(f"Keywords: {average_shannonKW}")

'''
average_shannonFT = Fcsv["sig_entropy_score"].sum() / len(Fcsv)
print(f"Full Text: {average_shannonFT}")

average_shannonTitle = Fcsv["title_sig_entropy_score"].sum() / len(Fcsv)
print(f"Title: {average_shannonTitle}")

average_shannonKWperWord = average_shannonFT / Fcsv["merged_info"].apply(len).sum()
average_shannonTitleperWord = average_shannonTitle / Fcsv["title"].apply(len).sum()
print(f"Title: {average_shannonTitleperWord} vs Keywords: {average_shannonKWperWord}")'''