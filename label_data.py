import os
import json
import pandas as pd


df = pd.read_csv("./MANUAL_LABELS_Sentence_Intent_Classification_1_30000~Extract2_Deduped.tsv", sep="\t")
df.head()

def lower_labels(in_str):
    return in_str.lower()

df["jd_sent_manual_label_NEW"] = df["jd_sent_manual_label_NEW"].apply(lower_labels)

select_labels_list = ["certification","skill_knowledge","experience"]
NER_df = df.loc[df['jd_sent_manual_label_NEW'].isin(select_labels_list)]

# IDs -->              0         1        2        3          4        5
# Entity Labels --> Language, Service, Degree, Experience, Practice, Library


def label_row(in_str):
    word_dict = dict()
    current_str = in_str + ' '
    for i in range(len(current_str)):
        word_dict[i] = current_str[i]

    print(current_str)
    print(word_dict)

    entities = []
    entity_id_to_label = {0:"LANGUAGE",1:"SERVICE",2:"DEGREE",3:"EXPERIENCE",4:"PRACTICE",5:"LIBRARY"}
    while True:
        print("Enter 'd' for done or enter starting index")
        first_inp = input()
        if str(first_inp) == "d":
            return entities
        else:
            print("Enter Final Index -Remember This is not Included-")
            final_inp = input()
            print("Enter Entity Label ID")
            id_inp = input()
            final_tuple = (int(first_inp),int(final_inp),entity_id_to_label[int(id_inp)])
            entities.append(final_tuple)

final_list = []

NER_df["NER_label"] = NER_df["jd_sentence_text"].apply(label_row)
NER_df.to_json("MANUAL_NER_labeled.json",orient="records")
