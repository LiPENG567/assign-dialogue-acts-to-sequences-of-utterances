#!/usr/bin/env python3

"""hw2_corpus_tools.py: CSCI544 Homework 2 Corpus Code

USC Computer Science 544: Applied Natural Language Processing

Provides three functions and two data containers:
get_utterances_from_file - loads utterances from an open csv file
get_utterances_from_filename - loads utterances from a filename
get_data - loads all the CSVs in a directory
DialogUtterance - A namedtuple with various utterance attributes
PosTag - A namedtuple breaking down a token/pos pair

Feel free to import, edit, copy, and/or rename to use in your assignment.
Do not distribute.

Written in 2015 by Christopher Wienberg.
Questions should go to your instructor/TAs.
"""
from collections import namedtuple
import csv
import glob
import os
import pycrfsuite
import sys


def get_utterances_from_file(dialog_csv_file):
    """Returns a list of DialogUtterances from an open file."""
    reader = csv.DictReader(dialog_csv_file)
    return [_dict_to_dialog_utterance(du_dict) for du_dict in reader]

def get_utterances_from_filename(dialog_csv_filename):
    """Returns a list of DialogUtterances from an unopened filename."""
    with open(dialog_csv_filename, "r") as dialog_csv_file:
        return get_utterances_from_file(dialog_csv_file)

def get_data(data_dir):
    """Generates lists of utterances from each dialog file.

    To get a list of all dialogs call list(get_data(data_dir)).
    data_dir - a dir with csv files containing dialogs"""
    dialog_filenames = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    for dialog_filename in dialog_filenames:
        yield get_utterances_from_filename(dialog_filename)

DialogUtterance = namedtuple(
    "DialogUtterance", ("act_tag", "speaker", "pos", "text"))

DialogUtterance.__doc__ = """\
An utterance in a dialog. Empty utterances are None.

act_tag - the dialog act associated with this utterance
speaker - which speaker made this utterance
pos - a list of PosTag objects (token and POS)
text - the text of the utterance with only a little bit of cleaning"""

PosTag = namedtuple("PosTag", ("token", "pos"))

PosTag.__doc__ = """\
A token and its part-of-speech tag.

token - the token
pos - the part-of-speech tag"""

def _dict_to_dialog_utterance(du_dict):
    """Private method for converting a dict to a DialogUtterance."""

    # Remove anything with 
    for k, v in du_dict.items():
        if len(v.strip()) == 0:
            du_dict[k] = None

    # Extract tokens and POS tags
    if du_dict["pos"]:
        du_dict["pos"] = [
            PosTag(*token_pos_pair.split("/"))
            for token_pos_pair in du_dict["pos"].split()]
    return DialogUtterance(**du_dict)


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
            'word.lower=' + word,
            'postag=' + postag,
            ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1,
            '-1:postag=' + postag1,
        ])
    else:
        features.append('BOS')
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1,
            '+1:postag=' + postag1,
        ])
    else:
        features.append('EOS')
    return features 

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


#data_dir = '/Users/zailipeng/Desktop/my_research/Important_information/books/CS/csci544/HW2/train/train'
data_dir = sys.argv[1]
data = get_data(data_dir)
data_dir1 = sys.argv[2]
data1 = get_data(data_dir1)

XX_train = []
YY_train = []
XX_test = []
YY_test = []
for seq in data:


    first= []
    fir = seq[0][1][0]

    sent2label = []
    send = []
    x_train = []
#    xtrain = []

    for a in seq:
        send = []
        xtrain = []
        sent2label.append(a[0]) # create a label list
        # define the baseline features
        pos = a[2]
       
        if len(first)> 0 and a[1] == first[-1]:
            fea1 = 'Continue'  
            xtrain.append(fea1)


        first.append(a[1])
        if a[1] == fir:
#            print(a[1])
            fea2 = 'This is first'
            xtrain.append(fea2)
#        else:
#            xtrain['key2'] = []
            
        if pos:
            xtrain.append(str(len(pos))) 
            for j in pos:
                send.append((j[0],j[1]))
            fea3 = sent2features(send)
            flat_list = [item for sublist in fea3 for item in sublist]
            for ele in flat_list:
                xtrain.append(ele) 
        else:
            xtrain.append('NO_WORDS')  
        x_train.append(xtrain)

    y_train = sent2label        
    XX_train.append(x_train)
    YY_train.append(y_train)
      
    
XX_test = []
YY_test = []
for seq in data1:


    first= []
    fir = seq[0][1][0]

    sent2label = []
    send = []
    x_train = []
#    xtrain = []

    for a in seq:
        send = []
        xtrain = []
        sent2label.append(a[0]) # create a label list
        # define the baseline features
        pos = a[2]
       
        if len(first)> 0 and a[1] == first[-1]:
            fea1 = 'Continue'  
            xtrain.append(fea1)


        first.append(a[1])
        if a[1] == fir:
#            print(a[1])
            fea2 = 'This is first'
            xtrain.append(fea2)
#        else:
#            xtrain['key2'] = []
            
        if pos:
            xtrain.append(str(len(pos))) 
            for j in pos:
                send.append((j[0],j[1]))
            fea3 = sent2features(send)
            flat_list = [item for sublist in fea3 for item in sublist]
            for ele in flat_list:
                xtrain.append(ele)
        else:
            xtrain.append('NO_WORDS')                  
        x_train.append(xtrain)

    y_train = sent2label        
    XX_test.append(x_train)
    YY_test.append(y_train)
#print(len(XX_train),len(YY_train))
trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(XX_train, YY_train):
#    print(xseq, yseq)
    trainer.append(xseq, yseq)
        
trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})     

trainer.train('hm2.crfsuite')

#### make predictions:
tagger = pycrfsuite.Tagger()
tagger.open('hm2.crfsuite')
y_pred = [tagger.tag(xseq) for xseq in XX_test]
count = 0
count_c = 0
filename = sys.argv[3]
with open(filename,'w') as zaili:
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            zaili.write(str(y_pred[i][j])+'\n')
    #        print(y_pred[i][j])
            count += 1
            if y_pred[i][j] == YY_test[i][j]:
                count_c += 1
        zaili.write('\n')
Accuracy = count_c / count
print('I am advanced '+str(Accuracy))
