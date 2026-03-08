Here we have implemented the training loop of Word2Vec for the following dataset:
https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu/viewer/pqa_artificial?row=6

This code includes the concepts of forwards pass, gradient derivation and parameter updates. For this implementation, Skip-gram architecture was used with Negative sampling.
Here the clear objective was to understand vector embeddings for words that appear in similar contexts and have similar vector representations. 

The Skip-gram model uses the center word of a sentence to predict context words.

The flow of the code in the form of a flowchart:
dataset rows converted to text documents-> Tokenization-> vocab building-> skip-gram pair generation-> Negative sampling-> forwards pass-> Gradient computaiton-> parameter updates

I used the artificial questions(pqa_artificial) part of the dataset and due to my laptop's limited capacity, used only the first 300 rows out of 9000 for implementation.

This repository includes an embeddings file that shows all the word embeddings and a vocab file that provides mapping between word and its numeric ID's used by the model.

test.py is the python code to view the word embeddings. This can't be done in notepad as it is a binary file.
