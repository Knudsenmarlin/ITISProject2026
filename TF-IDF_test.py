import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab', quiet=True)
stemmer = nltk.stem.PorterStemmer()
import os
from pathlib import Path
import math
import numpy as np

#Defining our list of words used for check for AI-usage.
AIWordsList = ['delve', 'underscore', 'showcase', 'unveil', 'intricate', 'meticulous', 'pivotal', 'heighten',
               'nuance', 'bolster', 'foster', 'interplay', 'garnered', 'excels', 'aligning', 'notably', 'additionally', 
               'significant', 'potential', 'crucial', 'exhibit', 'comprehensive', 'notably', 'valuable', 'enhance',]
AIWords = ' '.join(AIWordsList)

#Putting the words on their shortest possible form.
AIWordsStemmed = [stemmer.stem(word) for word in word_tokenize(AIWords)]

#Turn a .txt file into a usable list of words.
def txt_to_words(filename):
    file = open(filename, encoding="utf-8", errors="replace")
    file = file.read().replace('\n', ' ')
    clean_text = re.sub(r'[^a-zA-Z ]', '', file)
    clean_text = clean_text.lower()
    stemmedText = [stemmer.stem(word) for word in word_tokenize(clean_text)]
    return stemmedText

#Count the occurnces of all our AI-terms in the given document.
def count_words(stemmedText):
    occurancesTerm = dict()
    totalWords = len(stemmedText)
    for word in AIWordsStemmed:
        count = stemmedText.count(word)
        if occurancesTerm.get(word) is None:
            occurancesTerm[word] = count
        else:
            occurancesTerm[word] = occurancesTerm.get(word) + count
    return occurancesTerm, totalWords

#Count the total amount of documents in mulitple folders, only counting files. This is needed for calculating the "IDF" part of "TF-IDF".
def count_total_documents(*folder):
    num_files = 0
    for folderName in folder:
        folder_path = Path(folderName)
        num_files += sum(1 for f in folder_path.iterdir() if f.is_file())
    return num_files

#Count whether or not one of our AI-terms was used in a .txt file. This is for calculating the "IDF" part of "TF-IDF".
def count_documents_with_term(*folder):
    documentTermsFinalDict = dict()
    for folderName in folder:
        folder_path = Path(folderName)
        documentTerms = dict()
        for f in folder_path.glob('*.txt'):
            stemmedText = txt_to_words(f)
            for word in AIWordsStemmed:
                if word in stemmedText:
                    if documentTerms.get(word) is None:
                        documentTerms[word] = 1
                    else:
                        documentTerms[word] = documentTerms.get(word) + 1
        documentTermsFinalDict = combine_dictionaries(documentTermsFinalDict, documentTerms)
    return documentTermsFinalDict

#This function combines the results of multiple dictionaries together. This is so we can have different folders in different places, but put their counts together when needed.
def combine_dictionaries(*dicts):
    finalDict = dict()
    for tempDict in dicts:
        for key in tempDict:
            if finalDict.get(key) is None:
                finalDict[key] = tempDict.get(key)
            else:
                finalDict[key] = finalDict.get(key) + tempDict.get(key)
    return finalDict

documentsWithTerm = count_documents_with_term('HumanArticles', 'ChatGPTArticles', 'DeepSeekArticles')
totalDocuments = count_total_documents('HumanArticles', 'ChatGPTArticles', 'DeepSeekArticles')

#The magic happens, we find the actual TF-IDF score of a given document. 
def find_combined_TFIDF_score(document):
    TFIDFDict = dict()
    occurancesTerm, totalWordsInDocument = count_words(txt_to_words(document))
    for word in AIWordsStemmed:
        termFrequency = occurancesTerm[word] / totalWordsInDocument
        if documentsWithTerm.get(word) is None:
            inverseDocumentFrequency = 0
        else:
            inverseDocumentFrequency = math.log(totalDocuments / documentsWithTerm[word])
        TFIDF = termFrequency * inverseDocumentFrequency
        TFIDFDict[word] = TFIDF
    combinedTFIDF = 0
    for key in TFIDFDict:
        combinedTFIDF += TFIDFDict[key]
    print(combinedTFIDF)
    return combinedTFIDF

#Here we find the average TF-IDF score of a folder (or mulitple folders), and the standard deviation. 
def average_TDFIF_score(*folder):
    scoreList = np.array([])
    for folderName in folder:
        folder_path = Path(folderName)
        for f in folder_path.iterdir():
            if f.is_file():
                scoreList = np.append(scoreList, find_combined_TFIDF_score(f))
    scoreMean = np.mean(scoreList)
    scoreStd = np.std(scoreList, ddof=1)
    return scoreMean, scoreStd

if __name__ == "__main__":
    print(find_combined_TFIDF_score('HumanArticles/l_Article2H.txt'))
    print(average_TDFIF_score('HumanArticles'))
    print(average_TDFIF_score('ChatGPTArticles'))
    print(average_TDFIF_score('DeepSeekArticles'))
