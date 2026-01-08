import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab', quiet=True)
stemmer = nltk.stem.PorterStemmer()
import os
from pathlib import Path
import math


AIWords = 'delve underscore showcase unveil intricate meticulous pivotal heighten nuance bolster foster interplay garnered excels aligning notably additionally significant potential findings crucial exhibit comprehensive notably enhance valuable'
AIWordsStemmed = [stemmer.stem(word) for word in word_tokenize(AIWords)]

def txtToWords(filename):
    file = open(filename)
    file = file.read().replace('\n', ' ')
    clean_text = re.sub(r'[^a-zA-Z ]', '', file)
    clean_text = clean_text.lower()
    stemmedText = [stemmer.stem(word) for word in word_tokenize(clean_text)]
    return stemmedText

def countWords(stemmedText):
    occurancesTerm = dict()
    totalWords = len(stemmedText)
    for word in AIWordsStemmed:
        count = stemmedText.count(word)
        if occurancesTerm.get(word) is None:
            occurancesTerm[word] = count
        else:
            occurancesTerm[word] = occurancesTerm.get(word) + count
    return occurancesTerm, totalWords

def countTotalDocuments(*folder):
    num_files = 0
    for folderName in folder:
        folder_path = Path(folderName)
        num_files += sum(1 for f in folder_path.iterdir() if f.is_file())
    return num_files

def countDocumentsWithTerm(*folder):
    documentTermsFinalDict = dict()
    for folderName in folder:
        folder_path = Path(folderName)
        documentTerms = dict()
        for f in folder_path.glob('*.txt'):
            stemmedText = txtToWords(f)
            for word in AIWordsStemmed:
                if word in stemmedText:
                    if documentTerms.get(word) is None:
                        documentTerms[word] = 1
                    else:
                        documentTerms[word] = documentTerms.get(word) + 1
        documentTermsFinalDict = combineDictionaries(documentTermsFinalDict, documentTerms)
    return documentTermsFinalDict

def combineDictionaries(*dicts):
    finalDict = dict()
    for tempDict in dicts:
        for key in tempDict:
            if finalDict.get(key) is None:
                finalDict[key] = tempDict.get(key)
            else:
                finalDict[key] = finalDict.get(key) + tempDict.get(key)
    return finalDict

def findCombinedTFIDFScore(document):
    TFIDFDict = dict()
    occurancesTerm, totalWordsInDocument = countWords(txtToWords(document))
    documentsWithTerm = countDocumentsWithTerm('HumanArticles')
    totalDocuments = countTotalDocuments('HumanArticles')
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
    return combinedTFIDF
    



if __name__ == "__main__":
    print(findCombinedTFIDFScore('HumanArticles\l_Article2H.txt'))
    print(findCombinedTFIDFScore('HumanArticles\l_Article3H.txt'))
    print(findCombinedTFIDFScore('HumanArticles\l_Article4H.txt'))
    print(findCombinedTFIDFScore('HumanArticles\l_Article5H.txt'))
    print(findCombinedTFIDFScore('HumanArticles\l_Article6H.txt'))
    print(findCombinedTFIDFScore('HumanArticles\l_Article7H.txt'))
    print(findCombinedTFIDFScore('DeepSeekArticles\l_Article1DS.txt'))
    print(findCombinedTFIDFScore('DeepSeekArticles\l_Article2DS.txt'))
    print(findCombinedTFIDFScore('DeepSeekArticles\l_Article3DS.txt'))
    print(findCombinedTFIDFScore('ChatGPTArticles\l_Article1GPT.txt'))
    print(findCombinedTFIDFScore('ChatGPTArticles\l_Article2GPT.txt'))
    print(findCombinedTFIDFScore('ChatGPTArticles\l_Article3GPT.txt'))
    print(findCombinedTFIDFScore('ChatGPTArticles\l_Article4GPT.txt'))


