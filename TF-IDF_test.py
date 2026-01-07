import re
import nltk
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab', quiet=True)
stemmer = nltk.stem.PorterStemmer()

AIWords = 'a the delve underscore showcase unveil intricate meticulous pivotal heighten nuance bolster foster interplay garnered excels aligning notably additionally significant potential findings crucial exhibit comprehensive notably enhance valuable'
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

def totalDocuments(folder):
    folder_path = Path(folder)
    num_files = sum(1 for f in folder_path.iterdir() if f.is_file())
    return num_files

def countDocumentsWithTerm(folder):
    folder_path = Path(folder)
    documentTerms = dict()
    for f in folder_path.glob('*.txt'):
        stemmedText = txtToWords(f)
        for word in AIWordsStemmed:
            if word in stemmedText:
                if documentTerms.get(word) is None:
                    documentTerms[word] = 1
                else:
                    documentTerms[word] = documentTerms.get(word) + 1
    return documentTerms


testStemmedText = txtToWords('HumanArticles\s_Article1H.txt')
print(countWords(testStemmedText))
print(totalDocuments('HumanArticles'))
print(countDocumentsWithTerm('HumanArticles'))
print(countDocumentsWithTerm('AIArticles'))
