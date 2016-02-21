
# coding: utf-8

# In[ ]:

#import statements
from nltk.tokenize import word_tokenize as wordTok, sent_tokenize as sentTok
from nltk.tag import pos_tag
from nltk import ne_chunk_sents as chunk
import nltk


# In[ ]:

#function to remove problem characters
def removeNonAscii(text):
    return ''.join(i for i in text if ord(i)<128)


# In[ ]:

#open, read, remove problem characters, remove punctuation for tokenization
teslaSummary = removeNonAscii(open(
        "D:\Google Drive\Grad School\UIC\MS-MIS\IDS566\Assignment4\TeslaSummary1.txt", "r").read())
NYTimes = removeNonAscii(open(
        "D:\Google Drive\Grad School\UIC\MS-MIS\IDS566\Assignment4\NYTimes.txt", "r").read())
ESPN = removeNonAscii(open(
        "D:\Google Drive\Grad School\UIC\MS-MIS\IDS566\Assignment4\ESPN.txt", "r").read())
#print teslaSummary


# In[ ]:

#apply sentence tokenization
teslaSummarySentTok = sentTok(teslaSummary)
NYTimesSentTok = sentTok(NYTimes)
ESPNSentTok = sentTok(ESPN)
#print teslaSummarySentTok


# In[ ]:

#tokenize the the sentences
teslaSummaryTok = [wordTok(sentence) for sentence in teslaSummarySentTok]
NYTimesTok = [wordTok(sentence) for sentence in teslaSummarySentTok]
ESPNTok = [wordTok(sentence) for sentence in teslaSummarySentTok]
#print teslaSummaryTok


# In[ ]:

#tag the tokens
teslaSummaryPOS = [pos_tag(sentence) for sentence in teslaSummaryTok]
NYTimesPOS = [pos_tag(sentence) for sentence in NYTimesTok]
ESPNPOS = [pos_tag(sentence) for sentence in ESPNTok]
#print teslaSummaryPOS


# In[ ]:

def extract_tags(t):
    entity = []
    # extract tags from each item in chunked_sentences
    for child in t:
        if type(child) == nltk.tree.Tree:  # check type to see if it is a tree
            entity.extend(extract_tags(child))
        else:
            entity.append(child)
    return entity


# In[ ]:

#chunk the tagged sentences
teslaSummaryChunk = chunk(teslaSummaryPOS, binary=True)
NYTimesChunk = chunk(NYTimesPOS, binary=True)
ESPNChunk = chunk(ESPNPOS, binary=True)


# In[ ]:


teslaSummaryChunk = extract_tags(teslaSummaryChunk)
NYTimesChunk = extract_tags(NYTimesChunk)
ESPNChunk = extract_tags(ESPNChunk)
#print teslaSummaryChunk


# In[ ]:



