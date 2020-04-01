#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
import re


# In[4]:


#文本切分函数
def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence for sentence in sentences)]
    return word_sentences


# In[5]:


#在分词后删除特殊字符
def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None,[pattern.sub('',token) for token in tokens])
    return filtered_tokens


# In[6]:


#在分词前删除特殊字符
def remove_characters_before_tokenization(sentence,keep_apostrophes=False):
    sentence = sentence.split()
    if keep_apostrophes:
        PATTERN=r'[?|$|&|*|%|@|(|)|~]'
        filtered_sentence = re.sub(PATTERN,r'',sentence)
    else:
        PATTERN=r'[^a-zA-Z0-9]'
        filtered_sentence = re.sub(PATTERN,r'',sentence)
    return filtered_sentence
            


# In[7]:


#扩展缩写词，QQ截图
CONTRACTION_MAP = {
    "isn't":"is not",
    "aren't":'are not',
    
}


# In[9]:


#基于词性标注的词形还原
from pattern.en import tag
from nltk.corpus import wordnet as wn

def pos_tag_text(text):
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    tagged_text = tag(text)
    tagged_lower_text = [(word.lower(),penn_to_wn_tags(pos_tag)) for word,pos_tag in tagged_text]
    return tagged_lower_text
        


# In[10]:


#对标注词性后的词语进行词形还原
def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wn.lemmatize(word,pos_tag) if pos_tag else word for word,pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


# In[11]:


#去除特殊符号
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None,[pattern.sub('',token) for token in tokens])
    filtered_text = ' '.join(filterd_tokens)
    return filtered_text


# In[15]:


#去除停用词
def remove_stopwords(text):
    stop = stopwords.words('english')
    with open(r"D:\大创项目\LDA\stopwords\ENstopwords-US.txt", 'r', encoding='utf-8') as f:
        for lines in f:
            stop.append(lines)
    stop = set(stop)
    
    tokens = tokenize_text(text)
    filterd_tokens = [token for token in tokens if token not in stop]
    filtered_text = ' '.join(filterd_tokens)
    return filtered_text


# In[13]:


#文本预处理总函数
def normalized_corpus(corpus,tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = lemmatize_text(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalize_corpus.append(text)
    return normalized_corpus


# In[14]:


#收到一个文本文档时，删除换行符，解析文本，转换为ASCII格式，并分解为句子成分
def parse_document(document):
    document = re.sub('\n',' ',document)
    if isinstance(document,str):
        document = document
    elif isinstance(document,unicode):
        return unicodedata.normalize('NKFD',document).encode('ascii','ignore')
    else:
        raise ValueError('Document is not string or unicode')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


# In[19]:


#回复特殊HTML字符为未转义形式
# from HTMLParser import HTMLParser
# html_parser = HTMLParser()
# def unescape_html(parser,text):
#     return parser.unescape(text)


# In[ ]:





# In[ ]:




