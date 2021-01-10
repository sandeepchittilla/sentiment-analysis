# scikit-learn (>=0.19.1), pandas (>= 0.23.0), nltk (>= 3.3.0) and spacy (>= 2.0.12), 
# gensim (>=3.8.0), transformers (>=2.0.0).
import pandas as pd
import sklearn
import numpy as np

from textblob import TextBlob
import category_encoders as ce

import pickle
import re
import warnings
import string
from pathlib import Path
from urllib.request import urlretrieve
import gzip

import spacy
from spacy import displacy

import nltk
WNlemma = nltk.WordNetLemmatizer()
nlp = spacy.load('en_core_web_md',disable=['ner'])

import networkx as nx

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning as CW
from sklearn.linear_model import PassiveAggressiveClassifier
warnings.filterwarnings(action='ignore', category=CW)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import class_weight

class Classifier:


    def __init__(self):
        # body of the constructor
        self.model_filename = 'model.h5'
        self.PATH_TO_DATA = Path('../resources')
        self.embedding_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz'
        self.EMBDEDDING_LENGTH = 300
        self.le = preprocessing.LabelEncoder()
        self.cat_ohe = ce.OneHotEncoder()
        self.tfidf_model = TfidfVectorizer()

        ## Using BERT
        # self.bert_sentence = bert_sentence
        # self.bert_word = bert_word
        
        
        # if self.bert_sentence:
        #     model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        #     # Load pretrained model/tokenizer
        #     self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        #     self.model = model_class.from_pretrained(pretrained_weights)
            
        # if self.bert_word==False:
        self.word2vec = self.Word2VecCustom(self.getWordEmbeddings(), vocab_size=50000)
    
    class Word2VecCustom():

        def __init__(self, filepath, vocab_size=50000):
            self.words, self.embeddings = self.load_wordvec(filepath, vocab_size)
            # Mappings for word/id retrieval:
            self.word2id = {word: idx for idx, word in enumerate(self.words)}
            self.id2word = {idx: word for idx, word in enumerate(self.words)}

        def load_wordvec(self, filepath, vocab_size):
            assert str(filepath).endswith('.gz')
            words = []
            embeddings = []
            with gzip.open(filepath, 'rt') as f:  # Read compressed file directly
                next(f)  # Skip header
                for i, line in enumerate(f):
                    word, vec = line.split(' ', 1)
                    words.append(word)
                    embeddings.append(np.fromstring(vec, sep=' '))
                    if i == (vocab_size - 1):
                        break
            # print('Loaded %s pretrained word vectors' % (len(words)))
            return words, np.vstack(embeddings)

        def encode(self, word):
            # Returns the 1D embedding of a given word
            return self.embeddings[self.word2id[word]]
    
    def getWordEmbeddings(self):
        # New Word Embeddings from above URL
        # Download word vectors, might take a few minutes and about ~2GB of storage space
        en_embeddings_path = self.PATH_TO_DATA / 'cc.en.300.vec.gz'
        if not en_embeddings_path.exists():
            urlretrieve(self.embedding_url, en_embeddings_path)
        return en_embeddings_path

    def parseDependancyGraph(self, doc, sentiment_terms, target_word):
        edges = []
        sentiment_map = {}

        for token in doc:
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),
                              '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)
        # Get the length and path
        entity1 = target_word.lower()

        for term in sentiment_terms:
            try:
                sentiment_map[term] = nx.shortest_path_length(graph, source=entity1, target=term.lower())
            except:
                pass # do nothing if the target word doesnt reach the adjective 
        return {k: v for k, v in sorted(sentiment_map.items(), key=lambda item: item[1])}

    def getClosestModifiers(self, distance_map):
        closest_modifiers = []

        item = None
        if len(distance_map)<2:
            try:
                item = next(iter(distance_map))
            except Exception as e:
                pass
            closest_modifiers.append(item)
        else:
            for i,item in enumerate(distance_map.items()):
                if(i==2):
                    break
                closest_modifiers.append(item[0])
        return closest_modifiers

    def getWordModifiers(self, doc, target_word, sentiment_terms, modifier_list=['amod','advmod','attr']):
        target_word_modifiers=[]
        flag = 0
        for chunk in doc.noun_chunks:
            if(chunk.root.text == target_word):
                # get all children
                children = [i for i in chunk.root.head.children]
                if(children!=[]):
                    flag=1
                    for i in children:
                        if i.dep_ in modifier_list:
                            target_word_modifiers.append(i)

        # in case spacy couldn't get the word modifier from the noun_chunks we traverse the dependancy tree
        if(target_word_modifiers==[]):
            distance_map = self.parseDependancyGraph(doc, sentiment_terms, target_word)
            if(distance_map==None):
                return target_word_modifiers
            target_word_modifiers = self.getClosestModifiers(distance_map)

        return target_word_modifiers


    def getTargetWordModifiers(self, doc, target_word, sentiment_terms):
    #     doc = nlp(preProcessSentence(doc))
        target_word = WNlemma.lemmatize(target_word, pos='v')
        modifier_list=['amod','advmod','attr']
        target_word_modifiers = self.getWordModifiers(doc, target_word, sentiment_terms, modifier_list)
        return target_word_modifiers


    def preProcessSentence(self, x): # function to preprocess each sentence from the training data
        try:
            x = x.lower()
            x = x.replace('+', ' and ')
            x = nlp(x)
            lemmatized = list()
            for word in x:
                lemma = word.lemma_.strip()
                if lemma:
                    lemmatized.append(lemma)
        except:
            pass # do nothing
#             print('x is not String but {}'.format(type(x)))
        return " ".join(lemmatized)

    def getSentimentTerms(self, df_data):
    ## Text processing to get sentiment terms
        docs = nlp.pipe(df_data['basic_clean_sent'])
        sentiment_terms = []
        for doc in docs:
            temp_terms = []
            for token in doc:
                if(not token.is_stop and not token.is_punct and 
                   (token.pos_ == "ADJ" or token.pos_ == "VERB" or token.pos_ == "RB")):
                    temp_terms.append(str(token))
            sentiment_terms.append(temp_terms)
        return sentiment_terms

    def fixNoneModifiers(self, df_data):
        if df_data.target_modifiers[0]==None:
            if df_data['sentiment_terms']==[]:
                return [df_data['target_modified'].split()[0]]
            else:
                return df_data['sentiment_terms']
        else:
            return df_data['target_modifiers']

    def fixSpellings(self, sentence,fix_words):
        fixed_sentence=[]
        to_replace = [',','.']
        for i in to_replace:
            sentence = sentence.replace(i,'')
        for word in sentence.split():
            if word in fix_words:
                corrected = str(TextBlob(word).correct())
                fixed_sentence.append(corrected)
            else:
                fixed_sentence.append(word)
        return str(' '.join(fixed_sentence))
    
    def generateFeatures(self, df_data, training):
        # if self.bert_sentence :
        #     if self.bert_word :
        #         print('Using BERT sentence and word embeddings')
        #         df_sentence_vector, df_qualifier_embedding = self.generateBertEmbeddings(df_data, training)
        #     else:
        #         print('Using BERT only for sentence embeddings')
        #         df_sentence_vector = self.generateBertEmbeddings(df_data, training)
        #         df_qualifier_embedding = df_data.apply(lambda x:
        #                                            self.generateTargetQualifierEmbeddings(x['single_modifier'],self.word2vec),
        #                                            result_type='expand',
        #                                            axis=1)
        #         df_qualifier_embedding.columns = ['embed_'+str(i) for i in range(self.EMBDEDDING_LENGTH)]
            
        # else :
        # print('Using TfIdf embeddings')
        df_sentence_vector = self.generateTfIdfFeatures(df_data, training)
        
        df_qualifier_embedding = df_data.apply(lambda x:
                                               self.generateTargetQualifierEmbeddings(x['single_modifier'],self.word2vec),
                                               result_type='expand',
                                               axis=1)
        df_qualifier_embedding.columns = ['embed_'+str(i) for i in range(self.EMBDEDDING_LENGTH)]
        
        return df_qualifier_embedding, df_sentence_vector

    def generateTargetQualifierEmbeddings(self, series, word2vec):
        for word in series[0]:
            try:
                if word in word2vec.words:
                    return(word2vec.encode(word))
            except:
                try:
                    if str(TextBlob(word).correct()) in word2vec.words:
                        return(word2vec.encode(word))
                except:
                    try:
                        if nlp(word)[0].lemma_ in word2vec.words:
                            return(word2vec.encode(word))
                    except:
                        return(word2vec.encode('average'))

    def generateTfIdfFeatures(self, df_data, training):
        if(training):
            self.tfidf_model.fit(list(df_data['sentence_lemmatized']))
        sentence_tfidf = self.tfidf_model.transform(list(df_data['sentence_lemmatized'])).toarray()
        TFIDF_length = sentence_tfidf.shape[1]
        sentence_tfidf = pd.DataFrame(sentence_tfidf, columns = ['tfidf_'+str(i) for i in range(TFIDF_length)])
        return sentence_tfidf
    
#     def generateBertEmbeddings(self, df_data, training):
#         tokenized = df_data['sentence_lemmatized'].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))
#         # padding
#         max_len = 0
#         for i in tokenized.values:
#             if len(i) > max_len:
#                 max_len = len(i)

#         padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
        
#         input_ids = torch.tensor(np.array(padded))
#         with torch.no_grad():
#             outputs = self.model(input_ids)
            
#         features = outputs[0][:,0,:].numpy()
#         qualifier_features = np.zeros((features.shape[0], features.shape[1]))
        
#         if self.bert_word == True :
#             for j, row_ in enumerate(input_ids):
#                 for i, id_ in enumerate(row_):
#                     if self.tokenizer.convert_tokens_to_ids(df_data['single_modifier'].iloc[j]) == id_:
#                         print(id_,j)
#                         qualifier_features[j] = outputs[0][j,i,:]
                    
#             return pd.DataFrame(features), pd.DataFrame(qualifier_features)
#         return pd.DataFrame(features)

    def retrieveSentenceSubjectivity(self, df_data):
        df_data['polarity_sent'] = df_data['basic_clean_sent'].apply(lambda x:TextBlob(str(x)).sentiment[0])
        df_data['polarity_single_modifier'] = df_data['single_modifier'].apply(lambda x:TextBlob(str(x)).sentiment[0])
        df_data['subjectivity_sent'] = df_data['basic_clean_sent'].apply(lambda x:TextBlob(str(x)).sentiment[1])
        df_data['subjectivity_single_modifier'] = df_data['single_modifier'].apply(lambda x:TextBlob(str(x)).sentiment[1])
        return df_data

    def handleConsecutiveRepetitions(self, sent):
        sent = sent.lower()
        sent = sent.replace('+',' and ')

        clean_sent = []

        for word in sent.split(" "):
            clean_word = re.sub(r'(.)\1+', r'\1\1', str(word))
            clean_sent.append(clean_word)
        return " ".join(clean_sent)

    def labelEncodeColumn(self, df_data, col, training):
        if training:
            self.le.fit(df_data[col])
        df_data[str('encoded_'+col)] = self.le.transform(df_data[col])
        return df_data
    
    def inverseLabelEncode(self, y_pred):
        return self.le.inverse_transform(y_pred)

    def oneHotEncodeColumn(self, df_data, cols, training):
        self.cat_ohe.cols = cols
        if training:
            self.cat_ohe.fit(df_data)
        df_data = self.cat_ohe.transform(df_data)
        return df_data

    def split_aspect_col(self, df_data):
        df_data['aspect_cat']    = df_data['aspect'].apply(lambda x : str(x).split('#')[0])
        df_data['aspect_subcat'] = df_data['aspect'].apply(lambda x : str(x).split('#')[1])
        return df_data

    def clubSentenceUsingTarget(self, df_data):
        clubbed_sentences = []
        for index, row in df_data.iterrows():
            temp = row.sentence.replace(row.target,row.target_modified)
            clubbed_sentences.append(temp)
        df_data['clubbed_sentence'] = clubbed_sentences
        return df_data
    
    def is_elongated(self, sent):
        is_elong = []
        for i in sent:
            elong = re.search(r'(.)\1{2}', str(i))
            if elong is not None:
                is_elong.append(1)
            else:
                is_elong.append(0)
        return(is_elong)

    def is_caps(self, sent):
        is_caps = []
        for i in sent:
            caps = re.search(".*[A-Z].*[A-Z].*", str(i))
            if caps is not None:
                is_caps.append(1)
            else:
                is_caps.append(0)
        return(is_caps)
    
    def mapCharLength(self, sent):
        char_len = len(str(sent))

        if char_len<50:
            return 0
        elif char_len<75:
            return 1
        elif char_len<100:
            return 2
        elif char_len<150:
            return 3
        else:
            return 4
    
    def prepareInputData(self, df_data, training):
        df_data.rename(columns={0:'polarity',
                            1:'aspect',
                            2:'target',
                            3:'offset',
                            4:'sentence'
                           }, 
                     inplace=True)

        # combining target phrases with more than one word into one word
        df_data['target_modified'] = df_data['target'].apply(lambda x: ''.join(str(x).split(' ')))
        df_data.target_modified.apply(lambda x:len(x.split(" "))).value_counts()

        # making similar changes to the target phrase occurrence in the original sentence
        df_data = self.clubSentenceUsingTarget(df_data)

        # splitting aspect column into aspect category and sub-category
        df_data = self.split_aspect_col(df_data)

        # one hot encoding aspect category and subcategory
        df_data = self.oneHotEncodeColumn(df_data, cols = ['aspect_cat','aspect_subcat'], training=training)

        # boolean feature 1/0
        df_data['is_elong'] = self.is_elongated(df_data['clubbed_sentence'])
        
        # 
        df_data['is_caps'] = self.is_caps(df_data['clubbed_sentence'])
        
        # 
        df_data['char_len'] = df_data['clubbed_sentence'].apply(lambda x: self.mapCharLength(x))
        
        # handling words with repeating consecutive letters
        df_data['basic_clean_sent'] = df_data['clubbed_sentence'].apply(lambda cell:self.handleConsecutiveRepetitions(cell))

        # lemmatizing cleaned sentence
        df_data['sentence_lemmatized'] = df_data['basic_clean_sent'].apply(lambda x: self.preProcessSentence(x))
        
        # add column for the sentiment terms
        df_data['sentiment_terms'] = pd.Series(self.getSentimentTerms(df_data))

        # getting target modifiers via spacy's noun chunking or by searching in graph
        df_data['target_modifiers'] = df_data.apply(lambda x : self.getTargetWordModifiers(nlp(x['basic_clean_sent']),x['target_modified'],x['sentiment_terms']),axis=1)

        # if the modifier returned is [None] then use the modifiers from 'sentiment_terms' column
        df_data['final_modifiers'] = df_data.apply(lambda x: self.fixNoneModifiers(x),axis=1)


        df_data['single_modifier'] = df_data['final_modifiers'].apply(lambda x: x[0])
        
        df_data['single_modifier'] = df_data['single_modifier'].apply(lambda x:[str(x)])

        # label encoding the polarity column as {0,1,2}
        df_data = self.labelEncodeColumn(df_data, col = 'polarity', training=training)

        df_data = self.retrieveSentenceSubjectivity(df_data)

        # if you want to save the dataframe uncomment below
        ## if training:
        ##     df_data.to_csv('../data/df_train_data.csv')
        ## else:
        ##     df_data.to_csv('../data/df_test_data.csv')
                
        df_qualifier_embedding, df_sentence_vector = self.generateFeatures(df_data, training=training)
    
        req_cols = ['aspect_cat_1', 'aspect_cat_2', 'aspect_cat_3','aspect_cat_4', 'aspect_cat_5', 'aspect_cat_6', 
                    'aspect_subcat_1', 'aspect_subcat_2', 'aspect_subcat_3', 'aspect_subcat_4', 'aspect_subcat_5', 'is_caps', 
                    'polarity_sent', 'subjectivity_sent', 'polarity_single_modifier', 'subjectivity_single_modifier',
                   ]

        train_X = pd.concat([df_data[req_cols], df_qualifier_embedding, df_sentence_vector],axis=1)
        train_y = df_data.pop('encoded_polarity')

        return train_X, train_y.ravel()
    
    # @ignore_warnings(category=ConvergenceWarning)
    def train(self, trainfile):
        df_train_data = pd.read_csv(trainfile, delimiter='\t', header=None)
        train_X, train_y = self.prepareInputData(df_train_data, training=True)
        
        
        estimators =  [('rf', RandomForestClassifier(n_estimators = 1500, max_depth=40, random_state=1)), 
                       ('PAC', PassiveAggressiveClassifier(C = 0.01, max_iter=2500)),
                       ('SVC',LinearSVC(random_state=0, C=0.05, multi_class="crammer_singer", max_iter=3000))]
        
        model = VotingClassifier(estimators=estimators, voting='hard')
        
        model.fit(train_X, train_y)
        
#         save model to disk
        pickle.dump(model, open('../resources/'+self.model_filename, 'wb'))
        return train_X, train_y
    
    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        df_test_data = pd.read_csv(datafile, delimiter='\t', header=None)
        test_X, test_y = self.prepareInputData(df_test_data, training=False)
        model = pickle.load(open('../resources/'+self.model_filename, 'rb'))
        
        try:
            y_pred = model.predict(test_X)
        except:
            return -1*np.ones(y_pred.shape)
        y_pred = self.inverseLabelEncode(y_pred)
        return y_pred




