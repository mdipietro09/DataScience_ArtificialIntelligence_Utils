
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import langdetect 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn import feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline
from tensorflow.keras import models, layers, preprocessing
import wordcloud
import gensim
import spacy
import difflib
import collections
import random



###############################################################################
#                  TEXT ANALYSIS                                              #
###############################################################################
'''
'''
def utils_plot_distributions(dtf, x, top=None, y=None, bins=None, figsize=(10,5)):
    ## univariate
    if y is None:
        if top is None:
            dtf[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", figsize=figsize, legend=False, grid=True)
        else:
            dtf[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", figsize=figsize, legend=False, grid=True)
    ## bivariate
    else:
        bins = dtf[x].nunique() if bins is None else bins
        for i in dtf[y].unique():
            dtf[dtf[y]==i][x].hist(alpha=0.8, figsize=figsize, bins=bins)
        plt.legend(dtf[y].unique())
    plt.show()



'''
'''
def add_detect_lang(dtf, column):
    dtf[column+'_lang'] = dtf[column].apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")    
    print(dtf[[column+'_lang']].describe().T)
    return dtf



'''
Computes the count of words and the count of characters.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    dtf: input dataframe with 2 new columns
'''
def add_text_count(dtf, column):
    dtf[column+'_word_count'] = dtf[column].apply(lambda x: len(str(x).split(" ")))
    dtf[column+'_text_length'] = dtf[column].str.len()
    print(dtf[[column+'_word_count',column+'_text_length']].describe().T)
    return dtf



'''
Computes the words frequencies.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param top: num - plot the top frequent words
    :param figsize: tupla - pyplot figsize
:return
    dtf_count: dtf with word frequency
'''
def words_freq(dtf, column, top=30, figsize=(10,10)):
    huge_str = dtf[column].str.cat(sep=" ")
    lst_tokens = nltk.tokenize.word_tokenize(huge_str)
    dic_words_freq = nltk.FreqDist(lst_tokens)
    dtf_count = pd.DataFrame(dic_words_freq.most_common(), columns=["Word", "Freq"])
    dtf_count.set_index("Word").iloc[:top, :].sort_values(by="Freq").plot(kind="barh", title="Most frequent words", figsize=figsize)
    plt.show()
    return dtf_count



'''
Creates a list of stopwords.
:parameter
    :param lst_langs: list - ["english", "italian"]
    :param lst_new_words: list - list of new stop words to add
:return
    stop_words: list of stop words
'''      
def create_stopwords(lst_langs=["english"], lst_new_words=[]):
    stop_words = set()
    for lang in lst_langs:
        words_nltk = set(nltk.corpus.stopwords.words(lang))
        stop_words = stop_words.union(words_nltk)
    stop_words = stop_words.union(lst_new_words)
    return set(stop_words)
        


'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_regex: list - list of regex to remove
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text, lst_regex=None, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## regex
    if lst_regex is not None: 
        for regex in lst_regex:
            text = re.sub(regex, '', text)
    
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text



'''
Adds a column of preprocessed text.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    dtf: input dataframe with two new columns
'''
def add_preprocessed_text(dtf, column, lst_regex=None, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    dtf = dtf[ pd.notnull(dtf[column]) ]
    dtf[column+"_clean"] = dtf[column].apply(lambda x: utils_preprocess_text(x, lst_regex, flg_stemm, flg_lemm, lst_stopwords))
    return dtf



###############################################################################
#                      SENTIMENT                                              #
###############################################################################
'''
Computes the sentiment using Textblob.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param algo: string - 
:return
    dtf: input dataframe with new sentiment column
'''
def add_sentiment(dtf, column, algo="nltk"):
    if algo=="nltk":
        nltk_sentim = SentimentIntensityAnalyzer()
        dtf[column+"_sentiment"] = dtf[column].apply(lambda x: nltk_sentim.polarity_scores(x)["compound"])
    elif algo=="textblob":
        dtf[column+"_sentiment"] = dtf[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    print(dtf[[column+'_sentiment']].describe().T)
    return dtf



'''
Creates a pivot table of sentiment sources breakdown.
:parameter
    :param dtf: dataframe - dtf with a sentiment column
    :param index: string or list - name of column to use as rows
    :param columns: string or list - name of column to use as columns
    :param aggfunc: string or list - functions to apply to summarize results
:return
    dtf_sentim: pivot table as dataframe
'''
def pivot_sentiment(dtf, index, columns, aggfunc="sum"):
    try:
        dtf_sentim = pd.pivot_table(dtf, values="sentiment", index=index, columns=columns, aggfunc=aggfunc, fill_value=0)
        dtf_sentim["sentiment"] = round(dtf_sentim.sum(axis=1), ndigits=2)
        dtf_sentim = dtf_sentim.sort_values("sentiment", ascending=False)
        return dtf_sentim

    except Exception as e:
        print("--- got error ---")
        print(e)
        

        
###############################################################################
#                            NER                                              #
###############################################################################
'''
Counts the elements in a list.
:parameter
    :param lst: list
    :param top: num - number of top elements to return
:return
    lst_top - list with top elements
'''
def utils_lst_count(lst, top=None):
    dic_counter = collections.Counter()
    for x in lst:
        dic_counter[x] += 1
    dic_counter = collections.OrderedDict(sorted(dic_counter.items(), key=lambda x: x[1], reverse=True))
    lst_top = [ {key:value} for key,value in dic_counter.items() ]
    if top is not None:
        lst_top = lst_top[:top]
    return lst_top



'''
Applies the spacy NER model.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param model: string - "en_core_web_lg", "en_core_web_sm", "xx_ent_wiki_sm"
    :param tag_type: string or list - "all" or ["ORG","PERSON","NORP","GPE", "EVENT", ...]
    :param top: num - plot the top frequent words
    :param figsize: tupla - pyplot figsize
:return
    {"dtf":dtf, "dtf_tags":dtf_tags}
'''
def add_ner_spacy(dtf, column, model="en_core_web_lg", tag_type="all", top=20, figsize=(10,10)):
    try:
        ## load model
        ner_model = spacy.load(model)

        ## apply model and add a column to the input dtf
        if tag_type == "all":
            #dtf["tags"] = dtf[column].apply(lambda x: list(set([(word.text, word.label_) for word in ner_model(x).ents])) )
            dtf["tags"] = dtf[column].apply(lambda x: [(word.text, word.label_) for word in ner_model(x).ents] )
        else:
            #dtf["tags"] = dtf[column].apply(lambda x: list(set([(word.text, word.label_) for word in ner_model(x).ents if word.label_ in tag_type])) )
            dtf["tags"] = dtf[column].apply(lambda x: [(word.text, word.label_) for word in ner_model(x).ents if word.label_ in tag_type] )
        
        dtf["tags"] = dtf["tags"].apply(lambda x: utils_lst_count(x, top=None))

        ## compute overall frequency
        tags_list = dtf["tags"].sum()
        map_lst = list(map(lambda x: list(x.keys())[0], tags_list))
        dtf_tags = pd.DataFrame(map_lst, columns=['tag','type'])
        dtf_tags["count"] = 1
        dtf_tags = dtf_tags.groupby(['type','tag']).count().reset_index() 
        dtf_tags = dtf_tags.sort_values("count", ascending=False)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.barplot(x="count", y="tag", hue="type", data=dtf_tags.iloc[:top,:], ax=ax)
        
        return {"dtf":dtf, "dtf_tags":dtf_tags}

    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Display the spacy NER model.
:parameter
    :param txt: string - text input for the model.
    :param model: string - "en_core_web_lg", "en_core_web_sm", "xx_ent_wiki_sm"
    :param lst_tags: list or None - example ["ORG", "GPE", "LOC"], None for all tags
    :param title: str or None
'''
def ner_displacy(txt, model="en_core_web_lg", lst_tags=None, title=None):
    ner_model = spacy.load(model)
    doc = ner_model(txt)
    doc.user_data["title"] = title
    spacy.displacy.serve( doc, style="ent", options={"ents":lst_tags} )
        
        
        
'''
Retrain spacy model with new tags.
:parameter
    :param train_data: list [
            ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
            ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}), 
        ]
    :param output_dir: string - path of directory to save model
    :param model: string - "blanck" or "en_core_web_lg", ...
    :param n_iter: num - number of iteration
'''
def retrain_spacy(train_data, output_dir, model="blank", n_iter=100):
    try:
        ## prepare data
#        train_data = []
#        for name in lst:
#            frase = "ciao la mia azienda si chiama "+name+" e fa business"
#            tupla = (frase, {"entities":[(30, 30+len(name), tag_type)]})
#            train_data.append(tupla)
        
        ## load model
        if model == "blank":
            ner_model = spacy.blank("en")
        else:
            ner_model = spacy.load(model)
        
        ## create a new pipe
        if "ner" not in ner_model.pipe_names:
            new_pipe = ner_model.create_pipe("ner")
            ner_model.add_pipe(new_pipe, last=True)
        else:
            new_pipe = ner_model.get_pipe("ner")
        
        ## add label
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                new_pipe.add_label(ent[2])
            
        ## train
        other_pipes = [pipe for pipe in ner_model.pipe_names if pipe != "ner"] ###ignora altre pipe
        with ner_model.disable_pipes(*other_pipes):
            print("--- Training spacy ---")
            if model == "blank":
                ner_model.begin_training()
            for n in range(n_iter):
                random.shuffle(train_data)
                losses = {}
                batches = spacy.util.minibatch(train_data, size=spacy.util.compounding(4., 32., 1.001)) ###batch up data using spaCy's minibatch
                for batch in batches:
                    texts, annotations = zip(*batch)
                    ner_model.update(docs=texts, golds=annotations, drop=0.5, losses=losses)  ###update
        
        ## test the trained model
        print("--- Test new model ---")
        for text, _ in train_data:
            doc = ner_model(text)
            print([(ent.text, ent.label_) for ent in doc.ents])

        ## save model to output directory
        ner_model.to_disk(output_dir)
        print("Saved model to", output_dir)

    except Exception as e:
        print("--- got error ---")
        print(e)        
        
           

###############################################################################
#                     BAG OF WORDS (VECTORIZER)                               #
###############################################################################
'''
'''
def bow_build_vocabolary(corpus, vectorizer=None):
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,2)) if vectorizer is None else vectorizer
    vectorizer.fit(corpus)
    dic_vocabolary = {word:idx for idx, word in enumerate(vectorizer.get_feature_names())}
    return dic_vocabolary



'''
'''
def words_correlation(dtf, x, y, max_ngrams=2, top=2):
    tfidf = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,max_ngrams))
    X = tfidf.fit_transform(dtf[x])
    dic_cat = {}
    for cat in dtf[y].unique():
        features_chi2 = feature_selection.chi2(X, dtf[y] == cat)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        print("# {}:".format(y))
        lst_grams = []
        for n in range(1, max_ngrams+1):
            grams = [v for v in feature_names if len(v.split(' ')) == n]
            print("  . {}".format('\n  . '.join(grams[-top:])))
            print(" ")
            lst_grams.append(grams)
        dic_cat.update({cat:lst_grams})
    return dic_cat














'''
Vectorizes the corpus (tfidf), fits the Bag of words model, plots the most frequent words.
:parameter
    :param lst_corpus: list - list with corpus
    :param ngrams: num - number of ngrams
    :param max_features: num or None -
    :param max_df: num -
    :param plot: bool - 
    :param top: num - plot the top frequent words
    :param figsize: tupla - pyplot figsize
:return
    dict with X_BoW, lst_vocabulary, dtf_vocabulary
'''
def BoW_tfidf(lst_corpus, ngrams=1, max_features=None, max_df=0.8, plot=True, top=20, figsize=(10,10)):
    try:
        ## fit Bag of Words (document terms matrix)
        tfidf = feature_extraction.text.TfidfVectorizer(max_df=max_df, max_features=max_features, ngram_range=(1,ngrams))
        vec = tfidf.fit(lst_corpus)
        X = vec.transform(lst_corpus)
        
        ## put into dtf
        lst_vocabulary = [(word, X.sum(axis=0)[0, idx]) for word, idx in vec.vocabulary_.items()]
        lst_vocabulary = sorted(lst_vocabulary, key=lambda x: x[1], reverse=True)
        dtf_vocabulary = pd.DataFrame(lst_vocabulary)
        dtf_vocabulary.columns = ["Word", "Freq"]
        dtf_vocabulary = dtf_vocabulary.set_index("Word")
        print(dtf_vocabulary.index)
        
        ## plot
        if plot == True:
            dtf_vocabulary.iloc[:top,:].sort_values(by="Freq").plot(kind="barh", title="Most frequent "+str(ngrams)+"-grams", figsize=figsize)
            plt.show()
        
        return {"X":X, "lst_vocabulary":lst_vocabulary, "dtf_vocabulary":dtf_vocabulary}
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        


'''
Vectorizes the corpus using a given vocabulary to find keywords into corpus with sklearn.
:parameter
    :param lst_corpus: list - list with corpus
    :param lst_vocabulary: list - vocabulary for CountVectorizer
    :param index: string "auto" or pandas Series
:return
    dtf_matrix - the sparse matrix as dataframe
'''
def BoW_search(lst_corpus, lst_vocabulary, index="auto"):
    try:
        ## One-Hot Encoding
        model = feature_extraction.text.TfidfVectorizer(vocabulary=lst_vocabulary, lowercase=True, binary=True)

        ## fit Bag of Words (document terms matrix)
        X_BoW = model.fit_transform(lst_corpus)

        ## put into dtf
        dtf_matrix = pd.DataFrame(X_BoW.todense(), columns=lst_vocabulary)
        if index == "auto":
            dtf_matrix = dtf_matrix.set_index( pd.Series(lst_corpus) )
            dtf_matrix = dtf_matrix.reset_index()
        elif type(index) == pd.core.series.Series:
            dtf_matrix = dtf_matrix.set_index( index )
            dtf_matrix = dtf_matrix.reset_index()
        elif type(index) == list:
            dtf_matrix = dtf_matrix.set_index( pd.Series(index) )
            dtf_matrix = dtf_matrix.reset_index()

        return dtf_matrix

    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Plots a wordcloud from a list of Docs or from a dictionary
:parameter
    :param lst_corpus: list or None
    :param dic_words: dict or None {"word1":freq1, "word2":freq2, ...}
    :param figsize: tuple
'''
def plot_wordcloud(lst_corpus=None, dic_words=None, figsize=(10,10)):
    if lst_corpus is not None:
         wc = wordcloud.WordCloud(background_color='black', max_words=150, max_font_size=35, random_state=42)
         wc = wc.generate(str(lst_corpus))
         fig = plt.figure(num=1, figsize=figsize)
         plt.axis('off')
         plt.imshow(wc, cmap=None)
    elif dic_words is not None:
        wc = wordcloud.WordCloud(max_font_size=35)
        wc = wc.generate_from_frequencies(dic_words)
        fig = plt.figure(num=1, figsize=figsize)
        plt.axis('off')
        plt.imshow(wc, cmap=None)
    else:
        print("--- choose: lst_corpus or dic_words")

    

###############################################################################
#                            WORD2VEC (EMBEDDINGS)                            #
###############################################################################
'''
Load a Word2Vec for spacy or gensim
'''
def load_w2v(library="spacy", model="en_core_web_lg", path=""):
    if library=="spacy":
        nlp = spacy.load(path+model)
    elif library=="gensim":
        nlp = gensim.models.KeyedVectors.load_word2vec_format(path+model, binary=True) #GoogleWord2Vec.bin.gz
    return nlp



'''
Tranforms the corpus into a list of list of words.
:parameter
    :param corpus: list or array
    :param ngrams: num - number of ngrams
:return
    list of lists of ngrams
'''
def utils_text_preprocessing_lstwords(corpus, ngrams=1):
    lst_corpus = []
    for string in lst_corpus:
        lst_words = string.split()
        lst_grams = [' '.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)
    return lst_corpus
 















    
'''
Fits the Word2Vec model from gensim.
:parameter
    :param corpus: list - list of lists of words
    :param min_count: num  -
    :param size: num - 
    :param window: num - 
    :param sg: num - 0 for CBOW, 1 for skipgrams
:return
    [model, dtf_vocabulary]: first item of list is the Word2Vec model, second is the dtf of the vocabulary
'''
def fit_w2v(corpus, ngrams=1, min_count=20, size=100, window=20, sg=0, plot="2d", figsize=(10,10)):
    try:
        ## preprocessing
        lst_corpus = utils_text_preprocessing_lstwords(corpus, ngrams=1)
        ## model
        print("--- training ---")
        model = gensim.models.word2vec.Word2Vec(lst_corpus, size=size, window=window, min_count=min_count, workers=4, sg=sg)
        print("--- model fitted ---")
        lst_vocabulary = list(model.vocab.keys())
        dtf_vocabulary = pd.DataFrame(lst_vocabulary)
        
        ## plot
        if plot == "2d":
            labels = []
            X = []
            for word in model.vocab:
                X.append( model[word] )
                labels.append( word )
            tsne_model = manifold.TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
            new_values = tsne_model.fit_transform( X )
            x = []
            y = []
            for value in new_values:
                x.append(value[0])
                y.append(value[1])
            plt.figure(figsize=figsize) 
            for i in range(len(x)):
                plt.scatter(x[i], y[i])
                plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
            plt.show()
        
        if plot == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            X = model[lst_vocabulary]
            tsne_model = manifold.TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)
            new_values = tsne_model.fit_transform(X)
            dtf_tmp = pd.DataFrame(new_values, index=lst_vocabulary, columns=['x','y','z'])
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dtf_tmp['x'], dtf_tmp['y'], dtf_tmp['z'], alpha=0.5)
            for word, row in dtf_tmp.iterrows():
                x, y, z = row
                ax.text(x, y, z, s=word, size=10, zorder=1)
            plt.show()
            
        return [model, dtf_vocabulary]
    
    except Exception as e:
        print("--- got error ---")
        print(e)
   


def plot_w2v(type="2d"):
    return 0
        
        
        
    
    


'''
Prints the closest words to the input word according to Word2Vec model and plot them in a 2d or 3d space.
:parameter
    :param model: model - Word2Vec
    :param str_word: string - input word
    :param top: num - top words
    :param plot: string or None - "2d" or "3d"
    :param figsize: tupla - pyplot figsize
'''
def predict_Word2Vec(model, str_word, top=20, plot="2d", figsize=(20,13)):
    try:
        ## predict (embedda)
        word_vec = model[str_word]
        
        ## get context
        lst_context = model.most_similar( str_word, topn=top )
        
        ## plot
        if plot == "2d":
            labels = []
            X = []
            for tupla in lst_context:
                word = tupla[0]
                X.append( model[word] )
                labels.append( word )
            tsne_model = manifold.TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
            new_values = tsne_model.fit_transform( X )
            x = []
            y = []
            for value in new_values:
                x.append(value[0])
                y.append(value[1])
            fig = plt.figure(figsize=figsize)
            fig.suptitle("word: "+str_word, fontsize= 20)
            for i in range(len(x)):
                plt.scatter(x[i],y[i])
                plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
            plt.show()
        
        if plot == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            lst = []
            for tupla in lst_context:
                lst.append(tupla[0])
            X = model[lst]
            tsne_model = manifold.TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)
            new_values = tsne_model.fit_transform(X)
            dtf_tmp = pd.DataFrame(new_values, index=lst, columns=['x','y','z'])
            fig = plt.figure(figsize=figsize)
            fig.suptitle("word: "+str_word, fontsize= 20)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dtf_tmp['x'], dtf_tmp['y'], dtf_tmp['z'], alpha=0.5)
            for word, row in dtf_tmp.iterrows():
                x, y, z = row
                ax.text(x, y, z, s=word, size=10, zorder=1)
            plt.show()

        return {"word_vec":word_vec, "lst_context":lst_context}
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        print("maybe you are looking for ... ")
        lst_match = [word for word in list(model.vocab.keys())
                      if difflib.SequenceMatcher(isjunk=None, a=str_word, b=word).ratio()>0.7]
        print(lst_match)



'''
'''
def compute_distances_Word2Vec(model, lst_words, top=20, plot="2d", figsize=(20,13)):
    ## embedda
    lst_dics = []
    for str_word in lst_words:
        dic = predict_Word2Vec(model, str_word, top=top, plot=False)
        lst_dics.append({"str_word":str_word, "word_vec":dic["word_vec"], "lst_context":dic["lst_context"]})
        
    ## put in a dtf
    dtf = pd.DataFrame([ {'str_word':dic["str_word"], 'word_vec':dic["word_vec"], "lst_context":dic["lst_context"]} for dic in lst_dics ])
    
    return dtf



'''
Clusters a Word2Vec vocabolary with nltk Kmeans.
:parameter
    :param lst_lst_corpus: lst of lists (if given, trains a new Word2Vec)
    :param modelW2V: word2vec model (if given uses the model already trained)
    :param k: num - clusters
    :param repeats: num - kmeans loop
:return
    dtf with words and clusters
'''
def clustering_words_Word2Vec(lst_lst_corpus, modelW2V, k=3, repeats=50):
    try:
        if (lst_lst_corpus is not None) and (modelW2V is None):
            model = gensim.models.word2vec.Word2Vec(lst_lst_corpus, min_count=1)
        elif (lst_lst_corpus is None) and (modelW2V is not None):
            model = modelW2V
        else:
            return("--- pick one: model or corpus ---")
            
        X = model[model.vocab.keys()]
        lst_vocabulary = list(model.vocab.keys())
        kmeans_model = nltk.cluster.KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=50)
        assigned_clusters = kmeans_model.cluster(X, assign_clusters=True)
        dtf_cluster = pd.DataFrame({"word":word, "cluster":str(assigned_clusters[i])}  
                                        for i,word in enumerate(lst_vocabulary))
        dtf_cluster = dtf_cluster.sort_values(["cluster", "word"], ascending=[True,True])
        print("--- ok done ---")
        return dtf_cluster
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        
 
    
###############################################################################
#                     TEXT CLASSIFICATION                                     #
###############################################################################
'''
'''
def encode_variable(dtf, column):
    dtf[column+"_id"] = dtf[column].factorize(sort=True)[0]
    dic_class_mapping = dict( dtf[[column+"_id",column]].drop_duplicates().sort_values(column+"_id").values )
    return dtf, dic_class_mapping



'''
'''
def preprocess_text(X, y, X_extra):
    return 0






'''
'''
def features_selection():
    model = smf.ols(num+' ~ '+cat, data=dtf).fit()
    table = sm.stats.anova_lm(model)
    p = table["PR(>F)"][0]
    coeff, p = None, round(p, 3)
    conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
    print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")
    


'''
Fits a sklearn classification model.
:parameter
    :param classifier: model object - model to fit (before fitting)
    :param X_train: array of text
    :param y_train: array of classes
    :param X_test: array of text
    :param y_test: array of classes
    :param vectorizer: vectorizer object - if None Tfidf is used
    :param classifier: model object - if None MultinomialNB is used
:return
    fitted model and predictions
'''
def ml_text_classif(X_train, y_train, X_test, y_test, vectorizer=None, classifier=None): 
    ## preprocessing
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,2)) if vectorizer is None else vectorizer
    
    ## model
    classifier = naive_bayes.MultinomialNB() if classifier is None else classifier

    ## pipeline
    model = pipeline.Pipeline([
        ("vectorizer", vectorizer),
        ("model", classifier) ])
    
    ## train/test
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)

    ## check
    print("True:", y_test[0])
    print("Pred:", predicted[0], np.max(predicted_prob[0]))
    
    ## kpi
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("Accuracy (overall correct predictions):",  round(accuracy,3))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))
    return {"model":model, "predicted_prob":predicted_prob, "predicted":predicted}



'''
'''
def utils_text2seq(corpus, dic_vocabolary, vectorizer=None):
    ## objects
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,1)) if vectorizer is None else vectorizer
    tokenizer = vectorizer.build_tokenizer()
    preprocessor = vectorizer.build_preprocessor()
    
    ## from text to tokens
    X = []
    for text in corpus:
        lst_words = tokenizer(preprocessor(text))
        lst_idx = [dic_vocabolary[word] for word in lst_words if word in dic_vocabolary]
        X.append(lst_idx)
        
    print("from: ", X[0], " | len:", len(X[0].split()))
    print("to: ", X[0], " | len:", len(X[0]))
    print("check: ", X[0].split()[0], " -- idx in vocabolary -->", dic_vocabolary[X[0].split()[0]])
    
    ## from tokens to sequences of same length
    maxlen = np.max([len(text.split()) for text in corpus])
    X = preprocessing.sequence.pad_sequences(X, maxlen=maxlen, value=len(vectorizer.get_feature_names()))
    print("shape:", X.shape)
    return X



'''
Embeds a vocabolary of unigrams with spacy w2v
'''
def vocabolary_embeddings(nlp, dic_vocabolary, dim_space=300):
    embeddings = np.zeros(len(dic_vocabolary), dim_space)
    for word,idx in dic_vocabolary.items():
        try:
            embeddings[idx] =  nlp.vocab[word].vector
        except:
            pass
    print("shape: ", embeddings.shape)
    return embeddings



'''
Fits a keras classification model.
:parameter
    :param dic_y_mapping: dict - {0:"A", 1:"B", 2:"C"}
    :param model: model object - model to fit (before fitting)
    :param embeddings: array of embeddings
    :param X_train: array of sequence
    :param y_train: array of classes
    :param X_test: array of sequence
    :param y_test: array of classes
:return
    model fitted and predictions
'''
def dl_text_classif(dic_y_mapping, model, embeddings, X_train, y_train, X_test, y_test, epochs=10, batch_size=256):
    ## model
    if model is None:
        ### params
        n_features = embeddings.shape[0]
        embeddings_dim = embeddings.shape[1]
        max_seq_lenght = X_train.shape[1]
        ### neural network
        model = models.Sequential()
        model.add( layers.Embedding(input_dim=n_features, 
                                    output_dim=embeddings_dim, 
                                    weights=[embeddings],
                                    input_length=max_seq_lenght, 
                                    trainable=False) )
        model.add( layers.LSTM(units=max_seq_lenght, dropout=0.2) )
        model.add( layers.Dense(len(np.unique(y_train)), activation='softmax') )
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    ## train
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0, validation_split=0.3)
    fig, ax = plt.subplots()
    ax.plot(training.history['loss'], label='loss')
    ax.grid(True)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    print(training.model.summary())
    
    ## test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(predicted_prob)] for pred in predicted_prob]
    y_test = [dic_y_mapping[y] for y in y_test]
    
    ## check
    print("True:", y_test[0])
    print("Pred:", predicted[0], np.max(predicted_prob[0]))
    
    ## kpi
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("Accuracy (overall correct predictions):",  round(accuracy,3))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))
    return {"model":training.model, "predicted_prob":predicted_prob, "predicted":predicted}



'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
    :param predicted_prob: array
    :param figsize: tuple - plot setting
'''
def evaluate_text_classif(y_test, predicted, predicted_prob, figsize=(20,10)):
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values
         
    ## confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes)
    plt.yticks(rotation=0)
    plt.title("Confusion matrix")
    plt.show()
    
    ## roc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=figsize)
    for i in range(len(classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test_array[:,i], predicted_prob[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=3, label='ROC curve of {0} (area={1:0.2f})'''.format(classes[i], roc_auc[i]))
    plt.plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    ## precision-recall curve
    precision = dict()
    recall = dict()
    plt.figure(figsize=figsize)
    for i in range(len(classes)):
        precision[i], recall[i], thresholds = metrics.precision_recall_curve(y_test_array[:,i], predicted_prob[:,i])
        plt.plot(recall[i], precision[i], lw=3, label='{}'.format(classes[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    


###############################################################################
#                  TOPIC MODELING                                             #
###############################################################################
'''
Fits Latent Dirichlet Allocation with gensim.
:parameter
    :param lst_lst_corpus: list - 
    :param n_topics: num -
    :param n_words: num -
    :param plot: logic -
:return
    dic with model and dtf topics
'''
def fit_lda(lst_lst_corpus, n_topics=10, n_words=5, plot=True, figsize=(10,10)):
    ## train the lda
    id2word = gensim.corpora.Dictionary(lst_lst_corpus) #map words with an id
    corpus = [id2word.doc2bow(word) for word in lst_lst_corpus]  #create dictionary Word:Freq
    print("--- training ---")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=n_topics, 
                                                random_state=123, update_every=1, chunksize=100, 
                                                passes=10, alpha='auto', per_word_topics=True)
    print("--- model fitted ---")
    
    ## output
    lst_dics = []
    for i in range(0, n_topics):
        lst_tuples = lda_model.get_topic_terms(i)
        for tupla in lst_tuples:
            word_id = tupla[0]
            word = id2word[word_id]
            weight = tupla[1]
            lst_dics.append({"topic":i, "id":word_id, "word":word, "weight":weight})
    dtf_topics = pd.DataFrame(lst_dics, columns=['topic','id','word','weight'])
    
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(y="word", x="weight", hue="topic", data=dtf_topics, ax=ax).set_title('Main Topics')
    ax.set(ylabel="", xlabel="Word Importance")
    
    return {"model":lda_model, "dtf_topics":dtf_topics}



'''
'''
def predict_lda(txt, lda_model):
    return topic          



###############################################################################
#                 STRINGS MATCHING                                            #
###############################################################################
'''
Computes the similarity of two strings with textdistance.
:parameter
    :param a: string
    :param b: string
    :param algo: string - "cosine", "gestalt", "jaccard"
:return
    similarity score
'''
def utils_strings_similarity(a, b, algo="cosine"):
    a = re.sub(r'[^\w\s]', '', str(a).lower().strip())
    b = re.sub(r'[^\w\s]', '', str(b).lower().strip())
    
    if algo == "cosine":
        lst_txt = [a, b]
        vectorizer = feature_extraction.text.CountVectorizer(lst_txt)
        matrix = vectorizer.fit_transform(lst_txt).toarray()
        lst_vectors = [vec for vec in matrix]
        cosine_matrix = metrics.pairwise.cosine_similarity(lst_vectors)
        cosine_sim_diag = cosine_matrix[0,1]
        return cosine_sim_diag
    
    elif algo == "gestalt": 
        return difflib.SequenceMatcher(isjunk=None, a=a, b=b).ratio()
    
    elif algo == "jaccard":
        return 1 - nltk.jaccard_distance(set(a), set(b))
    
    else:
        print('Choose one algo: "cosine", "gestalt", "jaccard"')
    


'''
Computes the similarity of two strings with textdistance.
:parameter
    :param str_name: string - str to lookup
    :param lst_strings: list - lst with possible matches
    :param algo: string - "cosine", "gestalt", "jaccard"
    :param threshold: num - similarity threshold to consider the match valid
    :param top: num or None - number of matches to return
:return
    dtf_matches - dataframe with matches
'''
def match_strings(stringa, lst_strings, algo="cosine", threshold=0.7, top=1):
    ## compute similarity
    dtf_matches = pd.DataFrame([{"stringa":stringa, "match":str_match,
                                 algo+"_similarity": utils_strings_similarity(stringa, str_match, algo=algo)}
                                 for str_match in lst_strings])
    ## put in a dtf
    dtf_matches = dtf_matches[ dtf_matches[algo+"_similarity"]>=threshold ]
    dtf_matches = dtf_matches[["stringa", "match", algo+"_similarity"]].sort_values(algo+"_similarity", ascending=False)
    if top is not None:
        dtf_matches = dtf_matches.iloc[0:top,:]
    if len(dtf_matches) == 0:
        dtf_matches = pd.DataFrame([[stringa,"None",0]], columns=['stringa','match',algo+"_similarity"])
    return dtf_matches



'''
Vlookup for similar strings.
:parameter
    :param strings_array - array or lst
    :param lookup_array - array or lst
    :param algo: string - "cosine", "gestalt", "jaccard"
    :param threshold: num - similarity threshold to consider the match valid
:return
    dtf_matches - dataframe with matches
'''
def vlookup(lst_left, lst_right, algo="cosine", threshold=0.7, top=1):
    try:
        dtf_matches = pd.DataFrame(columns=['stringa', 'match', algo+"_similarity"])
        for string in lst_left:
            dtf_match = match_strings(string, lst_right, algo=algo, threshold=threshold, top=top)
            for i in range(len(dtf_match)):
                print(string, " --", dtf_match.iloc[i,2], "--> ", dtf_match["match"].values[i])
            dtf_matches = dtf_matches.append(dtf_match)
        dtf_matches = dtf_matches.reset_index(drop=True)
        return dtf_matches

    except Exception as e:
        print("--- got error ---")
        print(e)