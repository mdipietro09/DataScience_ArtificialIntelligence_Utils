
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn import feature_extraction, metrics, manifold
import wordcloud
import gensim
import spacy
import requests
import json
import difflib
import collections
from bs4 import BeautifulSoup
import random



###############################################################################
#                  TEXT ANALYSIS                                              #
###############################################################################
'''
Counts the elements in a list.
:parameter
    :param lst: list
    :param top: num - number of top elements to return
:return
    lst_top - list with top elements
'''
def lst_count(lst, top=None):
    try:
        dic_counter = collections.Counter()
        for x in lst:
            dic_counter[x] += 1
        dic_counter = collections.OrderedDict(sorted(dic_counter.items(), key=lambda x: x[1], reverse=True))
        lst_top = [ {key:value} for key,value in dic_counter.items() ]
        if top is not None:
            lst_top = lst_top[:top]
        return lst_top

    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Computes the count of words and the count of characters.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    dtf: input dataframe with two new columns
'''
def text_count(dtf, column):
    try:
        dtf[column+'_word_count'] = dtf[column].apply(lambda x: len(str(x).split(" ")))
        dtf[column+'_char_count'] = dtf[column].str.len()
        print("--- ok done ---")
        print("min:" + str(round(dtf[column+'_word_count'].min(), 2)) + " median:" + str(round(dtf[column+'_word_count'].median(), 2)) +
              " mean:" + str(round(dtf[column+'_word_count'].mean(), 2)) + " max:" + str(round(dtf[column+'_word_count'].max(), 2)))
        return dtf

    except Exception as e:
        print("--- got error ---")
        print(e)



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
    try:
        ## produce dtf
        huge_str = dtf[column].str.cat(sep=" ")
        lst_tokens = nltk.tokenize.word_tokenize(huge_str)
        dic_words_freq = nltk.FreqDist(lst_tokens)
        dtf_count = pd.DataFrame(dic_words_freq.most_common(), columns=["Word", "Freq"])

        ## plot
        dtf_count.set_index("Word").iloc[:top, :].sort_values(by="Freq").plot(kind="barh", title="Most frequent words", figsize=figsize)
        plt.show()
        print("--- ok done ---")
        return dtf_count

    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Adds a column of clean text.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param lst_regex: list - list of regex
:return
    dtf: input dataframe with two new columns
'''
def clean_text_column(dtf, column, lst_regex=None):
    try:
        dtf[column+"_clean"] = dtf[column].apply(lambda x: str(x).lower()) ## lower

        if lst_regex is not None: ## remove regex
            for regex in lst_regex:
                dtf[column+"_clean"] = dtf[column+"_clean"].apply(lambda x: re.sub(regex, '', x))

        dtf[column+"_clean"] = dtf[column+"_clean"].apply(lambda x: re.sub(r'[^\w\s]', '', x)) ## remove punctuations and special characters
        dtf[column+"_clean"] = dtf[column+"_clean"].apply(lambda x: x.strip()) ## strip
        print("--- ok done ---")
        return dtf
    
    except Exception as e:
        print("--- got error ---")
        print(e)



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
    try:
        if algo=="nltk":
            nltk_sentim = SentimentIntensityAnalyzer()
            dtf[column+"_sentiment"] = dtf[column].apply(lambda x: nltk_sentim.polarity_scores(x)["compound"])
        elif algo=="textblob":
            dtf[column+"_sentiment"] = dtf[column].apply(lambda x: TextBlob(x).sentiment.polarity)
        print("--- ok done ---")
        print("min:" + str(round(dtf[column+"_sentiment"].min(), 2)) + " median:" + str(round(dtf[column+"_sentiment"].median(), 2)) +
              " mean:" + str(round(dtf[column+"_sentiment"].mean(), 2)) + " max:" + str(round(dtf[column+"_sentiment"].max(), 2)))
        return dtf

    except Exception as e:
        print("--- got error ---")
        print(e)



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
def ner_spacy(dtf, column, model="en_core_web_lg", tag_type="all", top=20, figsize=(10,10)):
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
        dtf["tags"] = dtf["tags"].apply(lambda x: lst_count(x, top=None))
        print("--- added tags column ---")

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
    try:
        ner_model = spacy.load(model)
        doc = ner_model(txt)
        doc.user_data["title"] = title
        spacy.displacy.serve( doc, style="ent", options={"ents":lst_tags} )

    except Exception as e:
        print("--- got error ---")
        print(e)
        
        
        
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
        
        

'''
Applies the microsoft NER model.
:parameter
    :param lst_txt: list - text to tag
    :param api_key: string - microsoft api key
:return
    lst_tags - list of dics with name, tag, wikipedia, industry
'''
def ner_msft(lst_txt, api_key):
    try:
        ## prepare the json to post
        lst_dics_txt = []
        i = 1
        for txt in lst_txt:
            lst_dics_txt.append({"id":i, "text":txt})
            i = i +1

        ## post requests
        url = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.0/entities"
        dic_headers = {"Ocp-Apim-Subscription-Key": api_key}
        json_txt = {'documents': lst_dics_txt}
        res = requests.post(url, headers=dic_headers, json=json_txt)
        if res.status_code != 200:
            print(res.status_code)
            next
        else:
            lst_res = json.loads(res.content)["documents"]
            print(str(res.status_code) + ", " + str(len(lst_dics_txt)) + " strings tagged")

            ## put tags into new list
            lst_tags = []
            for dic in lst_res:
                if len(dic["entities"]) != 0:
                    word = dic["entities"][0]["matches"][0]["text"]
                    tag = dic["entities"][0]["name"]
                    wikipedia_url = dic["entities"][0]["wikipediaUrl"]

                    ######## Parse Wikipedia ##########
                    if len(wikipedia_url) != 0:
                        print(word+" - "+wikipedia_url)
                        res_wiki = requests.get(wikipedia_url)
                        soup = BeautifulSoup(res_wiki.content, "html.parser")
                        table_wiki = soup.find("table", {"class":"infobox vcard"})
                        industry = ""
                        if table_wiki is not None:
                            rows = table_wiki.find_all("tr")
                            for row in rows:
                                if "Industry" in str(row.find("th")):
                                    industry = row.text
                                    industry = industry.replace("Industry","")
                                    print(word+" - "+str(industry))
                    ###################################

                    dic_tag = {word: (tag, industry, wikipedia_url)}
                    lst_tags.append(dic_tag)
                else:
                    pass
            return lst_tags

    except Exception as e:
        print("--- got error ---")
        print(e)



###############################################################################
#                           BOW (ML)                                          #
###############################################################################
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
Creates a pre-processed corpus list from a dtf.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param stop_words: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    lst_corpus: list of documents
'''
def text_preprocessing_ListOfDocs(dtf, column, stop_words, flg_stemm=True, flg_lemm=False):
    try:
        ## remove null
        dtf = dtf[ pd.notnull(dtf[column]) ]
        
        ## create list for corpus
        lst_corpus = []
        for i in dtf.index:
            
            text = dtf[column][i]
            
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
            lst_text = [word for word in lst_text if word not in stop_words]
            
            ## riporto a stringa e append to corpus
            text = " ".join(lst_text)
            lst_corpus.append(text)
        
        print("--- ok done ---")
        return lst_corpus
        
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Vectorizes the corpus (tfidf), fits the Bag of words model, plots the most frequent words, plots a wordcloud.
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
def model_BoW(lst_corpus, ngrams=1, max_features=None, max_df=0.8, plot=True, top=20, figsize=(10,10)):
    try:
        ## fit Bag of Words (document terms matrix)
        model = feature_extraction.text.TfidfVectorizer(max_df=max_df, max_features=max_features, ngram_range=(1,ngrams))
        vec = model.fit(lst_corpus)
        X_BoW = vec.transform(lst_corpus)
        
        ## put into dtf
        lst_vocabulary = [(word, X_BoW.sum(axis=0)[0, idx]) for word, idx in vec.vocabulary_.items()]
        lst_vocabulary = sorted(lst_vocabulary, key=lambda x: x[1], reverse=True)
        dtf_vocabulary = pd.DataFrame(lst_vocabulary)
        dtf_vocabulary.columns = ["Word", "Freq"]
        dtf_vocabulary = dtf_vocabulary.set_index("Word")
        print(dtf_vocabulary.index)
        
        ## plot
        if plot == True:
            dtf_vocabulary.iloc[:top,:].sort_values(by="Freq").plot(kind="barh", title="Most frequent "+str(ngrams)+"-grams", figsize=figsize)
            plt.show()
        
        return {"X_BoW":X_BoW, "lst_vocabulary":lst_vocabulary, "dtf_vocabulary":dtf_vocabulary}
    
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

        print("--- ok done ---")
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
#                            W2V (DL)                                         #
###############################################################################
'''
Tranforms the corpus into a list of list of words, fits the Word2Vec model, plots vocabulary in a 2d or 3d space.
:parameter
    :param lst_corpus: list - list with corpus
    :param ngrams: num - number of ngrams
:return
    list of lists of ngrams
'''
def text_preprocessing_ListOfListsOfWords(lst_corpus, ngrams=1):
    lst_lst_corpus = []
    for string in lst_corpus:
        lst_words = string.split()
        lst_grams = [' '.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_lst_corpus.append(lst_grams)
    return lst_lst_corpus
 
    
    
'''
Tranforms the corpus into a list of list of words, fits the Word2Vec model, plots vocabulary in a 2d or 3d space.
:parameter
    :param lst_lst_corpus: list - list of lists of words
    :param min_count: num  -
    :param size: num - 
    :param window: num - 
    :param sg: num - 0 for CBOW, 1 for skipgrams
    :param plot: string or None - "2d" or "3d"
    :param figsize: tupla - pyplot figsize
:return
    [model, dtf_vocabulary]: first item of list is the Word2Vec model, second is the dtf of the vocabulary
'''
def fit_Word2Vec(lst_lst_corpus, min_count=20, size=100, window=20, sg=0, plot="2d", figsize=(10,10)):
    try:
        ## model
        print("--- training ---")
        model = gensim.models.word2vec.Word2Vec(lst_lst_corpus, size=size, window=window, min_count=min_count, workers=4, sg=sg)
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
   
     
        
'''
'''
def io_Word2Vec(modelpath, modelfile="GoogleWord2Vec.bin.gz", model=None):
    ## load
    if model is None:
        model = gensim.models.KeyedVectors.load_word2vec_format(modelpath+modelfile, binary=True)
        print("--- model loaded ---")
        return model
    ## save
    else:
        model.save(modelpath+modelfile)
        print("--- model saved ---")
    


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
def strings_similarity(a, b, algo="cosine"):
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
    try:
        ## compute similarity
        dtf_matches = pd.DataFrame([{"stringa":stringa, "match":str_match,
                                     algo+"_similarity": strings_similarity(stringa, str_match, algo=algo)}
                                     for str_match in lst_strings])
        ## put in a dtf
        dtf_matches = dtf_matches[ dtf_matches[algo+"_similarity"]>=threshold ]
        dtf_matches = dtf_matches[["stringa", "match", algo+"_similarity"]].sort_values(algo+"_similarity", ascending=False)
        if top is not None:
            dtf_matches = dtf_matches.iloc[0:top,:]
        if len(dtf_matches) == 0:
            dtf_matches = pd.DataFrame([[stringa,"None",0]], columns=['stringa','match',algo+"_similarity"])
        return dtf_matches

    except Exception as e:
        print("--- got error ---")
        print(e)



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
    