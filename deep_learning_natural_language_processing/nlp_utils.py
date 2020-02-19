
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import langdetect 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn import preprocessing, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline, decomposition
from tensorflow.keras import models, layers, preprocessing as kprocessing
import wordcloud
import gensim
import gensim.downloader as gensim_api
import spacy
import difflib
import collections
import random



###############################################################################
#                  TEXT ANALYSIS                                              #
###############################################################################
'''
Plot univariate and bivariate distributions.
'''
def utils_plot_distributions(dtf, x, top=None, y=None, bins=None, figsize=(10,5)):
    ## univariate
    if y is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(x, fontsize=12)
        if top is None:
            dtf[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
        else:   
            dtf[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", legend=False, ax=ax).grid(axis='x')
    ## bivariate
    else:
        bins = dtf[x].nunique() if bins is None else bins
        fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
        fig.suptitle(x, fontsize=12)
        for i in dtf[y].unique():
            sns.distplot(dtf[dtf[y]==i][x], hist=True, kde=False, bins=bins, hist_kws={"alpha":0.8}, axlabel="histogram", ax=ax[0])
            sns.distplot(dtf[dtf[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="density", ax=ax[1])
        ax[0].grid(True)
        ax[0].legend(dtf[y].unique())
        ax[1].grid(True)
    plt.show()



'''
Detect language of text.
'''
def add_detect_lang(dtf, column):
    dtf['lang'] = dtf[column].apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")    
    print(dtf[['lang']].describe().T)
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
    dtf['word_count'] = dtf[column].apply(lambda x: len(str(x).split(" ")))
    dtf['text_length'] = dtf[column].str.len()
    print(dtf[['word_count','text_length']].describe().T)
    return dtf



'''
Computes the sentiment using Textblob.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param algo: string - textblob or nltk
    :param sentiment_range: tuple - if not (-1,1) score is rescaled with sklearn
:return
    dtf: input dataframe with new sentiment column
'''
def add_sentiment(dtf, column, algo="nltk", sentiment_range=(-1,1)):
    ## calculate sentiment
    if algo=="nltk":
        nltk_sentim = SentimentIntensityAnalyzer()
        dtf["sentiment"] = dtf[column].apply(lambda x: nltk_sentim.polarity_scores(x)["compound"])
    elif algo=="textblob":
        dtf["sentiment"] = dtf[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    ## rescale
    if sentiment_range != (-1,1):
        dtf["sentiment"] = preprocessing.MinMaxScaler(feature_range=sentiment_range).fit_transform(dtf[["sentiment"]])
    print(dtf[['sentiment']].describe().T)
    return dtf



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
def add_preprocessed_text(dtf, column, lst_regex=None, flg_stemm=False, flg_lemm=True, lst_stopwords=None, remove_na=True):
    ## apply preprocess
    dtf = dtf[ pd.notnull(dtf[column]) ]
    dtf[column+"_clean"] = dtf[column].apply(lambda x: utils_preprocess_text(x, lst_regex, flg_stemm, flg_lemm, lst_stopwords))
    
    ## residuals
    dtf["check"] = dtf[column+"_clean"].apply(lambda x: len(x))
    if dtf["check"].min() == 0:
        print("--- found NAs ---")
        print(dtf[[column,column+"_clean"]][dtf["check"]==0].head())
        if remove_na is True:
            dtf = dtf[dtf["check"]>0] 
            
    return dtf.drop("check", axis=1)



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
    dtf_count.set_index("Word").iloc[:top, :].sort_values(by="Freq").plot(kind="barh", title="Top frequent words", figsize=figsize, legend=False).grid(axis='x')
    plt.show()
    return dtf_count



'''
Plots a wordcloud from a list of Docs or from a dictionary
:parameter
    :param lst_corpus: list or None
    :param dic_words: dict or None {"word1":freq1, "word2":freq2, ...}
'''
def plot_wordcloud(corpus, max_words=150, max_font_size=35, figsize=(10,10)):
    wc = wordcloud.WordCloud(background_color='black', max_words=max_words, max_font_size=max_font_size)
    wc = wc.generate(str(corpus)) if type(corpus) is not dict else wc.generate_from_frequencies(corpus)     
    fig = plt.figure(num=1, figsize=figsize)
    plt.axis('off')
    plt.imshow(wc, cmap=None)
    plt.show()
    

        
###############################################################################
#                            NER                                              #
###############################################################################
'''
Display the spacy NER model.
:parameter
    :param txt: string - text input for the model.
    :param model: string - "en_core_web_lg", "en_core_web_sm", "xx_ent_wiki_sm"
    :param lst_filter_tags: list or None - example ["ORG", "GPE", "LOC"], None for all tags
    :param title: str or None
'''
def ner_displacy(txt, ner=None, lst_filter_tags=None, title=None, serve=False):
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    doc = ner(txt)
    doc.user_data["title"] = title
    if serve == True:
        spacy.displacy.serve(doc, style="ent", options={"ents":lst_filter_tags})
    else:
        spacy.displacy.render(doc, style="ent", options={"ents":lst_filter_tags})
        
        

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
Creates columns
    :param lst_dics_tuples: [{('Texas','GPE'):1}, {('Trump','PERSON'):3}]
    :param tag: string - 'PERSON'
:return
    int
'''
def utils_ner_features(lst_dics_tuples, tag):
    if len(lst_dics_tuples) > 0:
        tag_type = []
        for dic_tuples in lst_dics_tuples:
            for tuple in dic_tuples:
                type, n = tuple[1], dic_tuples[tuple]
                tag_type = tag_type + [type]*n
                dic_counter = collections.Counter()
                for x in tag_type:
                    dic_counter[x] += 1
        return dic_counter[tag]   #pd.DataFrame([dic_counter])
    else:
        return 0



'''
Applies the spacy NER model.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param ner: spacy object - "en_core_web_lg", "en_core_web_sm", "xx_ent_wiki_sm"
    :param tag_type: string or list - "all" or ["ORG","PERSON","NORP","GPE","EVENT", ...]
:return
    {"dtf":dtf, "dtf_tags":dtf_tags}
'''
def add_ner_spacy(dtf, column, ner=None, tag_type="all", unique=False, create_features=True):
    ## ner
    print("--- tagging ---")
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    if tag_type == "all":
        if unique == True:
            dtf["tags"] = dtf[column].apply(lambda x: list(set([(word.text, word.label_) for word in ner(x).ents])) )
        else:
            dtf["tags"] = dtf[column].apply(lambda x: [(word.text, word.label_) for word in ner(x).ents] )
    else:
        if unique == True:
            dtf["tags"] = dtf[column].apply(lambda x: list(set([(word.text, word.label_) for word in ner(x).ents if word.label_ in tag_type])) )
        else:
            dtf["tags"] = dtf[column].apply(lambda x: [(word.text, word.label_) for word in ner(x).ents if word.label_ in tag_type] )
    
    ## count
    print("--- counting tags ---")
    dtf["tags"] = dtf["tags"].apply(lambda x: utils_lst_count(x, top=None))
    
    ## extract features
    if create_features == True:
        print("--- creating features ---")
        ### features set
        tags_set = []
        for lst in dtf["tags"].tolist():
            for dic in lst:
                for k in dic.keys():
                    tags_set.append(k[1])
        tags_set = list(set(tags_set))
        ### create columns
        for feature in tags_set:
            dtf["tags_"+feature] = dtf["tags"].apply(lambda x: utils_ner_features(x, feature))
    return dtf



'''
Compute frequency of spacy tags.
'''
def tags_freq(dtf, column, top=30, figsize=(10,10)):   
    tags_list = dtf[column].sum()
    map_lst = list(map(lambda x: list(x.keys())[0], tags_list))
    dtf_tags = pd.DataFrame(map_lst, columns=['tag','type'])
    dtf_tags["count"] = 1
    dtf_tags = dtf_tags.groupby(['type','tag']).count().reset_index().sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Top frequent tags", fontsize=12)
    sns.barplot(x="count", y="tag", hue="type", data=dtf_tags.iloc[:top,:], dodge=False, ax=ax)
    ax.grid(axis="x")
    plt.show()
    return dtf_tags
        

        
'''
Retrain spacy NER model with new tags.
:parameter
    :param train_data: list [
            ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
            ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}), 
        ]
    :param output_dir: string - path of directory to save model
    :param model: string - "blanck" or "en_core_web_lg", ...
    :param n_iter: num - number of iteration
'''
def retrain_ner_spacy(train_data, output_dir, model="blank", n_iter=100):
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
#             MODEL DESIGN & TESTING - MULTILABEL CLASSIFICATION              #
###############################################################################
'''
Transform an array of strings into an array of int.
'''
def encode_variable(dtf, column):
    dtf[column+"_id"] = dtf[column].factorize(sort=True)[0]
    dic_class_mapping = dict( dtf[[column+"_id",column]].drop_duplicates().sort_values(column+"_id").values )
    return dtf, dic_class_mapping



'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
    :param predicted_prob: array
    :param figsize: tuple - plot setting
'''
def evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(20,10)):
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
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ## roc
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test_array[:,i], predicted_prob[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        ax[0].plot(fpr[i], tpr[i], lw=3, label='ROC curve of {0} (area={1:0.2f})'''.format(classes[i], roc_auc[i]))
    ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[0.0,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)
    
    ## precision-recall curve
    precision, recall = dict(), dict()
    for i in range(len(classes)):
        precision[i], recall[i], thresholds = metrics.precision_recall_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[1].plot(recall[i], precision[i], lw=3, label='{}'.format(classes[i]))
    ax[1].set(xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()



###############################################################################
#                     BAG OF WORDS (VECTORIZER)                               #
###############################################################################
'''
Compute freqeuncy of features in a sparse matrix.
'''
def utils_features_freq(dic_vocabulary, X, top=20, figsize=(10,5)):
    lst_score = [(idx, word, X.sum(axis=0)[0,idx]) for word,idx in dic_vocabulary.items()]
    lst_score = sorted(lst_score, key=lambda x: x[1], reverse=True)
    dtf_score = pd.DataFrame(lst_score, columns=["idx","word","freq"]).sort_values(by="freq", ascending=False)
    dtf_score[["word","freq"]].set_index("word").sort_values(by="freq").tail(top).plot(kind="barh", title="Top frequent features", figsize=figsize, legend=False).grid(axis='x')
    return dtf_score.set_index("idx")



'''
Vectorizes the corpus (tfidf), fits the Bag of words model, plots the most frequent words.
:parameter
    :param corpus: list corpus
    :param vectorizer: sklearn vectorizer object
    :param vocabulary: list of words or dict, if None it creates from scratch, else it searches the words into corpus
    :param top: num - plot the top frequent words
:return
    dict with X, vectorizer, dic_vocabulary, dtf_freq, lst_text2tokens
'''
def fit_bow(corpus, vectorizer=None, vocabulary=None, top=None, figsize=(10,5)):
    ## vectorizer
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,1), vocabulary=vocabulary) if vectorizer is None else vectorizer
    vectorizer.fit(corpus)
    
    ## sparse matrix
    print("--- creating sparse matrix ---")
    X = vectorizer.transform(corpus)
    print("shape:", X.shape)
    
    ## vocabulary
    print("--- creating vocabulary ---") if vocabulary is None else print("--- used vocabulary ---")
    dic_vocabulary = vectorizer.vocabulary_   #{word:idx for idx, word in enumerate(vectorizer.get_feature_names())}
    print("len:", len(dic_vocabulary))
    
    ## text2tokens
    print("--- tokenization ---")
    tokenizer = vectorizer.build_tokenizer()
    preprocessor = vectorizer.build_preprocessor()
    lst_text2tokens = []
    for text in corpus:
        lst_tokens = [dic_vocabulary[word] for word in tokenizer(preprocessor(text)) if word in dic_vocabulary]
        lst_text2tokens.append(lst_tokens)
    print("len:", len(lst_text2tokens))
    
    ## plot stats
    dtf_freq = utils_features_freq(dic_vocabulary, X, top, figsize) if top is not None else None
    
    return {"X":X, "lst_text2tokens":lst_text2tokens,
            "vectorizer":vectorizer, "dic_vocabulary":dic_vocabulary, 
            "dtf_freq":dtf_freq}
    


'''
Perform feature selection.
'''
def features_selection(X, y, vectorizer_fitted, top=None):
    top_by_cat = int(top/len(np.unique(y))) if top is not None else top
    feature_names = np.array(vectorizer_fitted.get_feature_names())
    dic_words_selection = {}
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X, y == cat)
        index_sorted = np.argsort(p)
        p_sorted = p[index_sorted]
        features_sorted = feature_names[index_sorted]
        selected_features = features_sorted[:sum(p_sorted<0.05)] if top is None else features_sorted[:top_by_cat]
        dic_words_selection.update({cat:selected_features.tolist()})
        print("# {}:".format(cat))
        print("  . {}".format('\n  . '.join(selected_features[:5])))
        print(" ")
    dic_words_selection.update({"ALL":list(set([x for lst in dic_words_selection.values() for x in lst]))})
    return dic_words_selection



'''
Transform a sparse matrix into a dtf with selected features only.
'''
def sparse2dtf(X, dic_vocabulary, lst_words):
    dtf_X = pd.DataFrame()
    for word in lst_words:
        idx = dic_vocabulary[word]
        dtf_X["X_"+word] = np.reshape(X[:,idx].toarray(), newshape=(-1))
    return dtf_X



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
def ml_text_classif(X_train, y_train, X_test, y_test, preprocessing=False, vectorizer=None, classifier=None): 
    ## model
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,2)) if vectorizer is None else vectorizer
    classifier = naive_bayes.MultinomialNB() if classifier is None else classifier
    model = pipeline.Pipeline([("vectorizer",vectorizer), ("model",classifier)]) if preprocessing == True else classifier
    
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



###############################################################################
#                            WORD2VEC (EMBEDDINGS)                            #
###############################################################################
'''
'''
def utils_preprocess_ngrams(corpus, ngrams=1, grams_join=" "):
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)
    return lst_corpus



'''
Fits the Word2Vec model from gensim.
:parameter
    :param corpus: list - list of lists of words
    :param min_count: num - ignores all words with total frequency lower than this
    :param size: num - dimensionality of the vectors
    :param window: num - ( x x x ... x  word  x ... x x x)
    :param sg: num - 0 for CBOW, 1 for skipgrams
    :param lst_bigrams_stopwords: list - ["of","with","without","and","or","the","a"]
:return
    lst_corpus and the nlp model
'''
def fit_w2v(corpus, ngrams=1, grams_join="_", min_count=1, size=300, window=20, sg=0, epochs=30, lst_bigrams_stopwords=[]):
    ## preprocess
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams, grams_join)
    ## detect bigrams
    if ngrams == 1:
        phrases = gensim.models.phrases.Phrases(lst_corpus, common_terms=lst_bigrams_stopwords)
        bigram = gensim.models.phrases.Phraser(phrases)
        lst_corpus = list(bigram[lst_corpus])
    ## training
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=size, window=window, min_count=min_count, sg=sg, iter=epochs)
    return lst_corpus, nlp.wv
   


'''
'''
def plot_w2v(nlp=None, plot_type="2d", word=None, top=20, figsize=(10,5)):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    try:
        if plot_type == "2d": 
            labels, X, x, y = [], [], [], []
            ## plot word
            if type(word) is str:  
                for tupla in nlp.most_similar(word, topn=top):
                    X.append(nlp[tupla[0]])
                    labels.append(tupla[0])
             ## plot all
            else:
                for token in nlp.vocab:
                    X.append(nlp[token])
                    labels.append(token)
            ## pca
            pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
            new_values = pca.fit_transform(X)
            for value in new_values:
                x.append(value[0])
                y.append(value[1])
            ## plot
            fig = plt.figure(figsize=figsize)
            fig.suptitle("Word: "+word, fontsize=12) if type(word) is str else fig.suptitle("Vocabulary", fontsize=12)
            for i in range(len(x)):
                plt.scatter(x[i], y[i], c="black")
                plt.annotate(labels[i], xy=(x[i],y[i]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
            ## add center
            if type(word) is str:
                plt.scatter(x=0, y=0, c="red")
                plt.annotate(word, xy=(0,0), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
            
        elif plot_type == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            ## plot word
            if type(word) is str: 
                lst_words = [tupla[0] for tupla in nlp.most_similar(word, topn=top)]
            ## plot all
            else:
                lst_words = list(nlp.vocab.keys())
            X = nlp[lst_words]
            ## pca
            pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
            new_values = pca.fit_transform(X)
            dtf_tmp = pd.DataFrame(new_values, index=lst_words, columns=['x','y','z'])
            ## plot
            fig = plt.figure(figsize=figsize)
            fig.suptitle("Word: "+word, fontsize=12) if type(word) is str else fig.suptitle("Vocabulary", fontsize=12)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dtf_tmp['x'], dtf_tmp['y'], dtf_tmp['z'], c="black")
            for label, row in dtf_tmp.iterrows():
                x, y, z = row
                ax.text(x, y, z, s=label)
            ## add center
            if type(word) is str:
                ax.scatter(0, 0, 0, c="red")
                ax.text(0, 0, 0, s=word)
            
        plt.show()
        
    except Exception as e:
        print("--- got error ---")
        print(e)
        if type(word) is str:
            print("maybe you are looking for ... ")
            print([k for k in list(nlp.vocab.keys()) if difflib.SequenceMatcher(isjunk=None, a=word, b=k).ratio()>0.7])
    


'''
Embeds a vocabulary of unigrams with gensim w2v.
'''
def vocabulary_embeddings(dic_vocabulary, nlp=None, dim_space=300):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    embeddings = np.zeros((len(dic_vocabulary)+1, dim_space))
    for word,idx in dic_vocabulary.items():
        try:
            embeddings[idx] =  nlp[word]
        except:
            pass
    print("shape: ", embeddings.shape)
    return embeddings


    
'''
Transforms the corpus into an array of sequences of idx with same length.
    :param
'''
def text2seq(corpus, vectorizer=None, vocabulary=None, maxlen=None, top=None, figsize=(10,5)):
    ## fit BoW with fixed vocabulary
    dic = fit_bow(corpus, vectorizer=None, vocabulary=vocabulary, top=top, figsize=figsize)
    lst_text2tokens, dic_vocabulary = dic["lst_text2tokens"], dic["dic_vocabulary"]
    
    ## create sequence with keras preprocessing
    print("--- padding to sequence ---")
    maxlen = np.max([len(text.split()) for text in corpus]) if maxlen is None else maxlen
    X = kprocessing.sequence.pad_sequences(lst_text2tokens, maxlen=maxlen, value=len(dic_vocabulary))
    print("shape:", X.shape)
    return X, dic_vocabulary



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
def dl_text_classif(dic_y_mapping, embeddings, X_train, y_train, X_test, y_test, model=None, epochs=10, batch_size=256):
    ## encode y
    inverse_dic = {v:k for k,v in dic_y_mapping.items()}
    y_train = [inverse_dic[y] for y in y_train]   #y_test = [dic_y_mapping[y] for y in y_test]
    
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
    print(model.summary())
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=0.3)
    fig, ax = plt.subplots()
    ax.plot(training.history['loss'], label='loss')
    ax.grid(True)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    ## test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
    print("Check --> True:", y_test[0], "Pred:", predicted[0], "Prob:", np.max(predicted_prob[0]))
    
    ## kpi
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("Accuracy (overall correct predictions):",  round(accuracy,3))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))
    return {"model":training.model, "predicted_prob":predicted_prob, "predicted":predicted}



###############################################################################
#                        WORD2VEC CLUSTERING                                  #
###############################################################################
'''
Clusters a Word2Vec vocabulary with nltk Kmeans.
:parameter
    :param modelW2V: word2vec model (if given uses the model already trained)
    :param k: num - clusters
:return
    dtf with words and clusters
'''
def plot_word_clustering(nlp=None, k=3):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    X = nlp[nlp.vocab.keys()]    
    kmeans_model = nltk.cluster.KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=50)
    clusters = kmeans_model.cluster(X, assign_clusters=True)
    dic_clusters = {word:clusters[i] for i,word in enumerate(list(nlp.vocab.keys()))}
    dtf_cluster = pd.DataFrame({"word":word, "cluster":str(clusters[i])} for i,word in enumerate(list(nlp.vocab.keys())))
    dtf_cluster = dtf_cluster.sort_values(["cluster", "word"], ascending=[True,True])
    return 0



'''
Creates a feature matrix (num_docs x vector_size)
'''
def utils_text_embeddings(corpus, nlp, value_na=0):
    lst_X = []
    for text in corpus:
        lst_word_vecs = [nlp[word] if word in nlp.vocab.keys() else [value_na]*nlp.vector_size 
                         for word in text.split()]
        lst_X.append(np.mean( np.array(lst_word_vecs), axis=0 )) 
    X = np.stack(lst_X, axis=0)
    return X



'''
'''
def fit_pca_w2v(corpus, nlp):
    ## corpus embedding
    X = utils_text_embeddings(corpus, nlp, value_na=0)
    print("X shape:", X.shape)
    ## fit pca
    model = decomposition.PCA(n_components=nlp.vector_size)
    pca = model.fit_transform(X)
    print("pca shape:", pca.shape)
    return pca



'''
'''
def similarity_w2v(a, b, nlp=None):
    if type(a) is str and type(b) is str:
        cosine_sim = nlp.similarity(a,b)
    else:
        a = a.reshape(1,-1) if len(a.shape) == 1 else a
        b = b.reshape(1,-1) if len(b.shape) == 1 else b
        cosine_sim = metrics.pairwise.cosine_similarity(a,b)[0][0]
       #cosine_sim = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cosine_sim



'''
Clustering of text to specifi classes (Unsupervised Classification by similarity).
:parameter
    :param text: array of text to predict
    :param dic_clusters: dic of lists of strings - {'finance':['market','bonds','equity'], 'esg':['environment','green_economy','sustainability']}
:return
    dic_clusters_sim = {'finance':0.7, 'esg':0.5}
'''
def predict_clusters_w2v(corpus, dic_clusters, nlp=None, pca=None):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    print("--- embedding X and y ---")
    
    ## clusters embedding
    dic_y, cluster_names = {}, []
    for name, lst_keywords in dic_clusters.items():
        lst_word_vecs = [nlp[word] if word in nlp.vocab.keys() else [0]*nlp.vector_size 
                         for word in lst_keywords]
        dic_y.update({name:np.mean( np.array(lst_word_vecs), axis=0)})
        cluster_names.append(name)
        print(name, "shape:", dic_y[name].shape)
    
    ## text embedding
    X = utils_text_embeddings(corpus, nlp, value_na=0)
    print("X shape:", X.shape)
    
    ## remove pca
    if pca is not None:
        print("--- removing general component ---")
        ### from y
        for name, y in dic_y.items():
            dic_y[name] = y - y.dot(pca.transpose()).dot(pca)
        ### from X
        X = X - X.dot(pca.transpose()).dot(pca)
        
    ## compute similarity
    print("--- computing similarity ---")
    predicted_prob = np.array([metrics.pairwise.cosine_similarity(X,y.reshape(1,-1)).T.tolist()[0] for y in dic_y.values()]).T
    predicted = [cluster_names[np.argmax(pred)] for pred in predicted_prob]
    return predicted_prob, predicted



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
        cosine_sim = metrics.pairwise.cosine_similarity(lst_vectors)[0,1]
        return cosine_sim
    
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