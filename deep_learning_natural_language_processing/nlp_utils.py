
## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for analysis
import re
import langdetect 
import nltk
import wordcloud

## for sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

## for ner
import spacy
import collections

## for machine learning
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline, decomposition

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

## for explainer
from lime import lime_text
import shap

## for W2V
import gensim
import gensim.downloader as gensim_api



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
        fig.suptitle(x, fontsize=15)
        if top is None:
            dtf[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
        else:   
            dtf[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", legend=False, ax=ax).grid(axis='x')
        ax.set(ylabel=None)

    ## bivariate
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
        fig.suptitle(x, fontsize=15)
        for i in dtf[y].unique():
            sns.distplot(dtf[dtf[y]==i][x], hist=True, kde=False, bins=bins, hist_kws={"alpha":0.8}, axlabel="", ax=ax[0])
            sns.distplot(dtf[dtf[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="", ax=ax[1])
        ax[0].set(title="histogram")
        ax[0].grid(True)
        ax[0].legend(dtf[y].unique())
        ax[1].set(title="density")
        ax[1].grid(True)
    plt.show()



'''
Detect language of text.
'''
def add_detect_lang(dtf, column):
    dtf['lang'] = dtf[column].apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")
    return dtf



'''
Compute different text length metrics.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    dtf: input dataframe with 2 new columns
'''
def add_text_length(dtf, column):
    dtf['word_count'] = dtf[column].apply(lambda x: len(str(x).split(" ")))
    dtf['char_count'] = dtf[column].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
    dtf['sentence_count'] = dtf[column].apply(lambda x: len(str(x).split(".")))
    dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']
    dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']
    print(dtf[['word_count','char_count','sentence_count','avg_word_length','avg_sentence_lenght']].describe().T[["min","mean","max"]])
    return dtf



'''
Computes the sentiment using Textblob or Vader.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param algo: string - "textblob" or "vader"
    :param sentiment_range: tuple - if not (-1,1) score is rescaled with sklearn
:return
    dtf: input dataframe with new sentiment column
'''
def add_sentiment(dtf, column, algo="vader", sentiment_range=(-1,1)):
    ## calculate sentiment
    if algo == "vader":
        vader = SentimentIntensityAnalyzer()
        dtf["sentiment"] = dtf[column].apply(lambda x: vader.polarity_scores(x)["compound"])
    elif algo == "textblob":
        dtf["sentiment"] = dtf[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    ## rescaled
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
def utils_preprocess_text(text, lst_regex=None, lst_stopwords=None, flg_stemm=False, flg_lemm=True):
    ## regex (in case, before processing)
    if lst_regex is not None: 
        for regex in lst_regex:
            text = re.sub(regex, '', text)
    
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()

    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## remove leftover Stopwords
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
def add_preprocessed_text(dtf, column, lst_regex=None, lst_stopwords=None, flg_stemm=False, flg_lemm=True, remove_na=True):
    ## apply preprocess
    dtf = dtf[ pd.notnull(dtf[column]) ]
    dtf[column+"_clean"] = dtf[column].apply(lambda x: utils_preprocess_text(x, lst_regex, lst_stopwords, flg_stemm, flg_lemm))
    
    ## residuals
    dtf["check"] = dtf[column+"_clean"].apply(lambda x: len(x))
    if dtf["check"].min() == 0:
        print("--- found NAs ---")
        print(dtf[[column,column+"_clean"]][dtf["check"]==0].head())
        if remove_na is True:
            dtf = dtf[dtf["check"]>0] 
            
    return dtf.drop("check", axis=1)



'''
Compute n-grams frequency.
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: int or list - 1 for unigrams, 2 for bigrams, [1,2] for both
    :param top: num - plot the top frequent words
:return
    dtf_count: dtf with word frequency
'''
def word_freq(corpus, ngrams=1, top=10, figsize=(10,7)):
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    ngrams = [ngrams] if type(ngrams) is int else ngrams
    
    ## calculate
    dtf_freq = pd.DataFrame()
    for n in ngrams:
        dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
        dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=["word","freq"])
        dtf_n["ngrams"] = n
        dtf_freq = dtf_freq.append(dtf_n)
    dtf_freq["word"] = dtf_freq["word"].apply(lambda x: " ".join(string for string in x) )
    dtf_freq = dtf_freq.sort_values(["ngrams","freq"], ascending=[True,False])
    
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=dtf_freq.groupby('ngrams')["ngrams","freq","word"].head(top))
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    plt.show()
    return dtf_freq



'''
Plots a wordcloud from a list of Docs or from a dictionary
:parameter
    :param corpus: list - dtf["text"]
'''
def plot_wordcloud(corpus, max_words=150, max_font_size=35, figsize=(10,10)):
    wc = wordcloud.WordCloud(background_color='black', max_words=max_words, max_font_size=max_font_size)
    wc = wc.generate(str(corpus)) #if type(corpus) is not dict else wc.generate_from_frequencies(corpus)     
    fig = plt.figure(num=1, figsize=figsize)
    plt.axis('off')
    plt.imshow(wc, cmap=None)
    plt.show()



'''
Adds a column with word frequency.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param lst_words: list - ["donald trump", "china", ...]
    :param freq: str - "count" or "tfidf"
:return
    dtf: input dataframe with new columns
'''
def add_word_freq(dtf, column, lst_words, freq="count"):
    lst_grams = [len(word.split(" ")) for word in lst_words]
    if freq == "tfidf":
        vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=lst_words, ngram_range=(min(lst_grams),max(lst_grams)))
    else:
        vectorizer = feature_extraction.text.CountVectorizer(vocabulary=lst_words, ngram_range=(min(lst_grams),max(lst_grams)))
    dtf_X = pd.DataFrame(vectorizer.fit_transform(dtf[column]).todense(), columns=lst_words)
    return pd.concat([dtf, dtf_X.set_index(dtf.index)], axis=1)


        
###############################################################################
#                            NER                                              #
###############################################################################
'''
Display the spacy NER model.
:parameter
    :param txt: string - text input for the model.
    :param model: string - "en_core_web_lg", "en_core_web_sm", "xx_ent_wiki_sm"
    :param lst_tag_filter: list or None - example ["ORG", "GPE", "LOC"], None for all tags
    :param title: str or None
'''
def ner_displacy(txt, ner=None, lst_tag_filter=None, title=None, serve=False):
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    doc = ner(txt)
    doc.user_data["title"] = title
    if serve == True:
        spacy.displacy.serve(doc, style="ent", options={"ents":lst_tag_filter})
    else:
        spacy.displacy.render(doc, style="ent", options={"ents":lst_tag_filter})



'''
Find entities in text, replace strings with tags and extract tags:
    Donald Trump --> Donald_Trump
    [Donald Trump, PERSON]
'''
def utils_ner_text(txt, ner=None, lst_tag_filter=None, grams_join="_"):
    ## apply model
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    entities = ner(txt).ents

    ## tag text
    tagged_txt = txt
    for tag in entities:
        if (lst_tag_filter is None) or (tag.label_ in lst_tag_filter):
            try:
                tagged_txt = re.sub(tag.text, grams_join.join(tag.text.split()), tagged_txt) #it breaks with wild characters like *+
            except Exception as e:
                next

    ## extract tags list
    if lst_tag_filter is None:
        lst_tags = [(tag.text, tag.label_) for tag in entities]  #list(set([(word.text, word.label_) for word in ner(x).ents]))
    else: 
        lst_tags = [(word.text, word.label_) for word in entities if word.label_ in lst_tag_filter]

    return tagged_txt, lst_tags
        
        

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
Apply spacy NER model and add tag features.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
    :param ner: spacy object - "en_core_web_lg", "en_core_web_sm", "xx_ent_wiki_sm"
    :param lst_tag_filter: list - ["ORG","PERSON","NORP","GPE","EVENT", ...]. If None takes all
    :param grams_join: string - "_", " ", or more (ex. "new york" --> "new_york")
    :param create_features: bool - create columns with category features
:return
    dtf
'''
def add_ner_spacy(dtf, column, ner=None, lst_tag_filter=None, grams_join="_", create_features=True):
    ner = spacy.load("en_core_web_lg") if ner is None else ner

    ## tag text and exctract tags
    print("--- tagging ---")
    dtf[[column+"_tagged", "tags"]] = dtf[[column]].apply(lambda x: utils_ner_text(x[0], ner, lst_tag_filter, grams_join), 
                                                          axis=1, result_type='expand')

    ## put all tags in a column
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
def tags_freq(tags, top=30, figsize=(10,5)):   
    tags_list = tags.sum()
    map_lst = list(map(lambda x: list(x.keys())[0], tags_list))
    dtf_tags = pd.DataFrame(map_lst, columns=['tag','type'])
    dtf_tags["count"] = 1
    dtf_tags = dtf_tags.groupby(['type','tag']).count().reset_index().sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("Top frequent tags", fontsize=12)
    sns.barplot(x="count", y="tag", hue="type", data=dtf_tags.iloc[:top,:], dodge=False, ax=ax)
    ax.set(ylabel=None)
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
Split the dataframe into train / test
'''
def dtf_partitioning(dtf, y, test_size=0.3, shuffle=False):
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size, shuffle=shuffle) 
    print("X_train shape:", dtf_train.drop(y, axis=1).shape, "| X_test shape:", dtf_test.drop(y, axis=1).shape)
    print("y:")
    for i in dtf_train["y"].value_counts(normalize=True).index:
        print(" ", i, " -->  train:", round(dtf_train["y"].value_counts(normalize=True).loc[i], 2),
                          "| test:", round(dtf_test["y"].value_counts(normalize=True).loc[i], 2))
    print(dtf_train.shape[1], "features:", dtf_train.drop(y, axis=1).columns.to_list())
    return dtf_train, dtf_test



'''
Transform an array of strings into an array of int.
'''
def add_encode_variable(dtf, column):
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
def evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15,5)):
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")
    print("Accuracy:",  round(accuracy,2))
    print("Auc:", round(auc,2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))
    
    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))
    ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)
    
    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[1].plot(recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()



###############################################################################
#                     BAG OF WORDS (VECTORIZER)                               #
###############################################################################
'''
Vectorize corpus with Bag-of-Words (classic Count or Tf-Idf variant), plots the most frequent words.
:parameter
    :param corpus: list - dtf["text"]
    :param vectorizer: sklearn vectorizer object, like Count or Tf-Idf
    :param vocabulary: list of words or dict, if None it creates from scratch, else it searches the words into corpus
:return
    sparse matrix, list of text tokenized, vectorizer, dic_vocabulary, X_names
'''
def fit_bow(corpus, vectorizer=None, vocabulary=None):
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
    print(len(dic_vocabulary), "words")
    
    ## text2tokens
    print("--- tokenization ---")
    tokenizer = vectorizer.build_tokenizer()
    preprocessor = vectorizer.build_preprocessor()
    lst_text2tokens = []
    for text in corpus:
        lst_tokens = [dic_vocabulary[word] for word in tokenizer(preprocessor(text)) if word in dic_vocabulary]
        lst_text2tokens.append(lst_tokens)
    print(len(lst_text2tokens), "texts")
    return {"X":X, "lst_text2tokens":lst_text2tokens, "vectorizer":vectorizer, "dic_vocabulary":dic_vocabulary, "X_names":vectorizer.get_feature_names()}



'''
Perform feature selection using p-values (keep highly correlated features)
:parameter
    :param X: array - like sparse matrix or dtf.values
    :param y: array or dtf - like dtf["y"]
    :param X_names: list - like vetcorizer.get_feature_names()
    :param top: int - ex. 1000 takes the top 1000 features per classes of y. If None takes all those with p-value < 5%.
    :param print_top: int - print top features
:return
    dtf with features and scores
'''
def features_selection(X, y, X_names, top=None, print_top=10):    
    ## selection
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X, y == cat)
        dtf_features = dtf_features.append(pd.DataFrame({"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], ascending=[True,False])
    dtf_features = dtf_features[dtf_features["score"]>0.95] #p-value filter
    if top is not None:
        dtf_features = dtf_features.groupby('y')["y","feature","score"].head(top)
    
    ## print
    print("features selection: from", "{:,.0f}".format(len(X_names)), 
          "to", "{:,.0f}".format(len(dtf_features["feature"].unique())))
    print(" ")
    for cat in np.unique(y):
        print("# {}:".format(cat))
        print("  . selected features:", len(dtf_features[dtf_features["y"]==cat]))
        print("  . top features:", ", ".join(dtf_features[dtf_features["y"]==cat]["feature"].values[:print_top]))
        print(" ")
    return dtf_features["feature"].unique().tolist(), dtf_features



'''
Transform a sparse matrix into a dtf with selected features only.
:parameter
    :param X: array - like sparse matrix or dtf.values
    :param dic_vocabulary: dict - {"word":idx}
    :param X_names: list of words - like vetcorizer.get_feature_names()
    :param prefix: str - ex. "x_" -> x_word1, x_word2, ..
'''
def sparse2dtf(X, dic_vocabulary, X_names, prefix=""):
    dtf_X = pd.DataFrame()
    for word in X_names:
        idx = dic_vocabulary[word]
        dtf_X[prefix+word] = np.reshape(X[:,idx].toarray(), newshape=(-1))
    return dtf_X



'''
Fits a sklearn classification model.
:parameter
    :param X_train: feature matrix
    :param y_train: array of classes
    :param X_test: raw text
    :param vectorizer: ftted vectorizer object
    :param classifier: model object - if None MultinomialNB is used
:return
    fitted model and predictions
'''
def ml_text_classif(X_train, y_train, X_test, vectorizer, classifier=None): 
    ## model pipeline
    classifier = naive_bayes.MultinomialNB() if classifier is None else classifier
    model = pipeline.Pipeline([("vectorizer",vectorizer), ("classifier",classifier)])
    
    ## train
    model["classifier"].fit(X_train, y_train)
    
    ## test
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)
    return model, predicted_prob, predicted



'''
Use lime to build an a explainer.
:parameter
    :param model: pipeline with vectorizer and classifier
    :param Y_train: array
    :param txt_instance: string - raw text
    :param top: num - top features to display
:return
    dtf with explanations
'''
def explainer_lime(model, y_train, txt_instance, top=10):
    explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))
    explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=top)
    dtf_explainer = pd.DataFrame(explained.as_list(), columns=['feature','effect'])
    explained.show_in_notebook(text=txt_instance, predict_proba=False)
    return dtf_explainer



###############################################################################
#                            WORD2VEC (EMBEDDINGS)                            #
###############################################################################
'''
Create a list of lists of grams:
    [ ["hi", "my", "name", "is", "Tom"], 
      ["what", "is", "yours"] ]
'''
def utils_preprocess_ngrams(corpus, ngrams=1, grams_join="_"):
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)
    return lst_corpus



'''
Fits the Word2Vec model from gensim.
:parameter
    :param corpus: list - dtf["text"]
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
    ## detect bigrams ("new", "york" --> "new_york")
    if ngrams == 1:
        phrases = gensim.models.phrases.Phrases(lst_corpus, common_terms=lst_bigrams_stopwords)
        bigram = gensim.models.phrases.Phraser(phrases)
        lst_corpus = list(bigram[lst_corpus])
    ## training
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=size, window=window, min_count=min_count, sg=sg, iter=epochs)
    return lst_corpus, nlp.wv
   


'''
Plot words in vector space (2d or 3d):
    - If word is None: plot the whole vocabulary.
    - else: plot only the words around the input word.
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
            print([k for k in list(nlp.vocab.keys()) if 1-nltk.jaccard_distance(set(word),set(k)) > 0.7])
    


'''
Embeds a vocabulary of unigrams with gensim w2v.
:parameter
    :param dic_vocabulary: dict - {"word":1, "word":2, ...}
    :param nlp: gensim model
    :param size: num - dimensionality of the vectors
:return
    Matric and the nlp model
'''
def vocabulary_embeddings(dic_vocabulary, nlp=None, size=300):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    embeddings = np.zeros((len(dic_vocabulary)+1, size))
    for word,idx in dic_vocabulary.items():
        ## update the row with vector
        try:
            embeddings[idx] =  nlp[word]
        ## if word not in model then skip and the row stays all zeros
        except:
            pass
    print("vocabulary mapped to", embeddings.shape[0], "vectors of size", embeddings.shape[1])
    return embeddings


    
'''
Transforms the corpus into an array of sequences of idx with same length.
:parameter
    :param corpus: list - dtf["text"]
    :param vocabulary: list of words or dict, if None it creates from scratch, else it searches the words into corpus
    :param maxlen: num - dimensionality of the vectors, if None takes the max length in corpus
    :param padding: string - "pre" for [9999,1,2,3] "post"  for [1,2,3,9999]
:return
    Matrix of sequences and dic_vocabulary
'''
def text2seq(corpus, vocabulary=None, maxlen=None, padding="pre"):
    ## fit BoW to get text2tokens (because pad_sequences requires number inputs)
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,1), vocabulary=vocabulary)
    dic = fit_bow(corpus, vectorizer=vectorizer, vocabulary=vocabulary)
    lst_text2tokens, dic_vocabulary = dic["lst_text2tokens"], dic["dic_vocabulary"]
    
    ## create sequence with keras preprocessing (from [1,2],[3,4,5] to [1,2,9999],[3,4,5])
    print("--- padding to sequence ---")
    maxlen = np.max([len(text.split()) for text in corpus]) if maxlen is None else maxlen
    X = kprocessing.sequence.pad_sequences(lst_text2tokens, maxlen=maxlen, value=len(dic_vocabulary), padding=padding, truncating=padding)
    print(X.shape[0], "sequences of length", X.shape[1]) 
    return X, dic_vocabulary



'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()



'''
Fits a keras classification model.
:parameter
    :param dic_y_mapping: dict - {0:"A", 1:"B", 2:"C"}. If None it calculates.
    :param X_train: array of sequence
    :param y_train: array of classes
    :param X_test: array of sequence
    :param model: model object - model to fit (before fitting)
    :param weights: array of weights - like embeddings
:return
    model fitted and predictions
'''
def dl_text_classif(X_train, y_train, X_test, encode_y=False, dic_y_mapping=None, model=None, weights=None, epochs=10, batch_size=256):
    ## encode y
    if encode_y is True:
        dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
        inverse_dic = {v:k for k,v in dic_y_mapping.items()}
        y_train = [inverse_dic[y] for y in y_train]
        print(dic_y_mapping)
    
    ## model
    if model is None:
        ### params
        n_features, embeddings_dim = weights.shape
        max_seq_lenght = X_train.shape[1]
        ### neural network
        model = models.Sequential([
            layers.Embedding(input_dim=n_features, output_dim=embeddings_dim, weights=[weights], input_length=max_seq_lenght, trainable=False),
            layers.LSTM(units=X_train.shape[1], dropout=0.2),
            layers.Dense(len(np.unique(y_train)), activation='softmax')])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        
    ## train
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0, validation_split=0.3)
    utils_plot_keras_training(training)
    
    ## test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob] if encode_y is True else [np.argmax(pred)]
    return training.model, predicted_prob, predicted



'''
Takes the weights of an Attention layer and builds an explainer.
:parameter
    :param model: model instance (after fitting)
    :param X_instance: array of size n (n,)
    :param dic_vocabulary: dict - {"word":0, ...}
    :param txt_instance: string - raw text
    :param similarity_thres: num - if None it doesn't use similarity
    :param top: num - top features to display
:return
    text html, it can be visualized on notebook with display(HTML(text))
'''
def explainer_attention(model, X_instance, dic_vocabulary, txt_instance=None, similarity_thres=None, top=10, figsize=(5,3)):
    ## get attention weights
    layer = [layer for layer in model.layers if "attention" in layer.name][0]
    func = K.function([model.input], [layer.output])
    weights = func(X_instance.reshape(1,-1))[0]
    weights = np.mean(weights, axis=2).flatten()
    
    ## remove null vectors 
    inverted_dic = {v:k for k,v in dic_vocabulary.items()}
    words, weights_ = [], []
    for n,idx in enumerate(X_instance):
        if idx < len(dic_vocabulary):
            words.append(inverted_dic[idx])
            weights_.append(weights[n])
    
    ## rescale weights between 0-1   
    weights = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(weights_).reshape(-1,1)).reshape(-1)
    dic_word_weigth = {word:weights[n] for n,word in enumerate(words)}
    
    ## plot
    dtf = pd.DataFrame.from_dict(dic_word_weigth, orient='index', columns=["score"])
    dtf.sort_values(by="score", ascending=True).tail(top).plot(kind="barh", legend=False, figsize=figsize).grid(axis='x')
    plt.show()

    ## return html
    text = []
    txt_instance = txt_instance if txt_instance is not None else ' '.join(words)
    for word in txt_instance.split():

        ### match words in vocabulary with txt_instance if txt_instance is not preprocessed and vocabulary is
        if similarity_thres is not None:
            similarities = [1-nltk.jaccard_distance(set(word.lower()), set(word_in_vocab.lower())) for word_in_vocab in words]
            max_pos = np.argmax(similarities)
            word_in_vocab, similarity = words[max_pos], similarities[max_pos]
            weight = dic_word_weigth.get(word_in_vocab) if similarity > similarity_thres else None
        else:
            weight = dic_word_weigth.get(word_in_vocab)

        ### create html visualization (yellow:255,215,0 | blue:100,149,237)
        if weight is not None:
            text.append('<b><span style="background-color:rgba(100,149,237,' + str(weight) + ');">' + word + '</span></b>')
        else:
            text.append(word)

    text = ' '.join(text)
    return text



'''
Use shap to build an a explainer (works only if model has binary_crossentropy).
:parameter
    :param model: model instance (after fitting)
    :param X_train: array
    :param X_instance: array of size n (n,)
    :param dic_vocabulary: dict - {"word":0, ...}
    :param class_names: list - labels
    :param top: num - top features to display
:return
    dtf with explanations
'''
def explainer_shap(model, X_train, X_instance, dic_vocabulary, class_names, top=10):
    explainer = shap.DeepExplainer(model, data=X_train[:100])
    shap_values = explainer.shap_values(X_instance.reshape(1,-1))
    inv_dic_vocabulary = {v:k for k,v in dic_vocabulary.items()}
    X_names = [inv_dic_vocabulary[idx] if idx in dic_vocabulary.values() else " " for idx in X_instance]
    shap.summary_plot(shap_values, feature_names=X_names, class_names=class_names, plot_type="bar") 



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
Fit a PCA to model the general component common among the whole corpus.
    :param corpus: list - dtf["text"]
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
Compute cosine similarity between 2 words or 2 vectors.
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
    :param corpus: list - dtf["text"]
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
    :param corpus: list - dtf["text"]
    :param ngrams: num
    :param grams_join: string - "_", " ", or more (ex. "new york" --> "new_york")
    :param n_topics: num - number of topics to find
:return
    model and dtf topics
'''
def fit_lda(corpus, ngrams=1, grams_join="_", n_topics=3, figsize=(10,7)):
    ## train the lda
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams, grams_join)
    id2word = gensim.corpora.Dictionary(lst_corpus) #map words with an id
    dic_corpus = [id2word.doc2bow(word) for word in lst_corpus]  #create dictionary Word:Freq
    print("--- training ---")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=dic_corpus, id2word=id2word, num_topics=n_topics, 
                                                random_state=123, update_every=1, chunksize=100, 
                                                passes=10, alpha='auto', per_word_topics=True)
    
    ## output
    lst_dics = []
    for i in range(0, n_topics):
        lst_tuples = lda_model.get_topic_terms(i)
        for tupla in lst_tuples:
            lst_dics.append({"topic":i, "id":tupla[0], "word":id2word[tupla[0]], "weight":tupla[1]})
    dtf_topics = pd.DataFrame(lst_dics, columns=['topic','id','word','weight'])
    
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(y="word", x="weight", hue="topic", data=dtf_topics, dodge=False, ax=ax).set_title('Main Topics')
    ax.set(ylabel="", xlabel="Word Importance")
    plt.show()
    return lda_model, dtf_topics



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