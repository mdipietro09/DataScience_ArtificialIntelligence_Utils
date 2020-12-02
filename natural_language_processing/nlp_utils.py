
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
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

## for explainer
from lime import lime_text
import shap

## for W2V and textRank
import gensim
import gensim.downloader as gensim_api

## for bert
import transformers

## for summarization
import rouge



###############################################################################
#                  TEXT ANALYSIS                                              #
###############################################################################
'''
Plot univariate and bivariate distributions.
'''
def plot_distributions(dtf, x, max_cat=20, top=None, y=None, bins=None, figsize=(10,5)):
    ## univariate
    if y is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(x, fontsize=15)
        ### categorical
        if dtf[x].nunique() <= max_cat:
            if top is None:
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            else:   
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            ax.set(ylabel=None)
        ### numerical
        else:
            sns.distplot(dtf[x], hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
            ax.grid(True)
            ax.set(xlabel=None, yticklabels=[], yticks=[])

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
def add_detect_lang(data, column):
    dtf = data.copy()
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
def add_text_length(data, column):
    dtf = data.copy()
    dtf['word_count'] = dtf[column].apply(lambda x: len(nltk.word_tokenize(str(x))) )
    dtf['char_count'] = dtf[column].apply(lambda x: sum(len(word) for word in nltk.word_tokenize(str(x))) )
    dtf['sentence_count'] = dtf[column].apply(lambda x: len(nltk.sent_tokenize(str(x))) )
    dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']
    dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']
    print(dtf[['char_count','word_count','sentence_count','avg_word_length','avg_sentence_lenght']].describe().T[["min","mean","max"]])
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
def add_sentiment(data, column, algo="vader", sentiment_range=(-1,1)):
    dtf = data.copy()
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
    :param lst_add_words: list - list of new stopwords to add
    :param lst_keep_words: list - list words to keep (exclude from stopwords)
:return
    stop_words: list of stop words
'''      
def create_stopwords(lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
    lst_stopwords = set()
    for lang in lst_langs:
        lst_stopwords = lst_stopwords.union( set(nltk.corpus.stopwords.words(lang)) )
    lst_stopwords = lst_stopwords.union(lst_add_words)
    lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
    return sorted(list(set(lst_stopwords)))



'''
Preprocess a string.
:parameter
    :param txt: string - name of column containing text
    :param lst_regex: list - list of regex to remove
    :param punkt: bool - if True removes punctuations and characters
    :param lower: bool - if True convert lowercase
    :param lst_stopwords: list - list of stopwords to remove
    :param stemm: bool - whether stemming is to be applied
    :param lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(txt, lst_regex=None, punkt=True, lower=True, lst_stopwords=None, stemm=False, lemm=True):
    ## regex (in case, before processing)
    if lst_regex is not None: 
        for regex in lst_regex:
            txt = re.sub(regex, '', txt)
    
    ## clean 
    ### separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    ### remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
            
    ## Tokenize (convert from string to list)
    lst_txt = txt.split()

    ## remove Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]
                
    ## Lemmatization (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]

    ## remove leftover Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in lst_stopwords]
            
    ## back to string from list
    txt = " ".join(lst_txt)
    return txt



'''
Adds a column of preprocessed text.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    dtf: input dataframe with two new columns
'''
def add_preprocessed_text(data, column, lst_regex=None, punkt=False, lower=False, lst_stopwords=None, stemm=False, lemm=False, remove_na=True):
    dtf = data.copy()

    ## apply preprocess
    dtf = dtf[ pd.notnull(dtf[column]) ]
    dtf[column+"_clean"] = dtf[column].apply(lambda x: utils_preprocess_text(x, lst_regex, punkt, lower, lst_stopwords, stemm, lemm))
    
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
def word_freq(corpus, ngrams=[1,2,3], top=10, figsize=(10,7)):
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
def add_word_freq(data, column, lst_words, freq="count"):
    dtf = data.copy()

    ## query
    print("found records:")
    print([word+": "+str(len(dtf[dtf[column].str.contains(word)])) for word in lst_words])
    
    ## vectorizer
    lst_grams = [len(word.split(" ")) for word in lst_words]
    if freq == "tfidf":
        vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=lst_words, ngram_range=(min(lst_grams),max(lst_grams)))
    else:
        vectorizer = feature_extraction.text.CountVectorizer(vocabulary=lst_words, ngram_range=(min(lst_grams),max(lst_grams)))
    dtf_X = pd.DataFrame(vectorizer.fit_transform(dtf[column]).todense(), columns=lst_words)
    
    ## join
    for word in lst_words:
        dtf[word] = dtf_X[word]
    return dtf


        
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
                continue

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
def add_ner_spacy(data, column, ner=None, lst_tag_filter=None, grams_join="_", create_features=True):
    ner = spacy.load("en_core_web_lg") if ner is None else ner
    dtf = data.copy()

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
    ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate', 
              ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
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
    
    ## plot heatmap
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(X.todense()[:,np.random.randint(0,X.shape[1],100)]==0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sparse Matrix Sample')
    plt.show()
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
        chi2, p = feature_selection.chi2(X, y==cat)
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
    :param vectorizer: vectorizer object - if None Tf-Idf is used
    :param classifier: model object - if None MultinomialNB is used
:return
    fitted model and predictions
'''
def fit_ml_classif(X_train, y_train, X_test, vectorizer=None, classifier=None): 
    ## model pipeline
    vectorizer = feature_extraction.text.TfidfVectorizer() if vectorizer is None else vectorizer
    classifier = naive_bayes.MultinomialNB() if classifier is None else classifier
    model = pipeline.Pipeline([("vectorizer",vectorizer), ("classifier",classifier)])
    
    ## train
    if vectorizer is None:
        model.fit(X_train, y_train)
    else:
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
    explained.show_in_notebook(text=txt_instance, predict_proba=False)
    dtf_explainer = pd.DataFrame(explained.as_list(), columns=['feature','effect'])
    return dtf_explainer



###############################################################################
#                        WORD2VEC (WORD EMBEDDING)                            #
###############################################################################
'''
Create a list of lists of grams with gensim:
    [ ["hi", "my", "name", "is", "Tom"], 
      ["what", "is", "yours"] ]
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
:return
    lst of lists of n-grams
'''
def utils_preprocess_ngrams(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[]):
    ## create list of n-grams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)
    
    ## detect common bi-grams and tri-grams
    if len(lst_ngrams_detectors) != 0:
        for detector in lst_ngrams_detectors:
            lst_corpus = list(detector[lst_corpus])
    return lst_corpus



'''
Train common bigrams and trigrams detectors with gensim
:parameter
    :param corpus: list - dtf["text"]
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_common_terms: list - ["of","with","without","and","or","the","a"]
    :param min_count: int - ignore all words with total collected count lower than this value
:return
    list with n-grams models and dataframe with frequency
'''
def create_ngrams_detectors(corpus, grams_join=" ", lst_common_terms=[], min_count=5, top=10, figsize=(10,7)):
    ## fit models
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=1, grams_join=grams_join)
    bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=grams_join.encode(), common_terms=lst_common_terms, 
                                                     min_count=min_count, threshold=min_count*2)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=grams_join.encode(), common_terms=lst_common_terms, 
                                                      min_count=min_count, threshold=min_count*2)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    ## plot
    dtf_ngrams = pd.DataFrame([{"word":grams_join.join([gram.decode() for gram in k]), "freq":v} for k,v in trigrams_detector.phrasegrams.items()])
    dtf_ngrams["ngrams"] = dtf_ngrams["word"].apply(lambda x: x.count(grams_join)+1)
    dtf_ngrams = dtf_ngrams.sort_values(["ngrams","freq"], ascending=[True,False])
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=dtf_ngrams.groupby('ngrams')["ngrams","freq","word"].head(top))
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    plt.show()
    return [bigrams_detector, trigrams_detector], dtf_ngrams



'''
Fits the Word2Vec model from gensim.
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
    :param min_count: num - ignores all words with total frequency lower than this
    :param size: num - dimensionality of the vectors
    :param window: num - ( x x x ... x  word  x ... x x x)
    :param sg: num - 1 for skip-grams, 0 for CBOW
    :param lst_common_terms: list - ["of","with","without","and","or","the","a"]
:return
    lst_corpus and the nlp model
'''
def fit_w2v(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], min_count=1, size=300, window=20, sg=1, epochs=30):
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join, lst_ngrams_detectors=lst_ngrams_detectors)
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=size, window=window, min_count=min_count, sg=sg, iter=epochs)
    return lst_corpus, nlp.wv



'''
Creates a feature matrix (num_docs x vector_size)
:parameter
    :param x: string or list
    :param nlp: gensim model
    :param value_na: value to return when the word is not in vocabulary
:return
    vector or matrix 
'''
def embedding_w2v(x, nlp=None, value_na=0):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    null_vec = [value_na]*nlp.vector_size
    
    ## single word --> vec (size,)
    if (type(x) is str) and (len(x.split()) == 1):
        X = nlp[x] if x in nlp.vocab.keys() else null_vec
    
    ## list of words --> matrix (n, size)
    elif (type(x) is list) and (type(x[0]) is str) and (len(x[0].split()) == 1):
        X = np.array([nlp[word] if word in nlp.vocab.keys() else null_vec for word in x])
    
    ## list of lists of words --> matrix (n mean vectors, size)
    elif (type(x) is list) and (type(x[0]) is list):
        lst_mean_vecs = []
        for lst in x:
            lst_mean_vecs.append(np.array([nlp[word] if word in nlp.vocab.keys() else null_vec for word in lst]
                                          ).mean(0))
        X = np.array(lst_mean_vecs)
    
    ## single text --> matrix (n words, size)
    elif (type(x) is str) and (len(x.split()) > 1):
        X = np.array([nlp[word] if word in nlp.vocab.keys() else null_vec for word in x.split()])
    
    ## list of texts --> matrix (n mean vectors, size)
    else:
        lst_mean_vecs = []
        for txt in x:
            lst_mean_vecs.append(np.array([nlp[word] if word in nlp.vocab.keys() else null_vec for word in txt.split()]
                                          ).mean(0))
        X = np.array(lst_mean_vecs)

    return X



'''
Plot words in vector space (2d or 3d).
:parameter
    :param lst_words: list - ["donald trump","china", ...]. If None, it plots the whole vocabulary
    :param nlp: gensim model
    :param plot_type: string - "2d" or "3d"
    :param top: num - plot top most similar words (only if lst_words is given)
    :param annotate: bool - include word text
'''
def plot_w2v(lst_words=None, nlp=None, plot_type="2d", top=20, annotate=True, figsize=(10,5)):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    fig = plt.figure(figsize=figsize)
    if lst_words is not None:
        fig.suptitle("Word: "+lst_words[0], fontsize=12) if len(lst_words) == 1 else fig.suptitle("Words: "+str(lst_words[:5]), fontsize=12)
    else:
        fig.suptitle("Vocabulary")
    try:
        ## word embedding
        tot_words = lst_words + [tupla[0] for tupla in nlp.most_similar(lst_words, topn=top)] if lst_words is not None else list(nlp.vocab.keys())
        X = nlp[tot_words]
        
        ## pca
        pca = manifold.TSNE(perplexity=40, n_components=int(plot_type[0]), init='pca')
        X = pca.fit_transform(X)

        ## create dtf
        columns = ["x","y"] if plot_type == "2d" else ["x","y","z"]
        dtf = pd.DataFrame(X, index=tot_words, columns=columns)
        dtf["input"] = 0
        if lst_words is not None:
            dtf["input"].iloc[0:len(lst_words)] = 1  #<--this makes the difference between vocabulary and input words
        
        ## plot 2d
        if plot_type == "2d": 
            ax = fig.add_subplot()
            sns.scatterplot(data=dtf, x="x", y="y", hue="input", legend=False, ax=ax, palette={0:'black',1:'red'})
            ax.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
            if annotate is True:
                for i in range(len(dtf)):
                    ax.annotate(dtf.index[i], xy=(dtf["x"].iloc[i],dtf["y"].iloc[i]), 
                                xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
        
        ## plot 3d
        elif plot_type == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dtf[dtf["input"]==0]['x'], dtf[dtf["input"]==0]['y'], dtf[dtf["input"]==0]['z'], c="black")
            ax.scatter(dtf[dtf["input"]==1]['x'], dtf[dtf["input"]==1]['y'], dtf[dtf["input"]==1]['z'], c="red")
            ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[], yticklabels=[], zticklabels=[])
            if annotate is True:
                for label, row in dtf[["x","y","z"]].iterrows():
                    x, y, z = row
                    ax.text(x, y, z, s=label)
            
        plt.show()
        
    except Exception as e:
        print("--- got error ---")
        print(e)
        word = str(e).split("'")[1]
        print("maybe you are looking for ... ")
        print([k for k in list(nlp.vocab.keys()) if 1-nltk.jaccard_distance(set(word),set(k)) > 0.7])



'''
Embeds a vocabulary of unigrams with gensim w2v.
:parameter
    :param dic_vocabulary: dict - {"word":1, "word":2, ...}
    :param nlp: gensim model
:return
    Matric and the nlp model
'''
def vocabulary_embeddings(dic_vocabulary, nlp=None):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    embeddings = np.zeros((len(dic_vocabulary)+1, nlp.vector_size))
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
Transforms the corpus into an array of sequences of idx (tokenizer) with same length (padding).
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
    :param fitted_tokenizer: keras tokenizer - if None it creates one with fit and transorm (train set), if given it transforms only (test set)
    :param top: num - if given the tokenizer keeps only top important words
    :param oov: string - how to encode words not in vocabulary (ex. "NAN")
    :param maxlen: num - dimensionality of the vectors, if None takes the max length in corpus
    :param padding: string - "pre" for [9999,1,2,3] or "post" for [1,2,3,9999]
:return
    If training: matrix of sequences, tokenizer, dic_vocabulary. Else matrix of sequences only.
'''
def text2seq(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], fitted_tokenizer=None, top=None, oov=None, maxlen=None):    
    print("--- tokenization ---")
    
    ## detect common n-grams in corpus
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join, lst_ngrams_detectors=lst_ngrams_detectors)

    ## bow with keras to get text2tokens without creating the sparse matrix
    ### train
    if fitted_tokenizer is None:
        tokenizer = kprocessing.text.Tokenizer(num_words=top, lower=True, split=' ', char_level=False, oov_token=oov,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(lst_corpus)
        dic_vocabulary = tokenizer.word_index
        print(len(dic_vocabulary), "words")
    else:
        tokenizer = fitted_tokenizer
    ### transform
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

    ## padding sequence (from [1,2],[3,4,5,6] to [0,0,1,2],[3,4,5,6])
    print("--- padding to sequence ---")
    X = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=maxlen, padding="post", truncating="post")
    print(X.shape[0], "sequences of length", X.shape[1]) 

    ## plot heatmap
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(X==0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sequences Overview')
    plt.show()
    return {"X":X, "tokenizer":tokenizer, "dic_vocabulary":dic_vocabulary} if fitted_tokenizer is None else X



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
def fit_dl_classif(X_train, y_train, X_test, encode_y=False, dic_y_mapping=None, model=None, weights=None, epochs=10, batch_size=256):
    ## encode y
    if encode_y is True:
        dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
        inverse_dic = {v:k for k,v in dic_y_mapping.items()}
        y_train = np.array( [inverse_dic[y] for y in y_train] )
    print(dic_y_mapping)
    
    ## model
    if model is None:
        ### params
        n_features, embeddings_dim = weights.shape
        max_seq_lenght = X_train.shape[1]
        ### neural network
        x_in = layers.Input(shape=(X_train.shape[1],))
        x = layers.Embedding(input_dim=n_features, output_dim=embeddings_dim, weights=[weights], input_length=max_seq_lenght, trainable=False)(x_in)
        x = layers.Attention()([x,x])
        x = layers.Bidirectional(layers.LSTM(units=X_train.shape[1], dropout=0.2))(x)
        x = layers.Dense(units=64, activation='relu')(x)
        y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)
        ### compile
        model = models.Model(x_in, y_out)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        
    ## train
    verbose = 0 if epochs > 1 else 1
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_split=0.3)
    if epochs > 1:
        utils_plot_keras_training(training)
    
    ## test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob] if encode_y is True else [np.argmax(pred)]
    return training.model, predicted_prob, predicted



'''
Takes the weights of an Attention layer and builds an explainer.
:parameter
    :param model: model instance (after fitting)
    :param tokenizer: keras tokenizer (after fitting)
    :param txt_instance: string - raw text
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
    :param top: num - top features to display
:return
    text html, it can be visualized on notebook with display(HTML(text))
'''
def explainer_attention(model, tokenizer, txt_instance, lst_ngrams_detectors=[], top=5, figsize=(5,3)):
    ## preprocess txt_instance
    lst_corpus = utils_preprocess_ngrams([re.sub(r'[^\w\s]', '', txt_instance.lower().strip())], lst_ngrams_detectors=lst_ngrams_detectors)
    X_instance = kprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(lst_corpus), 
                                                    maxlen=int(model.input.shape[1]), padding="post", truncating="post")
    
    ## get attention weights
    layer = [layer for layer in model.layers if "attention" in layer.name][0]
    func = K.function([model.input], [layer.output])
    weights = func(X_instance)[0]
    weights = np.mean(weights, axis=2).flatten()
    
    ## rescale weights, remove null vector, map word-weight
    weights = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(weights).reshape(-1,1)).reshape(-1)
    weights = [weights[n] for n,idx in enumerate(X_instance[0]) if idx != 0]
    dic_word_weigth = {word:weights[n] for n,word in enumerate(lst_corpus[0]) if word in tokenizer.word_index.keys()}

    ## plot
    if len(dic_word_weigth) > 0:
        dtf = pd.DataFrame.from_dict(dic_word_weigth, orient='index', columns=["score"])
        dtf.sort_values(by="score", ascending=True).tail(top).plot(kind="barh", legend=False, figsize=figsize).grid(axis='x')
        plt.show()
    else:
        print("--- No word recognized ---")

    ## return html visualization (yellow:255,215,0 | blue:100,149,237)
    text = []
    for word in lst_corpus[0]:
        weight = dic_word_weigth.get(word)
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
#                        TOPIC MODELING                                       #
###############################################################################
'''
Use Word2Vec to get a list of similar words of a given input words list
:parameter
    :param lst_words: list - input words
    :param top: num - number of words to return
    :param nlp: gensim model
:return
    list with input words + output words
'''
def get_similar_words(lst_words, top, nlp=None):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    lst_out = lst_words
    for tupla in nlp.most_similar(lst_words, topn=top):
        lst_out.append(tupla[0])
    return list(set(lst_out))



'''
Clusters a Word2Vec vocabulary with nltk Kmeans using cosine similarity.
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
    :param n_clusters: num - number of topics to find
:return
    dtf with clusters
'''
def word_clustering(corpus, nlp=None, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], n_clusters=3):
    ## fit W2V
    if nlp is None:
        print("--- training W2V---")
        lst_corpus, nlp = fit_w2v(corpus, ngrams=ngrams, grams_join=grams_join, lst_ngrams_detectors=lst_ngrams_detectors, 
                                  min_count=1, size=300, window=20, sg=0, epochs=30)

    ## fit K-Means
    print("--- training K-means ---")
    X = nlp[nlp.vocab.keys()]
    kmeans_model = nltk.cluster.KMeansClusterer(n_clusters, distance=nltk.cluster.util.cosine_distance, repeats=50, avoid_empty_clusters=True)
    clusters = kmeans_model.cluster(X, assign_clusters=True)
    dic_clusters = {word:clusters[i] for i,word in enumerate(list(nlp.vocab.keys()))}
    dtf_clusters = pd.DataFrame({"word":word, "cluster":str(clusters[i])} for i,word in enumerate(list(nlp.vocab.keys())))
    dtf_clusters = dtf_clusters.sort_values(["cluster", "word"], ascending=[True,True]).reset_index(drop=True)
    return dtf_clusters



'''
Fits Latent Dirichlet Allocation with gensim.
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
    :param n_topics: num - number of topics to find
:return
    model and dtf topics
'''
def fit_lda(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], n_topics=3, figsize=(10,7)):
    ## train the lda
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join, lst_ngrams_detectors=lst_ngrams_detectors)
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
    ax.set(ylabel="", xlabel="importance")
    plt.show()
    return lda_model, dtf_topics



'''
Plot word clusters in vector space (2d or 3d).
:parameter
    :param dic_words: dict - {0:lst_words, 1:lst_words, ...}
    :param nlp: gensim model
    :param plot_type: string - "2d" or "3d"
    :param annotate: bool - include word text
'''
def plot_w2v_cluster(dic_words=None, nlp=None, plot_type="2d", annotate=True, figsize=(10,5)):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Word Clusters", fontsize=12)
    try:
        ## word embedding
        tot_words = [word for v in dic_words.values() for word in v]
        X = nlp[tot_words]
        
        ## pca
        pca = manifold.TSNE(perplexity=40, n_components=int(plot_type[0]), init='pca')
        X = pca.fit_transform(X)

        ## create dtf
        columns = ["x","y"] if plot_type == "2d" else ["x","y","z"]
        dtf = pd.DataFrame()
        for k,v in dic_words.items():
            size = len(dtf) + len(v)
            dtf_group = pd.DataFrame(X[len(dtf):size], columns=columns, index=v)
            dtf_group["cluster"] = k
            dtf = dtf.append(dtf_group)
        
        ## plot 2d
        if plot_type == "2d": 
            ax = fig.add_subplot()
            sns.scatterplot(data=dtf, x="x", y="y", hue="cluster", ax=ax)
            ax.legend().texts[0].set_text(None)
            ax.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
            if annotate is True:
                for i in range(len(dtf)):
                    ax.annotate(dtf.index[i], xy=(dtf["x"].iloc[i],dtf["y"].iloc[i]), 
                                xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
        
        ## plot 3d
        elif plot_type == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            colors = sns.color_palette(None, len(dic_words.keys()))
            for n,k in enumerate(dic_words.keys()):
                ax.scatter(dtf[dtf["cluster"]==k]['x'], dtf[dtf["cluster"]==k]['y'], dtf[dtf["cluster"]==k]['z'], c=colors[n], label=k)
            ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[], yticklabels=[], zticklabels=[])
            ax.legend()
            if annotate is True:
                for label, row in dtf[["x","y","z"]].iterrows():
                    x, y, z = row
                    ax.text(x, y, z, s=label)
        
        plt.show()
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        word = str(e).split("'")[1]
        print("maybe you are looking for ... ")
        print([k for k in list(nlp.vocab.keys()) if 1-nltk.jaccard_distance(set(word),set(k)) > 0.7])



###############################################################################
#                      BERT (TRANSFORMERS LANGUAGE MODEL)                     #
###############################################################################
'''
Word embedding with Bert (equivalent to nlp["word"]).
:parameter
    :param txt: string 
    :param tokenizer: transformers tokenizer
    :param nlp: transformers bert
:return
    tensor sentences x words x vector (1x3x768) 
'''
def utils_bert_embedding(txt, tokenizer, nlp, log=False):
    idx = tokenizer.encode(txt)
    if log is True:
        print("tokens:", tokenizer.convert_ids_to_tokens(idx))
        print("ids   :", tokenizer.encode(txt))
    idx = np.array(idx)[None,:]  
    embedding = nlp(idx)
    X = np.array(embedding[0][0][1:-1])
    return X



'''
Creates a feature matrix (num_docs x vector_size)
:parameter
    :param x: string or list
    :param tokenizer: transformers tokenizer
    :param nlp: transformers bert
    :param log: bool - print tokens
:return
    vector or matrix 
'''
def embedding_bert(x, tokenizer=None, nlp=None, log=False):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased') if tokenizer is None else tokenizer
    nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased') if nlp is None else nlp
    
    ## single word --> vec (size,)
    if (type(x) is str) and (len(x.split()) == 1):
        X = utils_bert_embedding(x, tokenizer, nlp, log).reshape(-1)
    
    ## list of words --> matrix (n, size)
    elif (type(x) is list) and (type(x[0]) is str) and (len(x[0].split()) == 1):
        X = utils_bert_embedding(x, tokenizer, nlp, log)
    
    ## list of lists of words --> matrix (n mean vectors, size)
    elif (type(x) is list) and (type(x[0]) is list):
        lst_mean_vecs = [utils_bert_embedding(lst, tokenizer, nlp, log).mean(0) for lst in x]
        X = np.array(lst_mean_vecs)
    
    ## single text --> matrix (n words, size)
    elif (type(x) is str) and (len(x.split()) > 1):
        X = utils_bert_embedding(x, tokenizer, nlp, log)
        
    ## list of texts --> matrix (n mean vectors, size)
    else:
        lst_mean_vecs = [utils_bert_embedding(txt, tokenizer, nlp, log).mean(0) for txt in x]
        X = np.array(lst_mean_vecs)
    return X



'''
Preprocess corpus to create features for Bert.
:parameter
    :param corpus: list - dtf["text"]
    :param tokenizer: transformer tokenizer
    :param maxlen: num - max length of the padded sequence 
:return
    tensor/list with idx, masks, segments
'''
# def tokenize_bert(corpus, tokenizer=None, maxlen=None):
#     tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) if tokenizer is None else tokenizer
#     maxlen = np.max([len(i.split()) for i in corpus]) if maxlen is None else maxlen
#     idx, masks, types = [],[],[]
#     for txt in corpus:
#         dic_tokens = tokenizer.encode_plus(txt, add_special_tokens=True, max_length=maxlen)
#         idx.append(dic_tokens['input_ids'])
#         masks.append(dic_tokens['special_tokens_mask'])
#         types.append(dic_tokens['token_type_ids'])        
#     return [np.asarray(idx, dtype='int32'), np.asarray(masks, dtype='int32'), np.asarray(types, dtype='int32')]

def tokenize_bert(corpus, tokenizer=None, maxlen=None):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) if tokenizer is None else tokenizer
    maxlen = np.max([len(txt.split(" ")) for txt in corpus]) if maxlen is None else maxlen
    if maxlen < 20:
        raise Exception("maxlen cannot be less than 20")
    else:
        print("maxlen:", maxlen)

    ## add special tokens: [CLS] my name is mau ##ro [SEP]
    maxqnans = np.int((maxlen-20)/2)
    corpus_tokenized = ["[CLS] "+
                        " ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '', str(txt).lower().strip()))[:maxqnans])+
                        " [SEP] " for txt in corpus]
   
    ## generate masks: [1, 1, 1, 1, 1, 1, 1, | (padding) 0, 0, 0, 0, 0, ...]
    masks = [[1]*len(txt.split(" ")) + [0]*(maxlen - len(txt.split(" "))) for txt in corpus_tokenized]
    
    ## padding
    #corpus_tokenized = kprocessing.sequence.pad_sequences(corpus_tokenized, maxlen=maxlen, dtype=object, value='[PAD]')
    txt2seq = [txt + " [PAD]"*(maxlen-len(txt.split(" "))) if len(txt.split(" ")) != maxlen else txt for txt in corpus_tokenized]
    
    ## generate idx: [101, 22, 35, 44, 50, 60, 102, 0, 0, 0, 0, 0, 0, ...]
    idx = [tokenizer.encode(seq.split(" ")) for seq in txt2seq]
    
    ## generate segments: [0, 0, 0, 0, 0, 0, 1 [SEP], 0, 0, 0, 0, 2 [SEP], 0, ...]
    segments = [] 
    for seq in txt2seq:
        temp, i = [], 0
        for token in seq.split(" "):
            temp.append(i)
            if token == "[SEP]":
                i += 1
        segments.append(temp)
    
    ## check
    genLength = set([len(seq.split(" ")) for seq in txt2seq])
    if len(genLength) != 1: 
        print(genLength)
        raise Exception("--- texts are not of same size ---")

    X = [np.asarray(idx, dtype='int32'), np.asarray(masks, dtype='int32'), np.asarray(segments, dtype='int32')]
    print("created tensor idx-masks-segments:", str(len(X))+"x "+str(X[0].shape))
    return X



'''
Pre-trained Bert + Fine-tuning (transfer learning) with tf2 and transformers.
:parameter
    :param X_train: array of sequence
    :param y_train: array of classes
    :param X_test: array of sequence
    :param model: model object - model to fit (before fitting)
    :param encode_y: bool - whether to encode y with a dic_y_mapping
    :param dic_y_mapping: dict - {0:"A", 1:"B", 2:"C"}. If None it calculates
    :param epochs: num - epochs to run
    :param batch_size: num - it does backpropagation every batch, the more the faster but it can use all the memory
:return
    model fitted and predictions
'''
def fit_bert_classif(X_train, y_train, X_test, encode_y=False, dic_y_mapping=None, model=None, epochs=1, batch_size=64):
    ## encode y
    if encode_y is True:
        dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
        inverse_dic = {v:k for k,v in dic_y_mapping.items()}
        y_train = np.array( [inverse_dic[y] for y in y_train] )
    print(dic_y_mapping)
    
    ## model
    if model is None:
        ### inputs
        idx = layers.Input((X_train[0].shape[1]), dtype="int32", name="input_idx")
        masks = layers.Input((X_train[1].shape[1]), dtype="int32", name="input_masks")
        segments = layers.Input((X_train[2].shape[1]), dtype="int32", name="input_segments")
        ### pre-trained bert
        bert = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        bert_out, _ = bert([idx, masks, segments])
        ### fine-tuning
        x = layers.GlobalAveragePooling1D()(bert_out)
        x = layers.Dense(64, activation="relu")(x)
        y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)
        ### compile
        model = models.Model([idx, masks, segments], y_out)
        for layer in model.layers[:4]:
            layer.trainable = False
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        
    ## train
    verbose = 0 if epochs > 1 else 1
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_split=0.3)
    if epochs > 1:
        utils_plot_keras_training(training)
    
    ## test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob] if encode_y is True else [np.argmax(pred)]
    return training.model, predicted_prob, predicted



###############################################################################
#               UNSEPERVISED CLASSIFICATION BY SIMILARITY                     #
###############################################################################
'''
Compute cosine similarity between 2 strings or 2 vectors/matrices: cosine_sim = matrix (rows_a x rows_b)
:parameter
    :param a: string, vector, or matrix
    :param b: string, vector, or matrix
    :param nlp: gensim model - used only if a and b are strings
:return
    cosine similarity score or matrix
'''
def utils_cosine_sim(a, b, nlp=None):
    ## string vs string = score
    if (type(a) is str) or (type(b) is str):
        nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
        cosine_sim = nlp.similarity(a,b)
        
    else:
        ## vector vs vector = score
        if (len(a.shape) == 1) and (len(a.shape) == 1):
            a = a.reshape(1,-1)
            b = b.reshape(1,-1)
            cosine_sim = metrics.pairwise.cosine_similarity(a, b)[0][0]  #np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        
        ## matrix vs matrix = matrix (rows_a x rows_b)
        else:
            a = a.reshape(1,-1) if len(a.shape) == 1 else a
            b = b.reshape(1,-1) if len(b.shape) == 1 else b
            cosine_sim = metrics.pairwise.cosine_similarity(a, b)
    return cosine_sim



'''
Clustering of text to specific classes (Unsupervised Classification by similarity).
:parameter
    :param X: feature matrix (num_docs x vector_size)
    :param dic_y: dic label:mean_vector - {'finance':mean_vec, 'esg':mean_vec}
:return
    predicted_prob, predicted
'''
def predict_similarity_classif(X, dic_y):
    predicted_prob = np.array([utils_cosine_sim(X, y).T.tolist()[0] for y in dic_y.values()]).T
    labels = list(dic_y.keys())
    
    ## adjust and rescale
    for i in range(len(predicted_prob)):
        ### assign randomly if there is no similarity
        if sum(predicted_prob[i]) == 0:
            predicted_prob[i] = [0]*len(labels)
            predicted_prob[i][np.random.choice(range(len(labels)))] = 1
        ### rescale so they sum=1
        predicted_prob[i] = predicted_prob[i] / sum(predicted_prob[i])
    
    predicted = [labels[np.argmax(pred)] for pred in predicted_prob]
    return predicted_prob, predicted



'''
Plot a text instance into a 2d vector space and compute similarity.
:parameter
    :param tokenizer: transformers tokenizer
    :param nlp: transformers bert
    :param dic_clusters: dict - dict - {0:lst_words, 1:lst_words, ...}
    :param txt_instance: string - raw text
    :param token_level: bool - if True the text is broken down into tokens otherwise the mean vector is taken
    :param top: num - top similarity to display
'''
def explainer_similarity_classif(tokenizer, nlp, dic_clusters, txt_instance, token_level=False, top=5, figsize=(20,10)):
    ## create embedding Matrix
    y = np.concatenate([embedding_bert(v, tokenizer, nlp) for v in dic_clusters.values()])
    X = embedding_bert(txt_instance, tokenizer, nlp) if token_level is True else embedding_bert(txt_instance, tokenizer, nlp).mean(0).reshape(1,-1)
    M = np.concatenate([y,X])
    
    ## pca
    pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
    M = pca.fit_transform(M)
    y, X = M[:len(y)], M[len(y):]
    
    ## create dtf clusters
    dtf = pd.DataFrame()
    for k,v in dic_clusters.items():
        size = len(dtf) + len(v)
        dtf_group = pd.DataFrame(y[len(dtf):size], columns=["x","y"], index=v)
        dtf_group["cluster"] = k
        dtf = dtf.append(dtf_group)
        
    ## plot clusters
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=dtf, x="x", y="y", hue="cluster", ax=ax)
    ax.legend().texts[0].set_text(None)
    ax.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    for i in range(len(dtf)):
        ax.annotate(dtf.index[i], xy=(dtf["x"].iloc[i],dtf["y"].iloc[i]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
    
    ## add txt_instance
    if token_level is True:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(txt_instance))[1:-1]
        dtf = pd.DataFrame(X, columns=["x","y"], index=tokens)
        dtf = dtf[~dtf.index.str.contains("#")]
        dtf = dtf[dtf.index.str.len() > 1]
        X = dtf.values
        ax.scatter(x=dtf["x"], y=dtf["y"], c="red")
        for i in range(len(dtf)):
            ax.annotate(dtf.index[i], xy=(dtf["x"].iloc[i],dtf["y"].iloc[i]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
    else:
        ax.scatter(x=X[0][0], y=X[0][1], c="red", linewidth=10)
        ax.annotate("x", xy=(X[0][0],X[0][1]), ha='center', va='center', fontsize=25)
    
    ## calculate similarity
    sim_matrix = utils_cosine_sim(X,y) 

    ## add top similarity
    for row in range(sim_matrix.shape[0]):
        ### sorted {keyword:score}
        dic_sim = {n:sim_matrix[row][n] for n in range(sim_matrix.shape[1])}
        dic_sim = {k:v for k,v in sorted(dic_sim.items(), key=lambda item:item[1], reverse=True)}
        ### plot lines
        for k in dict(list(dic_sim.items())[0:top]).keys():
            p1 = [X[row][0], X[row][1]]
            p2 = [y[k][0], y[k][1]]
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], c="red", alpha=0.5)
    plt.show()



###############################################################################
#                  STRING MATCHING                                            #
###############################################################################
'''
Matches strings with cosine similarity.
:parameter
    :param a: string - ex. "my house"
    :param lst_b: list of strings - ex. ["my", "hi", "house", "sky"]
    :param threshold: num - similarity threshold to consider the match valid
    :param top: num - number of matches to return
:return
    dtf with 1 column = a, index = lst_b, values = cosine similarity scores
'''
def utils_string_matching(a, lst_b, threshold=None, top=None):
    ## vectorizer ("my house" --> ["my", "hi", "house", "sky"] --> [1, 0, 1, 0])
    vectorizer = feature_extraction.text.CountVectorizer()
    X = vectorizer.fit_transform([a]+lst_b).toarray()

    ## cosine similarity (scores a vs lst_b)
    lst_vectors = [vec for vec in X]
    cosine_sim = metrics.pairwise.cosine_similarity(lst_vectors)
    scores = cosine_sim[0][1:]

    ## match
    match_scores = scores if threshold is None else scores[scores >= threshold]
    match_idxs = range(len(match_scores)) if threshold is None else [i for i in np.where(scores >= threshold)[0]] 
    match_strings = [lst_b[i] for i in match_idxs]

    ## dtf
    dtf_match = pd.DataFrame(match_scores, columns=[a], index=match_strings)
    dtf_match = dtf_match[~dtf_match.index.duplicated(keep='first')].sort_values(a, ascending=False).head(top)
    return dtf_match



'''
Vlookup for similar strings.
:parameter
    :param lst_left - array or lst
    :param lst_right - array or lst
    :param threshold: num - similarity threshold to consider the match valid
    :param top: num or None - number of matches to return
:return
    dtf_matches - dataframe with matches
'''
def vlookup(lst_left, lst_right, threshold=0.7, top=1):
    try:
        dtf_matches = pd.DataFrame(columns=['string','match','similarity'])
        for string in lst_left:
            dtf_match = utils_string_matching(string, lst_right, threshold, top)
            dtf_match = dtf_match.reset_index().rename(columns={'index':'match', string:'similarity'})
            dtf_match["string"] = string
            for i in range(len(dtf_match)):
                print(string, " --", round(dtf_match["similarity"].values[i], 2), "--> ", dtf_match["match"].values[i])
            dtf_matches = dtf_matches.append(dtf_match, ignore_index=True, sort=False)
        return dtf_matches[['string','match','similarity']]

    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Highlights the matched strings in text.
:parameter
    :param a: string - raw text
    :param b: string - raw text
    :param both: bool - search a in b and, if True, viceversa
    :param sentences: bool - if True search only strings
:return
    text html, it can be visualized on notebook with display(HTML(text))
'''
def display_string_matching(a, b, both=True, sentences=True, titles=[]):
    if sentences is True:
        lst_a, lst_b = nltk.sent_tokenize(a), nltk.sent_tokenize(b)
    else:
        lst_a, lst_b = a.split(), b.split()       
    
    ## highlight a
    first_text = []
    for i in lst_a:
        if i.lower() in [z.lower() for z in lst_b]:
            first_text.append('<span style="background-color:rgba(255,215,0,0.3);">' + i + '</span>')
        else:
            first_text.append(i)
    first_text = ' '.join(first_text)
    
    ## highlight b
    second_text = []
    if both is True:
        for i in lst_b:
            if i in [z.lower() for z in lst_a]:
                second_text.append('<span style="background-color:rgba(255,215,0,0.3);">' + i + '</span>')
            else:
                second_text.append(i)
    else:
        second_text.append(b) 
    second_text = ' '.join(second_text)
    
    ## concatenate
    if len(titles) > 0:
        first_text = "<strong>"+titles[0]+"</strong><br>"+first_text
    if len(titles) > 1:
        second_text = "<strong>"+titles[1]+"</strong><br>"+second_text
    else:
        second_text = "---"*65+"<br><br>"+second_text
    final_text = first_text +'<br><br>'+ second_text
    return final_text



###############################################################################
#                     TEXT SUMMARIZATION                                      #
###############################################################################
'''
Calculate ROUGE score.
:parameter
    :param y_test: string or list
    :param predicted: string or list
'''
def evaluate_summary(y_test, predicted):
    rouge_score = rouge.Rouge()
    scores = rouge_score.get_scores(y_test, predicted, avg=True)
    score_1 = round(scores['rouge-1']['f'], 2)
    score_2 = round(scores['rouge-2']['f'], 2)
    score_L = round(scores['rouge-l']['f'], 2)
    print("rouge1:", score_1, "| rouge2:", score_2, "| rougeL:", score_2, 
          "--> avg rouge:", round(np.mean([score_1,score_2,score_L]), 2))



'''
Summarizes corpus with TextRank.
:parameter
    :param corpus: list - dtf["text"]
    :param ratio: length of the summary (ex. 20% of the text)
:return
    list of summaries
'''
def textrank(corpus, ratio=0.2):
    lst_summaries = [gensim.summarization.summarize(txt, ratio=ratio) for txt in corpus]
    return lst_summaries



'''
Summarizes corpus with Bart.
:parameter
    :param corpus: list - dtf["text"]
    :param ratio: length of the summary (ex. 20% of the text)
:return
    list of summaries
'''
def bart(corpus, ratio=0.2):
    nlp = transformers.pipeline("summarization")
    lst_summaries = [nlp(txt, max_length=int(len(txt.split())*ratio), 
                              min_length=int(len(txt.split())*ratio)
                        )[0]["summary_text"].replace(" .", ".")
                     for txt in corpus]
    return lst_summaries