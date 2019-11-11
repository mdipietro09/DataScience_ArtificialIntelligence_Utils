# setup
runfile('C:/profili/u382270/Downloads/MyStuff/py/DataScience_Utils/deep_learning_natural_language_processing/nlp_utils.py', wdir='C:/profili/u382270/Downloads/MyStuff/py/DataScience_Utils/deep_learning_natural_language_processing')
datapath = "c:/profili/u382270/Downloads/MyStuff/data/"
modelpath = "c:/profili/u382270/Downloads/MyStuff/data/models/"


# data
dtf = pd.read_excel(datapath+"/txt/dtf_news.xlsx")
dtf_left = pd.read_excel(datapath+"/txt/test_left.xlsx")
dtf_right = pd.read_excel(datapath+"/txt/test_right.xlsx")


# word2vec
wordvec = io_Word2Vec(modelpath, modelfile="GoogleWord2Vec.bin.gz", model=None)
str_word = "virtualised_servers"
wordvec.most_similar(str_word)
dic_out = predict_Word2Vec(wordvec, str_word, top=20, plot="2d", figsize=(20,13))


# match strings
a = "Il tuo computer Ã¨ bloccato, paga"". Tutto falso, Ã¨ una truffa"
b = "Il tuo computer Ã¨ stato bloccato"" uova allerta per frode informatica"
strings_similarity(a, b, algo="jaccard")

lst_left = list(set(dtf_left.iloc[:,0].tolist()))
lst_right = list(set(dtf_right.iloc[:,0].tolist()))
dtf_matched = vlookup(lst_left, lst_right, algo="cosine", threshold=0.7, top=3)