# setup
runfile('C:/profili/u382270/Downloads/MyStuff/py/DataScience_Utils/deep_learning_natural_language_processing/nlp_utils.py', wdir='C:/profili/u382270/Downloads/MyStuff/py/DataScience_Utils/deep_learning_natural_language_processing')
datapath = "c:/profili/u382270/Downloads/MyStuff/data/"
modelpath = "c:/profili/u382270/Downloads/MyStuff/data/models/"


# data
dtf = pd.read_excel(datapath+"/txt/dtf_news.xlsx")


# test similarity
a = "Il tuo computer Ã¨ bloccato, paga"". Tutto falso, Ã¨ una truffa"
b = "Il tuo computer Ã¨ stato bloccato"" uova allerta per frode informatica"
strings_similarity(a, b, algo="jaccard")


# word2vec




# lda