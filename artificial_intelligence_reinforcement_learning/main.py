# setup -----------------------------------------------------------------------
dirpath = "c:/profili/u382270/Downloads/MyStuff/data/"
modelpath = "c:/profili/u382270/Downloads/MyStuff/data/models/"



# ML --------------------------------------------------------------------------
import pandas as pd
dirpath = "c:/profili/u382270/Downloads/MyStuff/py/glovo/data/"

dtf = pd.read_csv(dirpath+"Courier_lifetime_data.csv", converters={'courier':str})

dtf = pd.read_csv(dirpath+"Courier_weekly_data.csv", converters={'courier':str,'week':str})


dtf = pd.read_csv(dirpath+"feature_matrix.csv", converters={'courier':str, 'week':str, "pk":str})
dtf = dtf.drop(["courier", "week"], axis=1)

dict_out = data_preprocessing(dtf, pk="pk", y="Y", processNas=None, processCategorical=None, split=0.2, scale=None)
X_train, X_test = dict_out["X"]
Y_train, Y_test = dict_out["Y"]
X_names = dict_out["X_names"]

from sklearn import linear_model
model = linear_model.LogisticRegression(random_state=0)
dict_out = fit_model(model, X_train, Y_train, X_test, Y_test)
model = dict_out["model"]



# strings match ---------------------------------------------------------------
import pandas as pd
dirpath = "c:/profili/u382270/Downloads/cib/"

anagrafica = pd.read_excel(dirpath+"Anagrafica_CIB_tickers_v2.xlsx")

reuters = pd.read_excel(dirpath+"Anagrafica_Prospect_TMT_v0.xlsx")

col_anagrafica_gruppo = "ticker_Reuters_Gruppo"
col_anagrafica_societa = "ticker_Reuters_Societa"
lookup_array = list(set(anagrafica[col_anagrafica_gruppo].values)) + list(set(anagrafica[col_anagrafica_societa].values))

col_reuters = "Quote Symbol"
dtf_out = vlookup(list(set(reuters[col_reuters].values)), lookup_array, algo="gestalt", threshold=0.85)

dtf_out.to_excel(dirpath+"file_match.xlsx", index=False)



# NLP -------------------------------------------------------------------------
## Word2Vec
model = io_Word2Vec(modelpath, modelfile="GoogleWord2Vec.bin.gz", model=None)

dtf = pd.read_excel(dirpath+"/txt/dtf_news.xlsx")
dtf = dtf[dtf["language"]=="en"]
stop_words = create_stopwords(lst_langs=["english"], lst_new_words=[])
lst_corpus = text_preprocessing_ListOfDocs(dtf, "text", stop_words, flg_stemm=False, flg_lemm=False)
lst_lst_corpus = text_preprocessing_ListOfListsOfWords(lst_corpus, ngrams=1)
dic = fit_Word2Vec(lst_lst_corpus, min_count=20, size=100, window=20, sg=0, plot=False)
model = dic["model"]
io_Word2Vec(modelpath, modelfile="news_word2vec", model=model)

str_word = "highway"
predict_Word2Vec(model, str_word, top=20, plot="2d", figsize=(20,13))


## lda
dic_out = topic_modeling_lda(lst_lst_corpus, n_topics=10, n_words=5, plot=True, figsize=(10,10))



# IMG -------------------------------------------------------------------------
## cnn
lst_dogs = load_imgs("c:/profili/u382270/Downloads/MyStuff/data/img/dogs_and_cats/Dog/")
lst_cats = load_imgs("c:/profili/u382270/Downloads/MyStuff/data/img/dogs_and_cats/Cat/")

dic_yX = {1:lst_dogs, 0:lst_cats}
dic_Xy = imgs_preprocessing(dic_yX, size=224, remove_color=False, y_binary=True)
X, y = dic_Xy["X"], dic_Xy["y"]
model = fit_cnn(X, y, batch_size=32, epochs=100, figsize=(20,13))

predict_cnn(load_img(dirpath, "dog.jpg", figsize=(20,13)), model, size=224, 
            remove_color=False, dic_mapp_y={1:"dog", 0:"cat"})


## transfer learning
model = transfer_learning(X, y, batch_size=32, epochs=100, modelname="MobileNet", layers_in=6, figsize=(20,13))
predict_cnn(load_img(dirpath, "dog.jpg", figsize=(20,13)), 
            model, size=224, remove_color=False, dic_mapp_y={1:"dog", 0:"cat"})


## pre-trained models
file = "narda.jpeg"
img = load_img(dirpath, file, figsize=(20,13))
img_classification_keras(img, modelname="MobileNet")
obj_detection_imageai(img, modelpath)


## ocr
from_img_to_txt(load_img(dirpath, "img_text.png", plot=False), 
                modelpath, modelfile="Tesseract-OCR/tesseract.exe", plot=True, figsize=(20,13), lang="eng")



# TS --------------------------------------------------------------------------
import pandas_datareader as web
import datetime
dtf = web.DataReader(name="UCG.MI", data_source="yahoo", 
                     start=datetime.datetime(year=2010, month=12, day=31), 
                     end=datetime.datetime(year=2018, month=12, day=31), 
                     retry_count=10, access_key="CeHargBS3F69P_Pyxsh8")
ts = dtf["Close"]
dtf = web.DataReader(name="UCG.MI", data_source="yahoo", 
                     start=datetime.datetime(year=2018, month=12, day=31), 
                     end=datetime.datetime.now(), 
                     retry_count=10, access_key="CeHargBS3F69P_Pyxsh8")
new_ts = dtf["Close"]


## forecast
plot_ts(ts, plot_ma=True, plot_intervals=True, window=30, figsize=(20,13))
plot_2_ts(ts, dtf["Volume"], figsize=(20,13))

dic_decomposition = decompose_ts(ts, freq=250, figsize=(20,13))
plot_acf_pacf(ts, lags=30)
a = diff_ts(ts, shifts=1, na="fill")
plot_acf_pacf(a, lags=30)

dtf_arima = fit_arima(new_ts, d=1, D=1, s=60, max_integration=3, figsize=(20,13))


## lstm
dic = ts_preprocessing(ts, size=20)
X, y, scaler = dic["X"], dic["y"], dic["scaler"]
model = fit_lstm(X, y, batch_size=32, epochs=100, figsize=(20,13))
dtf_pred = predict_lstm(new_ts, model, scaler, size=20, ahead=10, figsize=(20,13))



# AI --------------------------------------------------------------------------
#classic_control	CartPole-v1			    si
#classic_control	MountainCar-v0		    si
#box2d				BipedalWalkerHardcore-v2		no (pip install box2d-py ---> SWIG ????)
#box2d				CarRacing-v0					no
#toy_text			Blackjack-v0			si
#toy_text			Roulette-v0				si
#mujoco				HumanoidStandup-v2      		no (pip install mujoco_py ---> mujoco_py per windows c'è una vecchia versione, MuJoCo è a pagamento)
#mujoco				Humanoid-v3						no
#atari				Boxing-ram-v4					no (pip install gym[atari] ---> C++ ????)
#atari				Boxing-v4


## env
import numpy as np
import gym
dtf_envs = get_gym_envs(["frozen"])
env_name = "MountainCar-v0"
env = gym.make(env_name)
dic_env_space = get_env_space(env)
dtf_random_play = env_random_play(env, episodes=10)


## init
ai = agentQ(states_space=dic_env_space["states_space"], actions_space=dic_env_space["actions_space"])
ai = agentDQN(states_space=dic_env_space["states_space"], actions_space=dic_env_space["actions_space"])
ai = agentDDQN(states_space=dic_env_space["states_space"], actions_space=dic_env_space["actions_space"])


### train / test
dtf_logs = trainAI(env, ai, episodes=5000, min_eploration=0.001, plot_rate=1000)
log = dtf_logs[dtf_logs["episode"]==5000]
testAI(env, ai, start_state=env.reset())



# DEPLOY ----------------------------------------------------------------------
## App.py
create_app(host="127.0.0.1", port="5000", threaded=False)
# http://localhost:5000/predict?data=[1,2,3]
