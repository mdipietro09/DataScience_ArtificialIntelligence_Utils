
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import utils, models, layers, applications
from imageai import Prediction, Detection
from PIL import Image
import pytesseract



###############################################################################
#                   IMG ANALYSIS                                              #
###############################################################################
'''
'''
def plot_img(img, title=None, figsize=(20,13)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=20)
    plt.imshow(img)

    

'''
'''
def plot_2_imgs(img1, img2, title=None, figsize=(20,13)):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
    fig.suptitle(title, fontsize=20)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show()
   
    
    
'''
'''
def load_img(dirpath, file, ext=['.png','.jpg','.jpeg'], plot=True, figsize=(20,13)):
    if file.endswith(tuple(ext)):
        img = cv2.imread( dirpath+file, cv2.IMREAD_UNCHANGED )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if plot == True:
        plot_img(img, figsize=figsize)
    return img
   
    

'''
'''
def load_imgs(dirpath):
    lst_imgs =[]
    for img in os.listdir(dirpath):
        try:
            t_img = load_img(dirpath=dirpath, file=img, plot=False)
            lst_imgs.append(t_img)
        except Exception as e:
            print(e)
            pass
    return lst_imgs



'''
'''
def single_img_preprocessing(img, size=224, remove_color=False, plot=False, figsize=(20,13)):
    try:
        ## original ---> height x width x RGBchannels(3)
        if plot == True:
            plot_img(img, figsize=figsize, title="original")
            print(img.shape)
        
        ## resize
        img_processed = cv2.resize(img, (size,size), interpolation=cv2.INTER_LINEAR)
        if plot == True:
            plot_img(img_processed, figsize=figsize, title="resized")
            print(img_processed.shape)
        
        ## remove color
        if remove_color == True:
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2GRAY)
            if plot == True:
                plot_img(img_processed, figsize=figsize, title="grayscale")
                print(img_processed.shape)
        
        ## denoise (blur)
        img_processed = cv2.GaussianBlur(img_processed, (5,5), 0)
        if plot == True:
            plot_img(img_processed, figsize=figsize, title="blurred")
            print(img_processed.shape)
        
        ## scale
        img_processed = img_processed/255
        if plot == True:
            plot_img(img_processed, figsize=figsize, title="scaled")
            print(img_processed.shape)
        
        ## output
        if plot == True:
            plot_2_imgs(img1=img, img2=img_processed, figsize=figsize)
        return img_processed

    except Exception as e:
        print("--- got error ---")
        print(e)



###############################################################################
#                      CNN                                                    #
###############################################################################
'''
'''
def imgs_preprocessing(dic_yX, size=224, remove_color=False, y_binary=True):
    try:
        ## list
        lst_y = []
        lst_X = []
        for label in dic_yX.keys():
            ### preprocessing
            lst_imgs = [single_img_preprocessing(img, plot=False, remove_color=remove_color, size=size) for img in dic_yX[label]]      
            ### partitioning
            lst_X = lst_X + lst_imgs
            lst_y = lst_y + [label]*len(lst_imgs)
            print(label, "--> n:", len(lst_imgs))        
        ## tensor
        X = np.array(lst_X)  #(n, size, size, channels=rgb)
        y = np.array(lst_y)  #(n, )
        if y_binary is not True:
            y = utils.to_categorical(y)
        return {"X":X, "y":y}
    
    except Exception as e:
        print("--- got error ---")
        print(e)

    

'''
'''
def fit_cnn(X, y, batch_size=32, epochs=100, figsize=(20,13)):
    ## cnn
    ### layer 1 conv 5x5 (32 neurons) + pool 2x2
    model = models.Sequential()
    model.add( layers.Conv2D(input_shape=X.shape[1:], kernel_size=(5,5), filters=32, activation='relu') )
    model.add( layers.MaxPooling2D(pool_size=(2,2)) )
    ### layer 2 conv 3x3 (64 neurons) + pool 2x2
    model.add( layers.Conv2D(kernel_size=(3,3), filters=64, activation='relu') )
    model.add( layers.MaxPooling2D(pool_size=(2,2)) )
    ### layer 3 fully connected (128 neuroni)
    model.add( layers.Flatten() )
    model.add( layers.Dense(units=128, activation="relu") )
    model.add( layers.Dropout(rate=0.5) )
    ### layer output (n_classes neurons)
    if len(y.shape) == 1:
        print("y binary --> using 1 neuron with 'sigmoid' activation and 'binary_crossentropy' loss")
        model.add( layers.Dense(units=1, activation="sigmoid") )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        print("y multiclasses --> using", y.shape[1], "neurons with 'softmax' activation and 'categorical_crossentropy' loss")
        model.add( layers.Dense(units=y.shape[1], activation="softmax") )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    ## fit
    training = model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, shuffle=True)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=figsize)
    ax[0].plot(training.history['acc'], label='accuracy')
    ax[1].plot(training.history['loss'], label='loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    print(training.model.summary())
    return training.model



'''
'''
def evaluate_cnn(model, X_test, Y_test):
    return 0



'''
'''
def predict_cnn(img, model, size=224, remove_color=False, dic_mapp_y=None):
    img_preprocessed = single_img_preprocessing(img, size=size, remove_color=remove_color, plot=False)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    pred = model.predict( img_preprocessed )        
    return dic_mapp_y[int(pred[0][0])] if dic_mapp_y is not None else int(pred[0][0])



###############################################################################
#                   TRANSFER LEARNING                                         #
###############################################################################
'''
'''
def transfer_learning(X, y, batch_size=32, epochs=100, modelname="MobileNet", layers_in=6, figsize=(20,13)):
    ## load pre-trained model
    if modelname == "ResNet50":
        model = applications.resnet50.ResNet50(weights='imagenet')          
    elif modelname == "MobileNet":
        model = applications.mobilenet.MobileNet()
        
    ## check input shapes
    print(modelname, "requires input of shape", model.input.shape[1:])
    print("X shape: ", X[0].shape)
    if X[0].shape != model.input.shape[1:]:
        print("Stopped for shape error")
        return False     
    
    ## add layer output to a selected hidden layer
    hidden_layer = model.layers[-layers_in].output
    if len(y.shape) == 1:
        print("y binary --> using 1 neuron with 'sigmoid' activation and 'binary_crossentropy' loss")
        output = layers.Dense(1, activation="sigmoid")(hidden_layer)
    else:
        print("y multiclasses --> using", y.shape[1], "neurons with 'softmax' activation and 'categorical_crossentropy' loss")
        output = layers.Dense(y.shape[1], activation="softmax")(hidden_layer)    
    new_model = models.Model(inputs=model.input, outputs=output)
    
    ## specify the layers that must be re-trained (transfer learning as we keep the weights of the other layers)
    for layer in new_model.layers[:-(layers_in-1)]:
        layer.trainable = False
    
    ## train
    if len(y.shape) == 1:
        new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        new_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    training = new_model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, shuffle=True)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=figsize)
    ax[0].plot(training.history['acc'], label='accuracy')
    ax[1].plot(training.history['loss'], label='loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    print(training.model.summary())
    return training.model   



###############################################################################
#                IMG CLASSIFICATION                                           #
###############################################################################
'''
'''
def img_classification_keras(img, modelname="MobileNet"):
    try:
        ## prepare data and call model
        x = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
        x = np.expand_dims(x, axis=0)
        
        if modelname == "ResNet50":
            model = applications.resnet50.ResNet50(weights='imagenet')
            decode_preds = applications.resnet50.decode_predictions
            x = applications.resnet50.preprocess_input(x)
            
        elif modelname == "MobileNet":
            model = applications.mobilenet.MobileNet()
            decode_preds = applications.mobilenet.decode_predictions
            x = applications.mobilenet.preprocess_input(x)
        
        ## predict
        preds = model.predict(x)
        lst_preds = decode_preds(preds, top=3)[0]
        return lst_preds
    
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
'''
def img_classification_imageai(img, modelpath, modelfile="resnet50_weights_tf_dim_ordering_tf_kernels.h5"):
    try:
        ## load imageAI model ResNet50 
        model = Prediction.ImagePrediction()        
        model.setModelTypeAsResNet()
        model.setModelPath( modelpath + modelfile )
        model.loadModel()
        
        ## predict
        lst_preds, lst_probs = model.predictImage(img, input_type="array", result_count=5)
        lst_out = [ (pred, prob) for pred, prob in zip(lst_preds, lst_probs) ]
        return lst_out
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        
        

###############################################################################
#                  OBJECT DETECTION                                           #
###############################################################################
'''
'''
def obj_detection_imageai(img, modelpath, modelfile="yolo.h5", plot=True, figsize=(20,13)):
    try:
        ## load imageAI model YOLO
        model = Detection.ObjectDetection()
        model.setModelTypeAsYOLOv3()
        model.setModelPath( modelpath + modelfile )
        model.loadModel()
        
        ## detect
        detected_img, lst_detections = model.detectCustomObjectsFromImage(input_image=img, input_type="array", 
                                                                          output_type="array",
                                                                          minimum_percentage_probability=30)
        lst_out = [ (dic["name"], dic["percentage_probability"]) for dic in lst_detections]
        if plot == True:
            plot_img(detected_img, figsize=figsize)
            print(lst_out)
        return lst_out
    
    except Exception as e:
        print("--- got error ---")
        print(e)



###############################################################################
#                               OCR                                           #
###############################################################################
'''
'''
def from_img_to_txt(img, modelpath, prepr_threshold=True, prepr_blur=True, modelfile="Tesseract-OCR/tesseract.exe", plot=True, figsize=(20,13), lang="eng"):
    try:
        pytesseract.pytesseract.tesseract_cmd = modelpath + modelfile
        
        ## convert to grayscale
        img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ## check to see if we should apply thresholding to preprocess the image
        if prepr_threshold == True:
            img_processed = cv2.threshold(img_processed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
        ## make a check to see if median blurring should be done to remove noise
        elif prepr_blur == True:
            img_processed = cv2.medianBlur(img_processed, 3)
        
        ## plot
        if plot == True:
            plot_2_imgs(img, img_processed, title=None, figsize=figsize)
        
        ## tesseract
        txt = pytesseract.image_to_string(Image.fromarray(img_processed), lang=None)
        return txt.replace("\n", " ")
    
    except Exception as e:
        print("--- got error ---")
        print(e)

