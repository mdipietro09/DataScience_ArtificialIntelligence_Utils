
## for data
import cv2
import os
import numpy as np

## for plotting
import matplotlib.pyplot as plt

## for object detection
from imageai import Detection
from imageai.Detection.Custom import DetectionModelTrainer, CustomObjectDetection

## for deep learning
from tensorflow.keras import models, layers, applications

## for ocr
import pytesseract



###############################################################################
#                   IMG ANALYSIS                                              #
###############################################################################
'''
Plot a single image with pyplot.
:parameter
    :param img: image array
    :param mask: image array
    :param rect: list of tuples - [(x1,y1), (x2,y2)]
    :param title: string
'''
def utils_plot_img(img, mask=None, rect=None, title=None, figsize=(5,3)):
    plot_img = img.copy()
    if mask is not None:
        mask = cv2.resize(mask, (img.shape[0],img.shape[1]), interpolation=cv2.INTER_LINEAR)
        plot_img = cv2.bitwise_and(plot_img, mask)
    if rect is not None:
        plot_img = cv2.rectangle(plot_img, rect[0], rect[1], (255,0,0), 4)
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=20)
    if len(img.shape) > 2:
        plt.imshow(plot_img)
    else:
        plt.imshow(plot_img, cmap=plt.cm.binary)



'''
Plot n images in (1 row) x (n columns).
'''
def plot_imgs(lst_imgs, lst_titles=[], figsize=(20,13)):
    fig, ax = plt.subplots(nrows=1, ncols=len(lst_imgs), sharex=False, sharey=False, figsize=figsize)
    if len(lst_titles) == 1:
        fig.suptitle(lst_titles[0], fontsize=20)
    for i,img in enumerate(lst_imgs):
        ax[i].imshow(img)
        if len(lst_titles) > 1:
            ax[i].set(title=lst_titles[i])
    plt.show()


    
'''
Load a single image with opencv.
'''
def utils_load_img(file, ext=['.png','.jpg','.jpeg','.JPG'], plot=True, figsize=(5,3)):
    if file.endswith(tuple(ext)):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if plot == True:
            utils_plot_img(img, figsize=figsize)
        return img
    else:
        print("file extension unknown")
    

    
'''
Load a folder of imgs.
'''
def load_imgs(dirpath, ext=['.png','.jpg','.jpeg','.JPG'], plot=False, figsize=(20,13)):
    lst_imgs =[]
    for file in os.listdir(dirpath):
        try:
            if file.endswith(tuple(ext)):
                img = utils_load_img(file=dirpath+file, ext=ext, plot=False)
                lst_imgs.append(img)
        except Exception as e:
            print("failed on:", file, "| error:", e)
            pass
    if plot is True:
        plot_imgs(lst_imgs[0:5], lst_titles=[], figsize=figsize)
    return lst_imgs


    
'''
Plot univariate and bivariate colors histogram.
'''
def utils_color_distributions(lst_imgs, lst_y=None, figsize=(5,3)):
    ## univariate
    if lst_y is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(xlim=[0,256], xlabel='bin', ylabel="pixel count", title=str(len(lst_imgs))+" imgs")
        ax.grid(True)
        for i,col in enumerate(("r","g","b")):
            hist = cv2.calcHist(images=lst_imgs, channels=[i], mask=None, histSize=[256], ranges=[0,256])
            ax.plot(hist, color=col)
    ## bivariate
    else:
        ### create samples
        dic_samples = {y:[] for y in np.unique(lst_y)}
        for i,x in enumerate(lst_imgs):
            dic_samples[lst_y[i]].append(x)
        ### plot
        fig, ax = plt.subplots(nrows=len(dic_samples.keys()), ncols=1, sharex=True, figsize=figsize)
        for n,y in enumerate(dic_samples.keys()):
            ax[n].set(xlim=[0,256], xlabel='bin', ylabel="pixel count", title=str(y)+": "+str(len(dic_samples[y]))+" imgs")
            ax[n].grid(True)
            for i,col in enumerate(("r","g","b")):
                hist = cv2.calcHist(images=dic_samples[y], channels=[i], mask=None, histSize=[256], ranges=[0,256])
                ax[n].plot(hist, color=col)
    plt.show()
    
    

'''
Preprocess a single image.
'''
def utils_preprocess_img(img, resize=256, denoise=False, remove_color=False, morphology=False, segmentation=False, plot=False, figsize=(20,13)):
    ## original
    img_processed = img
    lst_imgs = [img_processed]
    lst_titles = ["original:  "+str(img_processed.shape)]
    
    ## scale
    #img_processed = img_processed/255
    
    ## resize
    if resize is not False:
        img_processed = cv2.resize(img_processed, (resize,resize), interpolation=cv2.INTER_LINEAR)
        lst_imgs.append(img_processed)
        lst_titles.append("resized:  "+str(img_processed.shape))
    
    ## denoise (blur)
    if denoise is True:
        img_processed = cv2.GaussianBlur(img_processed, (5,5), 0)
        lst_imgs.append(img_processed)
        lst_titles.append("blurred:  "+str(img_processed.shape))
    
    ## remove color
    if remove_color is True:
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2GRAY)
        lst_imgs.append(img_processed)
        lst_titles.append("removed color:  "+str(img_processed.shape))
    
    ## morphology
    if morphology is True:
        if len(img_processed.shape) > 2:
            ret, mask = cv2.threshold(img_processed, 255/2, 255, cv2.THRESH_BINARY)
        else:
            mask = cv2.adaptiveThreshold(img_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=2)   
        lst_imgs.append(img_processed)
        lst_titles.append("morphology:  "+str(img_processed.shape))
    
        ## segmentation (after morphology)
        if segmentation is True:
            background = cv2.dilate(img_processed, np.ones((3,3),np.uint8), iterations=3)
            if len(img_processed.shape) > 2:
                print("--- need to remove color to segment ---")
            else:
                ret, foreground = cv2.threshold(cv2.distanceTransform(img_processed, cv2.DIST_L2, 5), 
                                                0.7 * cv2.distanceTransform(img_processed, cv2.DIST_L2, 5).max(), 
                                                255, 0)
                foreground = np.uint8(foreground)
                ret, markers = cv2.connectedComponents(foreground)
                markers = markers + 1
                unknown = cv2.subtract(background, foreground)
                markers[unknown == 255] = 0
                img_processed = cv2.watershed(cv2.resize(img, img_processed.shape, interpolation=cv2.INTER_LINEAR), markers)
                lst_imgs.append(img_processed)
                lst_titles.append("segmented:  "+str(img_processed.shape))
    if (segmentation is True) and (morphology is False):
        print("--- need to do morphology to segment ---")
    
    ## plot
    if plot is True:
        plot_imgs(lst_imgs, lst_titles, figsize)
    return img_processed



'''
Save array images into directory.
'''
def save_imgs(lst_imgs, dirpath, lst_names=None, i=0):
    for img in lst_imgs:
        try:
            name = str(lst_names[i]) if lst_names is not None else str(i)
            if len(img.shape) > 2:
                cv2.imwrite(dirpath+name+".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                cv2.imwrite(dirpath+name+".jpg", img)
            i += 1
        except Exception as e:
            print("failed on:", i, "| error:", e)
            i += 1
            pass



###############################################################################
#                  OBJECT DETECTION                                           #
###############################################################################
'''
Loads yolo model with imageai.
'''
def load_yolo(modelfile="yolo.h5", confjson=None):
    yolo = Detection.ObjectDetection() if confjson is None else CustomObjectDetection()
    yolo.setModelTypeAsYOLOv3()
    yolo.setModelPath(modelfile)
    if confjson is not None:
        yolo.setJsonPath(confjson)
    yolo.loadModel()
    return yolo



'''
Predict with yolo.
:return
    img array, dic with metadata, cropped img
'''
def obj_detect_yolo(img, yolo, min_prob=70, plot=False, figsize=(20,13)):
    predicted_img, info = yolo.detectObjectsFromImage(input_image=img, input_type="array", output_type="array",
                                                      minimum_percentage_probability=min_prob,
                                                      display_percentage_probability=True, display_object_name=True)
    ## crop image
    if len(info) > 0: 
        points = info[0]["box_points"]
        cropped_img = predicted_img[points[0]:points[3], points[1]:points[2]]
    else:
        cropped_img = predicted_img
    
    ## plot full img
    if plot == True:
        utils_plot_img(predicted_img, figsize=figsize)
    return predicted_img, info, cropped_img



'''
Retrain yolo model with custom labeled images.
:parameter
    :param lst_y: list of strings of unique labels
    :param train_path: str - directory with images, by default it searches for this scrtuture:
                        /train/images/img_1.jpg, img_2.jpg ...
                        /train/annotations/img_1.xml, img_2.xml ...
                        /validation/images/img_151.jpg, img_152.jpg ...
                        /validation/annotations/img_151.xml, img_152.xml ...
    :param modelfile_transfer: str - "models/yolo.h5" or empty string "" to train from scratch
'''
def train_yolo(lst_y, train_path="fs/training_yolo/", transfer_modelfile=""):
    ## setup
    yolo = DetectionModelTrainer()
    yolo.setModelTypeAsYOLOv3()
    yolo.setDataDirectory(data_directory=train_path)
    yolo.setTrainConfig(object_names_array=lst_y, batch_size=4, num_experiments=1, 
                        train_from_pretrained_model=transfer_modelfile)

    ## train
    print("--- training ---")
    yolo.trainModel()
    
    ## evaluate
    print("--- metrics ---")
    metrics = yolo.evaluateModel(model_path=train_path+"models", json_path=train_path+"json/detection_config.json", 
                                 iou_threshold=0.5, object_threshold=0.5, nms_threshold=0.5)
    print(metrics)
    
    ## laod model
    print("--- loading model ---")
    modelfile = os.listdir(train_path+"models/")[0]
    return load_yolo(modelfile=train_path+"models/"+modelfile, confjson=train_path+"/json/detection_config.json")    



###############################################################################
#             MODEL DESIGN & TESTING - MULTILABEL CLASSIFICATION              #
###############################################################################
'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metric = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)][0]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='blue')
    ax11.plot(training.history[metric], color='green')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='blue')
    ax11.set_ylabel(metric.capitalize(), color='green')
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='blue')
    ax22.plot(training.history['val_'+metric], color='green')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='blue')
    ax22.set_ylabel(metric.capitalize(), color='green')
    plt.show()
    


'''
Fit convolutional neural network.
:parameter
    :param X_train: ImageGen object
    :param X_test: ImageGen object
    :param savepath: str - path where to save the model, if None doesn't save
:return
    model
'''
def fit_cnn(X_train, X_test, model=None, batch_size=32, epochs=100, verbose=0, savepath=None):
    ## cnn
    if model is None:
        img_shape = X_train.image_shape
        n_classes = len(X_train.class_indices)
        ### layer 1 conv 5x5 (32 neurons) + pool 2x2
        model = models.Sequential()
        model.add( layers.Conv2D(input_shape=img_shape, kernel_size=(5,5), filters=32, strides=(1,1), padding="valid", activation='relu') )
        model.add( layers.MaxPooling2D(pool_size=(2,2), padding="valid") )
        ### layer 2 conv 3x3 (64 neurons) + pool 2x2
        model.add( layers.Conv2D(kernel_size=(3,3), filters=64, strides=(1,1), padding="valid", activation='relu') )
        model.add( layers.MaxPooling2D(pool_size=(2,2), padding="valid") )
        ### layer 3 fully connected (128 neuroni)
        model.add( layers.Flatten() )
        model.add( layers.Dense(units=128, activation="relu") )
        model.add( layers.Dropout(rate=0.2) )
        ### layer output (n_classes neurons)
        if n_classes == 2:
            print("y binary --> output layer: 1 neuron with 'sigmoid' activation and 'binary_crossentropy' loss")
            model.add( layers.Dense(units=1, activation="sigmoid") )
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        else:
            print("y multiclasses --> output layer: ", n_classes, "neurons with 'softmax' activation and 'categorical_crossentropy' loss")
            model.add( layers.Dense(units=n_classes, activation="softmax") )
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    ## train
    print(model.summary())
    training = model.fit_generator(X_train, steps_per_epoch=batch_size, epochs=epochs, validation_data=X_test, verbose=verbose)
    if savepath is not None:
        training.model.save(savepath+"cnn.h5")
    
    ## evaluate
    utils_plot_keras_training(training)
    return training.model



'''
Fit convolutional neural network from pre-trained layers.
:parameter
    :param X_train: ImageGen object
    :param X_test: ImageGen object
    :param base: keras layers object
    :param new_layers: keras layers object
    :param savepath: str - path where to save the model, if None doesn't save
:return
    model
'''
def fit_cnn_transfer_learning(X_train, X_test, base=None, new_layer=None, batch_size=32, epochs=100, verbose=0, savepath=None):
    img_shape = X_train.image_shape
    n_classes = len(X_train.class_indices)
    if n_classes == 2:
        output_neurons, activation, loss = 1, "sigmoid", "binary_crossentropy"
    else:
        output_neurons, activation, loss = n_classes, "softmax", "categorical_crossentropy"
    
    ## starting layers
    if base is None:
        base = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=img_shape) 
        for layer in base.layers:
            layer.trainable = False
    
    ## check input shapes
    print("--- check ---")
    print("Base model requires input of shape:", base.input.shape[1:])
    print("X shape:", X_train.image_shape)
    if X_train.image_shape != base.input.shape[1:]:
        print(" !!! Stopped for shape error")
        return False    
    
    ## add layers
    if new_layer is None:
        new_layer = base.output
        new_layer = layers.Flatten()(new_layer)
        new_layer = layers.Dense(units=1024, activation='relu')(new_layer)
        new_layer = layers.Dropout(rate=0.5)(new_layer)
        new_layer = layers.Dense(units=output_neurons, activation=activation)(new_layer)
    
    ## create the model
    model = models.Model(inputs=base.input, outputs=new_layer)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
   
    ## train
    print(model.summary())
    training = model.fit_generator(X_train, steps_per_epoch=batch_size, epochs=epochs, validation_data=X_test, verbose=verbose)
    if savepath is not None:
        training.model.save(savepath+"cnn_transfer.h5")
    
    ## evaluate
    utils_plot_keras_training(training)
    return training.model 



'''
Evaluates a model performance.
'''
def evaluate_cnn(y_test, predicted, predicted_prob, figsize=(20,10)):
    return 0



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
            plot_imgs([img, img_processed], figsize=figsize)
        
        ## tesseract
        txt = pytesseract.image_to_string(Image.fromarray(img_processed), lang=None)
        return txt.replace("\n", " ")
    
    except Exception as e:
        print("--- got error ---")
        print(e)



###############################################################################
#                  IMAGES MATCHING                                            #
###############################################################################
'''
Computes the similarity of two images.
:parameter
    :param a: img
    :param b: img
    :param algo: string - "orb" "sift" "flann"
    :param plot: logic
    :param plot_points: logic
:return
    full_matches, matches
'''
def utils_imgs_similarity(a, b, algo="sift", plot=False, plot_points=False, figsize=(20,13)):
    flags = 0 if plot_points is True else 2
    
    if algo == "orb":
        detector = cv2.ORB_create()
        key_points_A, des_A = detector.detectAndCompute(a, None)
        key_points_B, des_B = detector.detectAndCompute(b, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        full_matches = matcher.match(des_A, des_B)
        matches = sorted(full_matches, key=lambda x: x.distance)[:10]
        if plot is True:
            fig = cv2.drawMatches(a, key_points_A, b, key_points_B, matches, None, flags=flags)
            utils_plot_img(fig, title="Matching Algorithm: "+algo.upper(), figsize=figsize)
        
    elif algo == "sift":
        detector = cv2.xfeatures2d.SIFT_create()
        key_points_A, des_A = detector.detectAndCompute(a, None)
        key_points_B, des_B = detector.detectAndCompute(b, None)
        matcher = cv2.BFMatcher()
        full_matches = matcher.knnMatch(des_A, des_B, k=2)
        matches = [[m1] for m1,m2 in full_matches if m1.distance < 0.75*m2.distance] #ratio test
         
    elif algo == "flann":
        detector = cv2.xfeatures2d.SIFT_create()
        key_points_A, des_A = detector.detectAndCompute(a, None)
        key_points_B, des_B = detector.detectAndCompute(b, None)
        matcher = cv2.FlannBasedMatcher({"algorithm":0, "trees":5}, {"checks":50})
        full_matches = matcher.knnMatch(des_A, des_B, k=2)
        matches = [[m1] for m1,m2 in full_matches if m1.distance < 0.75*m2.distance] #ratio test
        
    if (algo != "orb") and (plot is True): 
        #fig = cv2.drawMatchesKnn(a, key_points_A, b, key_points_B, matches, None, flags=flags)
        matchesMask = [[0,0] for i in range(len(full_matches))]
        for i,(m1,m2) in enumerate(full_matches):
            matchesMask[i] = [1,0] if m1.distance < 0.75*m2.distance else matchesMask[i]
        fig = cv2.drawMatchesKnn(a, key_points_A, b, key_points_B, full_matches, None,
                                 **{"matchColor":(0,255,0), "singlePointColor":(255,0,0), "matchesMask":matchesMask, "flags":flags})
        utils_plot_img(fig, title="Matching Algorithm: "+algo.upper(), figsize=figsize)
            
    return full_matches, matches
    


'''
'''
def match_imgs():
    return 0



###############################################################################
#                       GENERATIVE MODELS                                     #
###############################################################################
'''
'''
def 