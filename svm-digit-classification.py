#!/usr/bin/env python

"""
Train a SVM to categorize 28x28 pixel images into digits (MNIST dataset).
"""

import numpy as np
from PIL import ImageTk, Image, ImageDraw
import PIL
from Tkinter import *
    
    

def get_image(clf):
   
    img = PIL.Image.open('digit.png').convert("L")
    im2 = img.resize((28, 28), PIL.Image.NEAREST)
    a = np.array(im2)
    imageArray = a.ravel()
    imageArray = imageArray/255.0*2 - 1
    imageDigit=clf.predict(np.matrix(imageArray))
    print("digito clasificado: ", imageDigit[0])
def analyze(clf,clf2,clf3,data):
    #problem 1.3 

    # Get confusion matrix
    from sklearn import metrics
    
    predicted = clf.predict(data['test']['X']) #predicting for the testing data
    predictedLin = clf2.predict(data['test']['X']) #predicting for the testing data
    predictedPoly = clf3.predict(data['test']['X']) #predicting for the testing data
    print("Confusion matrix for RBF kernel:\n%s" %
          metrics.confusion_matrix(data['test']['y'],
                                   predicted))
    print("metrics: \n")
    print(metrics.classification_report(data['test']['y'],
                                                     predicted))
    print("\n")
    print("\n")
    print("Confusion matrix for Linear kernel:\n%s" %
         metrics.confusion_matrix(data['test']['y'],
                                 predictedLin))
    print("metrics: \n")
    print(metrics.classification_report(data['test']['y'],
                                                     predictedLin))   

    print("\n")
    print("\n")
    print("Confusion matrix for Polynomial kernel:\n%s" %
         metrics.confusion_matrix(data['test']['y'],
                                 predictedPoly))
    print("metrics: \n")                             
    print(metrics.classification_report(data['test']['y'],
                                                     predictedPoly))   
    # Print example
    #try_id = 4
    #out = clf.predict(data['test']['X'][try_id])  # clf.predict_proba
    #print("out: %s" % out)
    #size = int(len(data['test']['X'][try_id])**(0.5))
    #view_image(data['test']['X'][try_id].reshape((size, size)),
    #           data['test']['y'][try_id])
    #print(predicted[try_id])
    
    
def view_image(image, label=""):
    """
    View a single image.

    Parameters
    ----------
    image : numpy array
        Make sure this is of the shape you want.
    label : str
    """
    from matplotlib.pyplot import show, imshow, cm
    from matplotlib.image import *
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    imsave("ima.png",image)
    show()


def get_data():
    """
    Get data ready to learn with.

    Returns
    -------
    dict
    """
    simple = False
    if simple:  # Load the simple, but similar digits dataset
        from sklearn.datasets import load_digits
        from sklearn.utils import shuffle
        digits = load_digits()
        x = [np.array(el).flatten() for el in digits.images]
        y = digits.target

        # Scale data to [-1, 1] - This is of mayor importance!!!
        # In this case, I know the range and thus I can (and should) scale
        # manually. However, this might not always be the case.
        # Then try sklearn.preprocessing.MinMaxScaler or
        # sklearn.preprocessing.StandardScaler
        x = x/255.0*2 - 1

        x, y = shuffle(x, y, random_state=0)

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)
        data = {'train': {'X': x_train,
                          'y': y_train},
                'test': {'X': x_test,
                         'y': y_test}}
    else:  # Load the original dataset
        from sklearn.datasets import fetch_mldata
        from sklearn.utils import shuffle
        mnist = fetch_mldata('MNIST original')

        x = mnist.data
        y = mnist.target
        # Scale data to [-1, 1] - This is of mayor importance!!!
        x = x/255.0*2 - 1

        x, y = shuffle(x, y, random_state=0)
        
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)
       
        data = {'train': {'X': x_train,
                          'y': y_train},
                'test': {'X': x_test,
                         'y': y_test}}
    return data

def save():
    filename = "digit.png"
    image1.save(filename)

def paint(event):
    # python_green = "#476042"
    event.widget
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="white",width=15)
    draw.line([x1, y1, x2, y2],fill="white",width=15)


if __name__ == '__main__':
    width = 200
    height = 200
    center = height//2
    white = (255, 255, 255)

    data = get_data()#retriving image data from MNIST

    # Choose classifier configuration
    from sklearn.svm import SVC
    clfRBF = SVC(probability=False,  # cache_size=200,
              kernel="rbf", C=2.8, gamma=.0073)
    
    clfLin = SVC(probability=False,  # cache_size=200,
              kernel="linear", C=2.8, gamma=.0073)

    clfPoly = SVC(probability=False,  # cache_size=200,
              kernel="poly", C=2.8, degree=3)          

    print("Entrenando, espere un momento...")

    #examples = len(data['train']['X']) # number of training images to use
    examples = 1000;
    clfRBF.fit(data['train']['X'][:examples], data['train']['y'][:examples]) #training
    clfLin.fit(data['train']['X'][:examples], data['train']['y'][:examples]) #training
    clfPoly.fit(data['train']['X'][:examples], data['train']['y'][:examples]) #training

    ###problem 1.3###########
    analyze(clfRBF,clfLin,clfPoly, data)
    #########################
    
    ava=input("\n\n Escriba 1 para probar el clasificador \n\n ")
    while(True):
        print("\n\n dibuje un digito presione listo! y luego cierre la ventana\n\n ")
        root = Tk()

        # Tkinter create a canvas to draw on
        cv = Canvas(root, width=width, height=height, bg='white')
        cv.pack()

        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        image1 = PIL.Image.new("RGB", (width, height), (0,0,0))
        draw = PIL.ImageDraw.Draw(image1)
        # do the Tkinter canvas drawings (visible)
        
        cv.pack(expand=YES, fill=BOTH)
        cv.bind("<B1-Motion>", paint)

        # do the PIL image/draw (in memory) drawings
        # draw.line([0, center, width, center], green)

        # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
        
        button=Button(text="Listo!",command=save)
        button.pack()
        root.mainloop()
        get_image(clfRBF)
        seguir=input("\n\n Desea continuar? 1. SI 2. NO \n\n ")
        if(seguir==2):
            break