#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load model and scaler and make predictions on new data
def ann_velo_predictor():
    data_path = 'input.npy'
    #!pip install tensorflow_addons
    import tensorflow_addons as tfa
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from pickle import load
    from keras.models import load_model
    MODEL_PATH = 'ann_velo_v3.h5'
    # Load your trained model
    model = load_model(MODEL_PATH)
    #def predict(data_path):
    import numpy as np
    data_f = np.load(data_path)
    scaler = load(open('scale_v3.pkl', 'rb'))
    data_f = scaler.transform(data_f.reshape(1,-1))
    #data_f = data_f.reshape(1,-1)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    preds = model.predict(data_f)
    preds = np.reshape(preds,(33,33))
    import matplotlib.pyplot as plt
    a = preds
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(a, interpolation='nearest', aspect=0.5)
    fig1.colorbar(im1)
    return(plt.title("Velocity field - ML PREDICTION"))

if __name__ = "__main__":
    ann_velo_predictor

