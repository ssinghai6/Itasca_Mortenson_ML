#!/usr/bin/env python
# coding: utf-8

# In[4]:


from email.mime import image
from flask import Flask, request, render_template
from keras.models import load_model

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'ann_velo_deploy_norm.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def predict(data_path):
    import numpy as np
    from pickle import load

    data_f = np.load(data_path)
    scaler = load(open('scaler.pkl', 'rb'))
    data_f = scaler.transform(data_f.reshape(-1,1))
    data_f = data_f.reshape(1,-1)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    preds = model.predict(data_f)
    preds = np.reshape(preds,(33,33))
    import matplotlib.pyplot as plt
    a = preds
    b = np.load('static/actual.npy')
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(a, interpolation='nearest', aspect=0.5)
    fig1.colorbar(im1)
    plt.title("Velocity field - ML PREDICTION")
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(b, interpolation='nearest',aspect = 0.5)
    fig2.colorbar(im2)
    plt.title("Velocity field - SIMULATION")
    fig2.savefig('static/actual.jpeg', dpi =200)
    fig1.savefig('static/ml.jpeg', dpi =200)
    return (preds)

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        data = request.files['file']
        data_path = "static/" + data.filename	
        #data.save(data_path)
        p = predict(data_path).all()
    return render_template("index.html", prediction = p)  
if __name__ =='__main__':
    app.run(debug = True)

