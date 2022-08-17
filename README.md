# Itasca_Mortenson_ML
The Repo consists of the Python Notebooks for Predicting the Velocity field and depth of failure using Neural Network

## Problem Statement
The present problems consists of a domain with a strip loading. Based on the loading there is development of velocity fields. The aim is to predict the velocity field and depth of failure based on the input dat.

## Approach 1: Considering Each node in the domain as a separate training case
## Variables
n_sim : no. of simulation  <br />
n_x : discritization along x- direction  <br />
n_y : discritization along y - direction <br />
velo : velocity at failure  <br />
coh_data : cohesion <br />
fric_data : friction <br />
poly_data : poly <br />


#### Input
- 4ft_cohesion.numpy : consits of variation of cohesion along the z direction. [Dimension - (25,n_sim)]
- 4ft_friction.numpy : consits of variation of friction along the z direction. [Dimension - (25,n_sim)]
- 4ft_poly.numpy : consits of variation of poly along the z direction. [Dimension - (25,n_sim)]
- 4ft_water_table.numpy : consits of variation of cohesion along the z direction. [Dimension - (1,n_sim)]
#### Output 
- 4ft_failure_depths.npy : failure depth [Dimension - (1,n_sim)]
- 4ft_velocity_plots.npy : velocity field [Dimension - (n_x,n_y,n_sim)]

### Method 1: Using ANN for predicting velocity plot
- The input variable will be (cohesion, friction and poly) and output as the velocity field. 
- Each training example will be considered with respect to the node(For Example if there are 825 nodes in the domain and total simulations are n). The number of training examples will be (825*n)
- The dense layer have RelU activation function and since the model is for the prediction. Therefore the last layers activation is taken as linear.
- The obtained results shows the overfitting with low mean square error (the finetunning is performed by considering different number of hidden layers and some additional dropout layers to avoid overfitting. <br />
#### Final Outcome
- The ANN based on the assumption of linear regression that each training sample will be indeprdent of each other. Therefore, it is difficult to capture the spatial realtionship using ANN approach.
- The problem of overfitting is leading to the loss function (Mean Square Error) reach to the very small Value. This can be overcome by using batch normalization (Presently the normaliztions are just performed for the input and dense layer)
- The variation of input features are only in the Y-direction making it difficult for model to learn efficiently.

### Method 2: Using CNN for predicting velocity plot
- The input variable will be (cohesion, friction and poly) in the form of image having each feature in the form of channels. The input will be like (n_sim, n_y, n_x, 3) and output as the velocity field (n_sim,825). 
- The training example will be sample images with 3 channels and size(25,33).
- Normalization and batch normalization is performed to get all the variables in the range (0-1)
- The convolution operation is performed and feature mapping is done using the different kernels followed by max polling and layer flattening.
- Each image is provided a label in the form of column matrix.
- The aim is to predict the whole column (825,1) corresponding to given input in the form of images.
#### Final Outcome
- The proposed architecture works fine. The problem is with the model optimization as during the traing after some steps the loss function reaches to the 
NaN. 
- The loss function and the laerning rate is varied to fix the NaN. 
- The input images are in the form of stripes as the properties are just varied along the z dirction making it difficult for feature mapping.
- The present work can be extended and the each pixel can be assigned as a label and Segmentation can be performed.

### Method 3: Using CNN for predicting vdepth of failure
- The input in this case are velocity plots and output is the depth of failure.
- Normalization is performed to get all the variables in the range (0-1)
- The convolution operation is performed and feature mapping is done using the different kernels followed by max polling and layer flattening.
- Each image is provided a label in the form of depth of failure.
#### Final Outcome
- The model works fine in predicting depth of failure.
- It can be made more accurate by adding more kernels and batch normalization.

## Approach 2: Considering whole domain as a training example
This approach is used on new updated dataset, where in only the variation along z direction (16 nodes) are taken into the consideration.
Each layer property is consider as a separate feature for training. For example: There are 16 layers with varying cohesion, friction and Poly and constant watertable.
The final matrix is tranformed into a row having 49 columns (16- cohesion, 16- friction, 16- poly, 1- watertable)
