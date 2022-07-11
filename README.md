# Itasca_Mortenson_ML
The Repo consists of the Python Notebooks for Predicting the Velocity field and depth of failure using Neural Network

## Problem Statement
The present problems consists of a domain with a strip loading. Based on the loading there is development of velocity fields. The aim is to predict the velocity field and depth of failure based on the input dat.
## Variables
n_sim : no. of simulation  <br />
n_x : discritization along x- direction  <br />
n_y : discritization along y - direction <br />
velo : velocity at failure  <br />
coh_data : cohesion <br />
fric_data : friction <br />
poly_data : poly <br />

### Approach 1
Use of ANN to predict the velocity field at individual nodes. The domain is discritized into 25 nodes along the z direction and 33 nodes along the x direction.
#### Input data
- 4ft_cohesion.numpy : consits of variation of cohesion along the z direction. [Dimension - (25,n_sim)]
- 4ft_friction.numpy : consits of variation of friction along the z direction. [Dimension - (25,n_sim)]
- 4ft_poly.numpy : consits of variation of poly along the z direction. [Dimension - (25,n_sim)]
- 4ft_water_table.numpy : consits of variation of cohesion along the z direction. [Dimension - (1,n_sim)]
#### Output 
- 4ft_failure_depths.npy : failure depth [Dimension - (1,n_sim)]
- 4ft_velocity_plots.npy : velocity field [Dimension - (n_x,n_y,n_sim)]


