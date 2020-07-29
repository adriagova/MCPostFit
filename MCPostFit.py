#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from scipy.sparse.linalg import spsolve 
import time
import math
from scipy.stats import uniform
from scipy.stats import multivariate_normal as mvnorm
from matplotlib import rcParams
from scipy import interpolate
import matplotlib.tri as tri
from scipy.interpolate import RectBivariateSpline
from numpy.random import choice 

time_start = time.clock()

# Read the file where the MonteCarlo Markov chain is saved
MCMC_total = pd.read_csv(str(sys.argv[1]),sep ="\t")

# If the option "random" is selected as an input (4th argument) then the code shuffles the original MCMC
if str(sys.argv[4]) == 'random':
	MCMC_total = MCMC_total.sample(frac=1)
	if str(sys.argv[3])=='full':
		print "\n\n Warning: the option -random- has no effect when the full MCMC is considered." 

# Total number of parameters
N_param = int(sys.argv[2])

# Vector containing the columns associated to the various parameters in the input file (the one that contains the chain)
columns_param = []
for i in range(1,N_param+1):
	columns_param.append(i)

# Number of lines from the MCMC used to perform the fit of the posterior.
if str(sys.argv[3])=='full':
	N_points = MCMC_total.shape[0]
else:
	N_points = int(sys.argv[3])
	if N_points>MCMC_total.shape[0]:
		print "Error: you have selected more lines than the total number contained in your Markov chain!"
		print "Notice that N_points should be less than:", MCMC_total.shape[0]
		sys.exit()
				
# This matrix contains the N_points of the full MCM that will be used to perform the fit
MCMC = MCMC_total.iloc[0:N_points,:]



# Select the first N_K parameters that will have cubic and quartic corrections in the fitting distribution
N_K = int(sys.argv[5])

print "\n\n###########################"
print "MONTE CARLO POSTERIOR FIT"
print "###########################\n"	

# Obtain the total number of points in the selected and full MCMC, the associated values of -2ln(L/L_max) and the degeneracy
rep_chi2_total = MCMC_total.iloc[:, 0:1]
rep_total = np.zeros((MCMC_total.shape[0]))
True_N_points_total = 0 
for i in xrange(rep_chi2_total.shape[0]):
    b =  rep_chi2_total.iat[i,0]  
    rep_total[i] = float(b.split(' ')[0])
    True_N_points_total = True_N_points_total + rep_total[i]
    
rep_chi2 = MCMC.iloc[:, 0:1]
rep = np.zeros((len(rep_chi2)))
chi2 = np.zeros((len(rep_chi2)))
True_N_points = 0    
for i in xrange(rep_chi2.shape[0]):
    b =  rep_chi2.iat[i,0]  
    rep[i] = float(b.split(' ')[0])
    chi2[i] = 2*float(b.split(' ')[2])
    True_N_points = True_N_points + rep[i]

print "Total number of points selected from the Markov chain(s):", True_N_points,"/",True_N_points_total

# Minimum value of -2ln(L/L_max) from the selected MCMC 
chi2_min = min(chi2)
print "chi2_min in the selected chain:", chi2_min



# Best-fit vector in the Markov chain, saved in BF

i, = np.where(chi2 == chi2_min)
index_min = i[0]	
BF = np.zeros((N_param))
for i in range(N_param):
	BF[i] = MCMC.iat[index_min,columns_param[i]]
	
for i in range(rep_chi2.shape[0]):
	chi2[i] = chi2[i]-chi2_min

# Create the matrix containing the differences with respect to the best-fit values for each entry of the Markov chain

dif_vec = np.zeros((len(rep_chi2),N_param))
dif_vec_mod = np.zeros((len(rep_chi2),N_param))
for j in xrange(N_param):
	for i in xrange(dif_vec.shape[0]):
		dif_vec[i][j] = MCMC.iat[i,columns_param[j]]-BF[j]	
		dif_vec_mod[i][j] = (MCMC.iat[i,columns_param[j]]-BF[j])*(rep[i])**0.5		

# We build a matrix containing the MCMC, instead of a table. It is used in some parts of the code

MCMC_matrix = np.zeros((MCMC.shape[0],N_param))
for i in range(MCMC.shape[0]):
	for j in range(N_param):
		MCMC_matrix[i][j] = MCMC.iat[i,j+1]
		
# Build dictionaries between the indexes of the fitting matrices and their vector forms

# For the operations with M
D_dim = N_param * (N_param+1)/2  # Dimension of the matrix M when it is written in the vector form
dictionary = np.zeros((D_dim,2))
cont = 0
for i in xrange(N_param):
	for j in range(i,N_param):
		dictionary[cont][0] = i
		dictionary[cont][1] = j
		cont = cont + 1


#For the operations with S and K
K_dim = math.factorial(N_K+3)/(24*math.factorial(N_K-1))
S_dim_red = math.factorial(N_K+2)/(6*math.factorial(N_K-1))
D_dim_red = N_K*(N_K+1)/2

dictionary_red = np.zeros((D_dim_red,2))
cont = 0
for i in xrange(N_K):
	for j in range(i,N_K):
		dictionary_red[cont][0] = i
		dictionary_red[cont][1] = j
		cont = cont + 1
		
dictionary_S_red = np.zeros((S_dim_red,4))

cont = 0
start_point = 0
for i in xrange(N_K):
	inter_sum = start_point
	for j in range(start_point,D_dim_red):
		dictionary_S_red[cont][0] = dictionary_red[inter_sum][0]
		dictionary_S_red[cont][1] = dictionary_red[inter_sum][1]
		dictionary_S_red[cont][2] = i
		if dictionary_S_red[cont][0] == dictionary_S_red[cont][1] and dictionary_S_red[cont][0] == dictionary_S_red[cont][2]:
			dictionary_S_red[cont][3] = 1
		if dictionary_S_red[cont][0] != dictionary_S_red[cont][1] and dictionary_S_red[cont][0] != dictionary_S_red[cont][2]:
			dictionary_S_red[cont][3] = 6
		if dictionary_S_red[cont][0] == dictionary_S_red[cont][1] and dictionary_S_red[cont][0] != dictionary_S_red[cont][2]:
			dictionary_S_red[cont][3] = 3
		if dictionary_S_red[cont][0] != dictionary_S_red[cont][1] and dictionary_S_red[cont][0] == dictionary_S_red[cont][2]:
			dictionary_S_red[cont][3] = 3
		if dictionary_S_red[cont][1] == dictionary_S_red[cont][2] and dictionary_S_red[cont][0] != dictionary_S_red[cont][1]:
			dictionary_S_red[cont][3] = 3			
		inter_sum = inter_sum + 1
		cont = cont + 1
	start_point = start_point + N_K-i

dictionary_K = np.zeros((K_dim,5))
cont = 0
start_point = 0
for i in xrange(N_K):
	inter_sum = start_point
	for j in range(start_point,S_dim_red):
		dictionary_K[cont][0] = dictionary_S_red[inter_sum][0]
		dictionary_K[cont][1] = dictionary_S_red[inter_sum][1]
		dictionary_K[cont][2] = dictionary_S_red[inter_sum][2]	
		dictionary_K[cont][3] = i
		if dictionary_K[cont][0] == dictionary_K[cont][1] and dictionary_K[cont][2] == dictionary_K[cont][3]:
			if dictionary_K[cont][0] == dictionary_K[cont][2]:
				dictionary_K[cont][4] = 1
			if dictionary_K[cont][0] != dictionary_K[cont][2]:
				dictionary_K[cont][4] = 6
		if dictionary_K[cont][0] == dictionary_K[cont][2] and dictionary_K[cont][1] == dictionary_K[cont][3]:
			if dictionary_K[cont][0] != dictionary_K[cont][1]:
				dictionary_K[cont][4] = 6
		if dictionary_K[cont][0] == dictionary_K[cont][3] and dictionary_K[cont][1] == dictionary_K[cont][2]:
			if dictionary_K[cont][0] != dictionary_K[cont][1]:
				dictionary_K[cont][4] = 6
		if dictionary_K[cont][0] != dictionary_K[cont][1] and dictionary_K[cont][0] != dictionary_K[cont][2] and dictionary_K[cont][0] != dictionary_K[cont][3]:
			if dictionary_K[cont][1] == dictionary_K[cont][2] and dictionary_K[cont][2] == dictionary_K[cont][3]:
				dictionary_K[cont][4] = 4
		if dictionary_K[cont][1] != dictionary_K[cont][0] and dictionary_K[cont][1] != dictionary_K[cont][2] and dictionary_K[cont][1] != dictionary_K[cont][3]:
			if dictionary_K[cont][0] == dictionary_K[cont][2] and dictionary_K[cont][0] == dictionary_K[cont][3]:
				dictionary_K[cont][4] = 4
		if dictionary_K[cont][2] != dictionary_K[cont][0] and dictionary_K[cont][2] != dictionary_K[cont][1] and dictionary_K[cont][2] != dictionary_K[cont][3]:
			if dictionary_K[cont][0] == dictionary_K[cont][1] and dictionary_K[cont][0] == dictionary_K[cont][3]:
				dictionary_K[cont][4] = 4
		if dictionary_K[cont][3] != dictionary_K[cont][0] and dictionary_K[cont][3] != dictionary_K[cont][1] and dictionary_K[cont][3] != dictionary_K[cont][2]:
			if dictionary_K[cont][0] == dictionary_K[cont][1] and dictionary_K[cont][0] == dictionary_K[cont][2]:
				dictionary_K[cont][4] = 4
		if dictionary_K[cont][0] != dictionary_K[cont][1] and dictionary_K[cont][0] != dictionary_K[cont][2] and dictionary_K[cont][0] != dictionary_K[cont][3]:
				if dictionary_K[cont][1] != dictionary_K[cont][2] and dictionary_K[cont][1] != dictionary_K[cont][3]:
					if dictionary_K[cont][2] != dictionary_K[cont][3]:
						dictionary_K[cont][4] = 24					
		inter_sum = inter_sum + 1
		cont = cont + 1
	suma = 0	
	if i>=1 :	
		for k in range(0,i):
			suma = suma + k	
	start_point = start_point + D_dim_red - i*N_K + suma

for i in xrange(K_dim):
	if dictionary_K[i][4] == 0:
		dictionary_K[i][4] = 12	

# Build the needed matrices containing the products of the vector param-BF 

D_matrix = np.zeros((dif_vec.shape[0],D_dim))
for i in xrange(dif_vec.shape[0]):
	for j in xrange(D_dim):
		a = int(dictionary[j][0])
		b = int(dictionary[j][1])
		D_matrix[i][j] = (dif_vec[i][a])*(dif_vec[i][b])*(rep[i])**0.5

D3_matrix = np.zeros((dif_vec.shape[0],S_dim_red))
for i in xrange(dif_vec.shape[0]):
	for j in xrange(S_dim_red):
		a = int(dictionary_S_red[j][0])
		b = int(dictionary_S_red[j][1])
		c = int(dictionary_S_red[j][2])
		D3_matrix[i][j] = (dif_vec[i][a])*(dif_vec[i][b])*(dif_vec[i][c])*(rep[i])**0.5

D4_matrix = np.zeros((dif_vec.shape[0],K_dim))
for i in xrange(dif_vec.shape[0]):
	for j in xrange(K_dim):
		a = int(dictionary_K[j][0])
		b = int(dictionary_K[j][1])
		c = int(dictionary_K[j][2])
		r = int(dictionary_K[j][3])
		D4_matrix[i][j] = (dif_vec[i][a])*(dif_vec[i][b])*(dif_vec[i][c])*(dif_vec[i][r])*(rep[i])**0.5

print "Computation of D matrices: DONE"

# Computation of the super_vector

super_dim = 1 + N_param + D_dim
superVec = np.zeros((super_dim))

super_dim_NG = 1 + N_param + D_dim + S_dim_red + K_dim	
superVec_NG = np.zeros((super_dim_NG))	

for i in range(super_dim_NG):
	if i == 0:
		for j in xrange(dif_vec.shape[0]):
			superVec[i] = superVec[i] + chi2[j]*rep[j]
			superVec_NG[i] = superVec_NG[i] + chi2[j]*rep[j]
	if i>0 and i<=N_param:
		for j in xrange(dif_vec.shape[0]):
			superVec[i] = superVec[i] + chi2[j]*dif_vec[j][i-1]*rep[j]
			superVec_NG[i] = superVec_NG[i] + chi2[j]*dif_vec[j][i-1]*rep[j]
	if i>N_param and i<=(N_param+D_dim):
		for j in xrange(dif_vec.shape[0]):
			superVec[i] = superVec[i] + (D_matrix[j][i-N_param-1])*chi2[j]*(rep[j])**0.5
			superVec_NG[i] = superVec_NG[i] + (D_matrix[j][i-N_param-1])*chi2[j]*(rep[j])**0.5
	if i>(N_param+D_dim) and i<=(N_param+D_dim+S_dim_red):
		for j in xrange(dif_vec.shape[0]):
			superVec_NG[i] = superVec_NG[i] +(D3_matrix[j][i-N_param-1-D_dim])*chi2[j]*(rep[j])**0.5
	if i>(N_param+D_dim+S_dim_red) and i<=(N_param+D_dim+S_dim_red+K_dim):
		for j in xrange(dif_vec.shape[0]):
			superVec_NG[i] = superVec_NG[i] + (D4_matrix[j][i-N_param-1-D_dim-S_dim_red])*chi2[j]*(rep[j])**0.5

print "Computation of vector B: DONE"
		

# Computation of the super_matrix

print "Computing matrix A. Starting by the building blocks..."

A_matrix = np.matmul(np.transpose(D_matrix),D_matrix)

B_matrix = np.matmul(np.transpose(D3_matrix),D_matrix)
print "B--> OK"
C_matrix = np.matmul(np.transpose(D4_matrix),D_matrix)
print "C--> OK"
X_matrix = np.matmul(np.transpose(D3_matrix),D3_matrix)
print "X--> OK"
Y_matrix = np.matmul(np.transpose(D4_matrix),D3_matrix)
print "Y--> OK"
W_matrix = np.matmul(np.transpose(D4_matrix),D4_matrix)
print "W--> OK"
P1_matrix = np.matmul(np.transpose(dif_vec_mod),dif_vec_mod)
P2_matrix = np.matmul(np.transpose(dif_vec_mod),D_matrix)
P3_matrix = np.matmul(np.transpose(dif_vec_mod),D3_matrix)
P4_matrix = np.matmul(np.transpose(dif_vec_mod),D4_matrix)
print "P1, P2, P3, P4--> OK"	


Sup1 = np.zeros((1,1))
Sup1[0][0] = 1

Sup2 = np.zeros((1,N_param))
for i in range(N_param):
	Sup2[0][i] = 0
	for j in range(dif_vec.shape[0]):
		Sup2[0][i] = Sup2[0][i] + dif_vec[j][i]*rep[j]
		
Sup3 = np.zeros((1,D_dim))
for i in range(D_dim):
	Sup3[0][i] = 0
	for j in range(dif_vec.shape[0]):
		Sup3[0][i] = Sup3[0][i] + D_matrix[j][i]*(rep[j])**0.5

Sup4 = np.zeros((1,S_dim_red))
for i in range(S_dim_red):
	Sup4[0][i] = 0
	for j in range(dif_vec.shape[0]):
		Sup4[0][i] = Sup4[0][i] + D3_matrix[j][i]*(rep[j])**0.5

Sup5 = np.zeros((1,K_dim))
for i in range(K_dim):
	Sup5[0][i] = 0
	for j in range(dif_vec.shape[0]):
		Sup5[0][i] = Sup5[0][i] + D4_matrix[j][i]*(rep[j])**0.5

print "Sup1, Sup2, Sup3, Sup4, Sup5 --> OK"	

print "Time to join the pieces and get matrix A ..."

super_matrix_NG = np.block([[Sup1,Sup2,Sup3,Sup4,Sup5],[np.transpose(Sup2),P1_matrix,P2_matrix,P3_matrix,P4_matrix],[np.transpose(Sup3),np.transpose(P2_matrix),A_matrix,np.transpose(B_matrix),np.transpose(C_matrix)],[np.transpose(Sup4),np.transpose(P3_matrix),B_matrix,X_matrix,np.transpose(Y_matrix)],[np.transpose(Sup5),np.transpose(P4_matrix),C_matrix,Y_matrix,W_matrix]])
super_matrix_inv_NG = np.linalg.inv(super_matrix_NG)	

super_matrix = np.block([[Sup1,Sup2,Sup3],[np.transpose(Sup2),P1_matrix,P2_matrix],[np.transpose(Sup3),np.transpose(P2_matrix),A_matrix]])
super_matrix_inv = np.linalg.inv(super_matrix)


print "Computation of A and its inverse: DONE"


M_vec_NG = np.matmul(np.transpose(superVec_NG),super_matrix_inv_NG)
M_vec = np.matmul(np.transpose(superVec),super_matrix_inv)

print "Computation of vector \mathcal{M}: DONE"	


# Now we need to decompress M_vec to obtain the original tensors M, S, and K. To do that we employ the dictionaries defined before


M_matrix = np.zeros((N_param,N_param))
M_matrix_NG = np.zeros((N_param,N_param))
P_vec = np.zeros((N_param))
P_vec_NG = np.zeros((N_param))
S_matrix = np.zeros((N_K,N_K,N_K))
K_matrix = np.zeros((N_K,N_K,N_K,N_K))

for i in range(super_dim_NG):
	if i == 0:
		beta = M_vec[i]
		beta_NG = M_vec_NG[i]
	if i<=N_param:
		P_vec[i-1] = M_vec[i] 
		P_vec_NG[i-1] = M_vec_NG[i]
	if i>N_param and i<=(N_param+D_dim):
		a = int(dictionary[i-N_param-1][0])
		b = int(dictionary[i-N_param-1][1])
		if a!=b:
			M_matrix_NG[a][b] = M_vec_NG[i]/2
			M_matrix_NG[b][a] = M_vec_NG[i]/2
			M_matrix[a][b] = M_vec[i]/2
			M_matrix[b][a] = M_vec[i]/2
		else:
			M_matrix_NG[a][b] = M_vec_NG[i]
			M_matrix[a][b] = M_vec[i]
	if i>(N_param+D_dim) and i<=(N_param+D_dim+S_dim_red):
		a = int(dictionary_S_red[i-N_param-D_dim-1][0])
		b = int(dictionary_S_red[i-N_param-D_dim-1][1])
		c = int(dictionary_S_red[i-N_param-D_dim-1][2])
		deg = int(dictionary_S_red[i-N_param-D_dim-1][3])
		S_matrix[a][b][c] = M_vec_NG[i]/deg
		S_matrix[a][c][b] = M_vec_NG[i]/deg
		S_matrix[b][a][c] = M_vec_NG[i]/deg
		S_matrix[b][c][a] = M_vec_NG[i]/deg
		S_matrix[c][a][b] = M_vec_NG[i]/deg
		S_matrix[c][b][a] = M_vec_NG[i]/deg								
	if i>(N_param+D_dim+S_dim_red) and i<=N_param+D_dim+S_dim_red+K_dim:
		a = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][0])
		b = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][1])
		c = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][2])
		d = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][3])
		deg = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][4])
		K_matrix[a][b][c][d] = M_vec_NG[i]/deg
		K_matrix[a][b][d][c] = M_vec_NG[i]/deg
		K_matrix[a][c][b][d] = M_vec_NG[i]/deg
		K_matrix[a][c][d][b] = M_vec_NG[i]/deg
		K_matrix[a][d][b][c] = M_vec_NG[i]/deg
		K_matrix[a][d][c][b] = M_vec_NG[i]/deg		
		K_matrix[b][a][c][d] = M_vec_NG[i]/deg
		K_matrix[b][a][d][c] = M_vec_NG[i]/deg
		K_matrix[b][c][a][d] = M_vec_NG[i]/deg
		K_matrix[b][c][d][a] = M_vec_NG[i]/deg
		K_matrix[b][d][a][c] = M_vec_NG[i]/deg
		K_matrix[b][d][c][a] = M_vec_NG[i]/deg
		K_matrix[c][b][a][d] = M_vec_NG[i]/deg
		K_matrix[c][b][d][a] = M_vec_NG[i]/deg
		K_matrix[c][a][b][d] = M_vec_NG[i]/deg
		K_matrix[c][a][d][b] = M_vec_NG[i]/deg
		K_matrix[c][d][b][a] = M_vec_NG[i]/deg
		K_matrix[c][d][a][b] = M_vec_NG[i]/deg
		K_matrix[d][b][c][a] = M_vec_NG[i]/deg
		K_matrix[d][b][a][c] = M_vec_NG[i]/deg
		K_matrix[d][c][b][a] = M_vec_NG[i]/deg
		K_matrix[d][c][a][b] = M_vec_NG[i]/deg
		K_matrix[d][a][b][c] = M_vec_NG[i]/deg
		K_matrix[d][a][c][b] = M_vec_NG[i]/deg				



print "Computation of the Covariance matrix obtained from the MC Posterior Gaussian Fit: DONE"	
print "\n Covariance matrix (MC Posterior Gaussian fit):"
Cov_matrix = np.linalg.inv(M_matrix)
print Cov_matrix
print "\n with BF being the best-fit vector found in the Markov chain:"
print BF
# Compute the corrected peak for the Gaussian fit
means = BF - np.dot(0.5,np.matmul(np.linalg.inv(M_matrix),P_vec))
print "\n Corrected position of the peak (for the MC Posterior Gaussian fit)"
print means
	
time_matrices = (time.clock() - time_start)
print "\n Computational time spent computing the matrices:", time_matrices,"s"
print "\n##############################################\n"		

np.savetxt('theta_corrected.txt', means, delimiter = '	') 
np.savetxt('Cov_matrix.txt', Cov_matrix, delimiter = '	') 

######################################### 
######################################### 
# MCMC in order to obtain the Markov chains needed to plot the 2D contours for the G and NG cases
# and also the marginalized 1D distributions for the NG scenario
######################################### 
######################################### 

# Number of sampling points we want to generate in the "marginalization" MonteCarlo.
N_MCMC = int(sys.argv[6])
	
limits = np.zeros((N_param,2))
for i in range(N_param):
	data1 = MCMC.iloc[:,columns_param[i]:columns_param[i]+1]
	limits[i][0] = data1.min()[0]
	limits[i][1] = data1.max()[0]	
	
# For the non-Gaussian distribution we need to carry out the Monte Carlo explicitly.

Cov_mat_MCMC = np.loadtxt(str(sys.argv[7]))	

# The following lines are only valid for the default example (LCDMwith curvature), and the input files provided in the github. 
# They rescale some of the covariances in order to match the definitions of the variables employed in the covariance file and
# MCMC file. If no rescalement is needed then the user should remove these lines.

Cov_mat_MCMC[0][0] = 10000*Cov_mat_MCMC[0][0]	
for i in range(1,N_param):
	Cov_mat_MCMC[i][0] = 100*Cov_mat_MCMC[i][0]
	Cov_mat_MCMC[0][i] = Cov_mat_MCMC[i][0]	

for i in range(N_param):
	Cov_mat_MCMC[i][26] = 1000*Cov_mat_MCMC[i][26]
	Cov_mat_MCMC[26][i] = Cov_mat_MCMC[i][26]	
Cov_mat_MCMC[26][26] = 1000*Cov_mat_MCMC[26][26]

for i in range(N_param):
	Cov_mat_MCMC[i][25] = 1000*Cov_mat_MCMC[i][25]
	Cov_mat_MCMC[25][i] = Cov_mat_MCMC[i][25]	
Cov_mat_MCMC[25][25] = 1000*Cov_mat_MCMC[25][25]

##########################################################	


# We reduce the standard deviation of all the parameters in order to improve the efficiency of our MCMC

for i in range(N_param):
	for j in range(N_param):
		Cov_mat_MCMC[i][j] = Cov_mat_MCMC[i][j]/100					 

print "We obtain a chain of", N_MCMC," from the MC Posterior NG fitted distribution."
print "This is to obtain the marginalized 1D and 2D distributions (histograms) needed to create the plots."
	
def dist_NG(vec0):
	suma = 0
	for i in range(1,super_dim_NG):			
		if i<=N_param:
			suma = suma + M_vec_NG[i]*(vec0[i-1]-BF[i-1]) 
		else:
			if i>N_param and i<=(N_param+D_dim):
				a = int(dictionary[i-N_param-1][0])
				b = int(dictionary[i-N_param-1][1])
				suma = suma + M_vec_NG[i]*(vec0[a]-BF[a])*(vec0[b]-BF[b]) 
			else:
				if i>(N_param+D_dim) and i<=(N_param+D_dim+S_dim_red):
					a = int(dictionary_S_red[i-N_param-D_dim-1][0])
					b = int(dictionary_S_red[i-N_param-D_dim-1][1])
					c = int(dictionary_S_red[i-N_param-D_dim-1][2])
					suma = suma + M_vec_NG[i]*(vec0[a]-BF[a])*(vec0[b]-BF[b])*(vec0[c]-BF[c])
				else:
					a = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][0])
					b = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][1])
					c = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][2])
					d = int(dictionary_K[i-N_param-D_dim-S_dim_red-1][3])
					suma = suma + M_vec_NG[i]*(vec0[a]-BF[a])*(vec0[b]-BF[b])*(vec0[c]-BF[c])*(vec0[d]-BF[d])
	return np.exp(-0.5*suma)	

#for i in range(N_points):
#	print chi2[i]
#	print -2*math.log(dist_NG(MCMC_matrix[i]))
#	print "########"

# This part of the code looks for the maximum value of the fitting distribution around BF. 
# This is used in the modified Metropolis-Hastings algorithm. 

N_gen = 6000
x0 = BF
vec_gen = np.zeros((N_gen))
for i in range(N_gen):	
	random_vec = np.random.multivariate_normal(x0, Cov_mat_MCMC,1)	
	x1 = random_vec[0,:]
	vec_gen[i] = dist_NG(x1)

maxi = max(vec_gen)
#print vec_gen
#print maxi


###########################################################

		
labels = np.zeros((N_points))
weights = np.zeros((N_points))
for i in range(N_points):
	weights[i] = rep[i] / True_N_points
	labels[i] = i	
		
# Modified Metropolis-Hastings algorithm
		
points_NG1 = []
x0 = BF
dist_NG_x0 = dist_NG(x0)
counter = 1
counter2 = 0
count_problem = 0
for i in range(N_MCMC):	
	check = -N_param
	# Check that the generated point by the MC falls inside the box set by the minimum and maximum values of all the parameters 
	# If it doesn't then another point is generated, and we repeat the process till this condition is fulfilled.
	while check<0:
		random_vec = np.random.multivariate_normal(x0, Cov_mat_MCMC,1)	
		for j in range(N_param):			
	 		if (random_vec[0][j]<limits[j][1]) and (random_vec[0][j]>limits[j][0]):
				check = check + 1
		if check<0:
			check = -N_param			
	x1 = random_vec[0,:]
	dist_NG_x1 = dist_NG(x1)
	alpha = dist_NG_x1/dist_NG_x0
	
	# Even if the generated point is inside the box, we must be sure that it has fallen in a "good" region
	# of parameter space
	if dist_NG_x1>maxi:
		unif_end = choice(labels,size=1,p=weights)
		x1 = MCMC_matrix[int(unif_end[0])]
		dist_NG_x1 = dist_NG(x1)
		count_problem = count_problem+1
		alpha = 2.
				
	unif = np.random.uniform(0,1,1)
	if alpha > unif:
		vec_trans = []
		for j in range(N_param):
			vec_trans.append(x0[j])
		vec_trans.append(counter)
		points_NG1.append(vec_trans)
		counter2 = counter2 + 1
		print i
		x0 = x1
		dist_NG_x0 = dist_NG_x1
		counter = 1
	else:
		counter = counter + 1
vec_trans = []
for j in range(N_param):
	vec_trans.append(x0[j])
vec_trans.append(counter)
points_NG1.append(vec_trans)
counter2 = counter2 + 1
	
print "Number of times that the MC generates a point in a bad region of parameter space:", count_problem	
	
points_NG = np.zeros((counter2,N_param+1))		
for i in range(counter2):	
	for j in range(N_param+1):	
		points_NG[i][j] = points_NG1[i][j]
	
np.savetxt('points_NG.txt', points_NG, delimiter = '	') 
				
time_MCMC = (time.clock() - time_start)

print "\n Computational time spent till the NG MCMC is finished (in s):", time_MCMC
print "\n"	
		
		




