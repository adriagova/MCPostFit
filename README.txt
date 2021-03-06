
In this Python code, called MCPostFit, we implement the MonteCarlo Posterior Fit method presented in arXiv:2007.02615.

Any use of this method requires the corresponding citation of arXiv:2007.02615.

Some useful guidelines about how to use the code are provided in the document MCPostFit_user_guidelines.pdf of this repository. We provide, though, a brief
summary in the following lines. The compilation of the code has to be carried out as follows, adding seven arguments:

python MCPostFit.py arg1 arg2 arg3 arg4 arg5 arg6 arg7

where

arg1 = Name and location of the input MCMCM .txt file, i.e. the file containing the Markov chain
arg2 = Total number of parameters (cosmological+nuisance)
arg3 = Number of lines from the input MCMC file that the user wants to employ in the posterior fit. To use all of them just write "full"
arg4 = "ordered" or "random", depending on the way of picking the aforesaid number of lines from the input MCMC file
arg5 = Number of parameters employed to compute the cubic and quartic corrections of the fitting posterior distribution. It has to be lower than or equal to arg2
arg6 = Number of sampling points that the user want to generate in the "marginalization" MonteCarlo
arg7 = Name and location of the file that contains the covariance matrix employed to perform the jumps in the "marginalization" MonteCarlo

Example: compilation of the attached files, in case they are located in the same folder as MCPostFit:

python MCPostFit.py Omegak.txt 28 60000 random 17 1000000 Omegak_cov.txt
