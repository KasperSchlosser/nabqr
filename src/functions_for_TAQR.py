import numpy as np
import scipy.linalg
import time
from scipy import linalg as la
from sklearn.linear_model import QuantileRegressor
from helper_functions import *
import pandas as pd
from icecream import ic

import torch
def opdatering_final(X, Xny, IX, Iy, Iex, Ih, Ihc, beta, Rny, K, n, xB, P, tau, i, bins, n_in_bin):
    # print("n+i-1 = ", n+i-1)
    # print("dim of Xny: ", Xny.shape) 
    # print("Ih coming in: ", Ih)
    # clean +inf and -inf from Xny
    Xny[Xny == np.inf] = 1
    Xny[Xny == -np.inf] = 1

    # print("Disp xNy last row: ", Xny[-1,:])
    Xny = np.vstack((Xny, X[n + i - 2, :]))  # The newest available observation is added to the design matrix 
    # - 1 -1, since python 0 indexes
    #print(n_in_bin)
    #Index = np.arange(len(Xny) - 1)  # An indexset corresponding to the old design matrix is created
    #print(Xny, Iy)
    Index = np.arange(Xny[:-1,Iy].shape[0])  # An indexset corresponding to the old design matrix is created
    #print("Index: line 14 : ", Index)
    #print("Index: line11 : ", Index)
    j = 1  # The while loop determines a number j s.t. the new explanatory variable is in the interval (bins[j], bins[j+1]]
    #print("Xny[-1, Iex]: ", Xny[-1, Iex], "bins: ", bins)
    while Xny[-1, Iex] > bins[j]: # bins[j]
        j += 1
    j -= 1
    #print("j: ", j)
    
    # The index set of the design matrix in this interval is determined and the oldest element in the interval is marked for deletion.
    In = Index[(Xny[:-1, Iex] > bins[j]) & (Xny[:-1, Iex] <= bins[j + 1])]
    #print("In, line 17: ", In)
    # "min of In: ", np.min(In))
    if In.size > 0:
        Leav = np.min(In)
    else:
        Leav = 1 # trying this to just simply remove the oldest observation from the design matrix/rolling windows ye... TODO: check correctness
        # This is at least what happened in the MATLAB script most often... 
    # print("Leav: ", Leav)
    # minIL is a marker if minIL=0 then the leaving variable is in Ih, and it cannot be removed before a simplex step have been taken away from this.
    # print("line 20: ", np.min(np.abs(Ih - Leav)))
    minIL, Inmin = np.min(np.abs(Ih - Leav)), np.argmin(np.abs(Ih - Leav))
    # print("minIL, Inmin : ", minIL, Inmin)
    # The if statement take one simplex step with an objective function with zero loss on the leaving variable. This simplex step ensure that the row marked for deletion can leave the design matrix.
    if minIL == 0 and len(In) == n_in_bin:
        cB = (P < 0) + P * tau
        invXh = np.linalg.inv(Xny[Ih, IX])
        # g = -np.dot((P * (Xny[Ihc, IX] @ invXh[:, Inmin])).T, cB)
        g = -(P * (Xny[Ihc, IX] @ invXh[:, Inmin])).T @ cB
        # h = np.vstack((invXh[:, Inmin], -P * (Xny[Ihc, IX] @ invXh[:, Inmin]))) * np.sign(g)
        h = np.vstack((invXh[:, Inmin], -P * (Xny[Ihc, IX] @ invXh[:, Inmin]))) @ np.sign(g)
        sigma = np.zeros(n - K)
        # print("n-K: ", n-K, "K: ", K)
        hm = h[K:]
        xBm = xB[K:]
        xBm[xBm < 0] = 0
        tolerance = 1e-10 # by the MATLAB paper, the tolerance was quite arbitrary, so I chose 1e-8 to try for now.
        sigma[hm > tolerance] = xBm[hm > tolerance] / hm[hm > tolerance]
        sigma[hm <= tolerance] = np.inf
        alpha, q = np.min(sigma), np.argmin(sigma)

        # print("line 46")
        z = xB - alpha * h
        Ihm = Ih[Inmin]
        Ih[Inmin] = Ihc[q]
        Ihc[q] = Ihm

        xB = z
        xB[q + K] = alpha

        P[q] = np.sign(g) + (g == 0)
        Ih = np.sort(Ih)
        Ihc, IndexIhc = np.sort(Ihc), np.argsort(Ihc)
        P = P[IndexIhc]
        xBm = xB[K:]
        xBm = xBm[IndexIhc]
        xB[K:] = xBm
        beta = xB[:K]

    # The residual of the one step ahead prediction is calculated
    # rny = X[n + i - 1, Iy] - X[n + i - 1, IX] @ beta
    # print("n+i-1: ", n+i-1, "Iy: ", Iy, "IX: ", IX)
    # print(X[(n+i-2), Iy])
    rny = X[n + i  -2, Iy] - X[n + i -2, IX] @ beta
    Rny = np.append(Rny, rny)

    # This residual is used to update P and xB
    if rny < 0:
        P = np.append(P, -1)
        xB = np.append(xB, -rny)
    else:
        P = np.append(P, 1)
        xB = np.append(xB, rny)
    # Ihc = np.append(Ihc, n + 1) # prev. MATLAB code
    Ihc = np.append(Ihc, n)

    # If the bin is filled then the updating of the indexsets is a little tricky since not all of the indexset is updated, if the bin is not filled then updating is straight forward.
    if len(In) == n_in_bin:
        print("ever activated? ", i)
        i += 1
        Stay = np.ones(len(Ihc), dtype=bool)
        Stay[Ihc == Leav] = False
        P = P[Stay]
        Ihc = Ihc[Stay]
        Xny = Xny[np.sort(np.hstack((Ih, Ihc))), :]
        xBm = xB[(K):] #K+1 in MATLAB code
        xBm = xBm[Stay]
        xB = xB[:-1]
        xB[(K):] = xBm
        Ihc[Ihc > Leav] -= 1
        Ih[Ih > Leav] -= 1
    else:
        n += 1

    return Ih, Ihc, xB, Xny, Rny, P, n, i

def rq_initialiser_final(X, r, beta, n):
    # if np.all(r == r[Index]):
    #     print("line 85, we good")
    # If the number of zero elements in r is equal to rank(X), then the work is essentially done.
    # Otherwise, the if statement will find the index set Ih s.t. X(Ih)*beta=y(Ih) and X(Ih)^(-1)*y(Ih)=beta,
    # the important note being that X(Ih) has an inverse.
    # This is done by using the LU transform of a non-quadratic matrix.
    ic(len(r))
    ic(n)

    Index = np.arange(n)  # Index is the indexset s.t. r=r(Index)
    if np.sum(r == 0) > len(beta): # and len(r) > 0: # usually doesn't run
        Lr0 = np.sum(r == 0)
        rs, Irs = np.sort(np.abs(r)), np.argsort(np.abs(r))
        Irs = Irs[:Lr0]
        Xh = X[Irs, :]
        # print(scipy.linalg.lu(Xh, permute_l=True)[1:])
        P, L , U = scipy.linalg.lu(Xh, permute_l=False)#[1:]
        # print("P: ", P)
        In = np.arange(Lr0)
        rI = np.zeros(Lr0 - len(beta), dtype=int)
        for i in range(len(beta), Lr0):
            rI[i - len(beta)] = In[P[:, i] == 1]
        # print("line 85, rI:", rI)
        r[Irs[rI]] = 1e-15

    # The indexsets, P, and xB are defined
    index_to_index = r.flatten() == 0
    # if len(index_to_index) != len(Index):
    if len(index_to_index) < len(Index):
        # add as many values as the difference in their lengths
        index_to_index = np.append(index_to_index, np.zeros(len(Index) - len(index_to_index), dtype=bool))

    Ih = Index[index_to_index]

    Ihc = np.setdiff1d(Index[:len(r)], Ih)
    # Ihc = Index[np.abs(r) > 0] ### ORIGINAL line... should be the same :) as the next line

    P = np.sign(r[Ihc])
    r[np.abs(r) < 1e-15] = 0
    xB = np.hstack((beta, np.abs(r[Ihc])))

    return xB, Ih, Ihc, P

def rq_simplex_alg_final(Ih, Ihc, n, K, xB, Xny, IH, P, tau):
    # print("line 116:", Xny.shape, Ih.shape)
    # invXh = np.linalg.inv(Xny[Ih, :])
    # invXh = np.linalg.solve(Xny[Ih, :], np.eye(K))
    # print(type(Xny[1,2]))
    # print(np.finfo(np.longdouble))

    # matrix_to_be_inverted = matrix(Xny[Ih, :])
    # invXh = la.solve(matrix_to_be_inverted, np.eye(K).astype(np.longdouble))
    # invXh = matrix_to_be_inverted**-1
    #print("Ih: ", Ih)
    #print("Xny og Ih shape: ", Xny.shape, Ih.shape)
    invXh = la.inv(Xny[Ih, :])

    # invXh = np.matrix(invXh.tolist(), dtype=np.float64)

    cB = (P < 0) + P * tau
    cC = np.vstack((np.ones(K) * tau, np.ones(K) * (1 - tau))).reshape(-1,1)

    #IB2 = -(P @ (np.ones((1,K)) * Xny[Ihc, :])) @ invXh
    # IB2 = -(np.diag(P) @ Xny[Ihc, :]) @ invXh

    # print("shapes , P, ones, Xny, invXh: ", P.shape, np.ones((1,K)).shape, Xny[Ihc, :].shape, invXh.shape, (np.ones((1,K)) * Xny[Ihc, :]).shape)
    
    #IB2 = - (P.reshape(-1,1) @ (np.ones((1,K)) * Xny[Ihc, :])) @ invXh
    # IB2 = -np.dot(P * (np.ones((1,K)) * Xny[Ihc, :]), invXh)
    IB2 = -np.dot(P.reshape(-1, 1) * (np.ones((1, K)) * Xny[Ihc, :]), invXh)
    # print(IB2.shape, "<-- IB2 shape new")
    # print(cB.shape, "<-- cB shape")
    # print(cB)
    g = IB2.T @ cB # we here only select the first K entries, since I believe g should be of length K
    # print("****g**** : ", g, g.shape)
    d = cC - np.vstack((g, -g)).reshape(-1,1)
    d[np.abs(d) < 1e-15] = 0
    d = d.flatten()

    md, s = np.sort(d), np.argsort(d)
    s = s[md < 0]
    md = md[md < 0]
    c = np.ones(len(s))
    c[s > (K-1)] = -1
    C = np.diag(c)
    s[s > (K-1)] -= K
    # print(invXh[s, :].shape , C.shape)
    # print(IB2[s, :].shape , C.shape)
    # OK UNTIL HERE ..... 26/3-24 Bastian. 
    # print("stacked shape: ", np.vstack([invXh[s, :] , IB2[s, :] ]).shape)
#    h = np.vstack([invXh[s, :] , IB2[s, :] ]) @ C
    # print(invXh[:,s] , IB2[:,s] , C)
    h = np.vstack([invXh[:,s] , IB2[:,s] ]) @ C 
    # h = np.vstack([np.take(invXh, s, axis=1), np.take(IB2, s, axis=1)]) @ C

    
    # print("h shape: ", h.shape, h)
    alpha = np.zeros(len(s))
    q = np.zeros(len(s), dtype=int)
    xm = xB[(K):] # In MATLAB code it is K+1: here... TODO: check if this is correct, think it is!
    xm[xm < 0] = 0
    # hm = h[1]
    hm = h[K:,:]
    cq = np.zeros(len(s))
    tol1 = 1e-12 # tolerance sat here by me, since the MATLAB code had it as a quite arb. number
    # print(hm)

    # if abs(hm).min() < 1e-8:
    #     print("hm is too small, iteration: ", IH.shape[0])
        
    # print("hm: ", hm)
    # print("xm: ", xm)
    for k in range(len(s)):
        sigma = xm.copy()
        sigma[hm[:, k] > tol1] = xm[hm[:, k] > tol1] / hm[hm[:, k] > tol1, k]
        sigma[hm[:, k] <= tol1] = np.inf
        alpha[k], q[k] = np.min(sigma), np.argmin(sigma)
        cq[k] = c[k]
    # print("sigma: ", sigma)
    # if np.isinf(alpha).all():
    #     for k in range(len(s)):
    #         alpha[k] = 0.01
    #         q[k] = 0
          
    # print("alpha: ", alpha)
    gain = md * alpha
    # print("gain: ", gain)
    Mgain, IMgain = np.sort(gain), np.argsort(gain)
    CON = np.inf
    j = 0

    if len(gain) == 0:
        gain = 1
    else:
        while CON > 1e6 and j < len(s):
            j += 1
            IhMid = Ih.copy()
            shifter = 0
            IhMid[s[IMgain[j - 1 + shifter]]] = Ihc[q[IMgain[j - 1 + shifter]]]
            IhMid = np.sort(IhMid)
            #print("inside while loop shape of IH, IHMid: ", IH.shape, IhMid.shape, "len of IH", len(IH))
            # if np.min(np.sum(np.abs(IH - IhMid.reshape(1,-1)), axis=1)) == 0:
            #print("shape af IhmId*ones: ", (IhMid*np.ones(IH.shape)).shape)
            # IH = IH[0]

            if IH.shape[1] <= 1:
                # we good

                #print("J: ",j)
                IH = IH.T.reshape(-1,1)
            IhMid = IhMid.reshape(1,-1)
                
            #print("IH: ", IH.shape)
            #print("IHMid: ", IhMid.shape )

            # IH2 = np.hstack([IH, IhMid.T]) # why should IH keep being bigger here.... TODO: check if this is correct 
            #print("IH post: ", IH2.shape)

#            print("ones: ", np.ones((1, IH.shape[1])).shape)
          #  print("j: ", j, "and the shape: ", np.abs(IH - IhMid.reshape(-1,1)  @ np.ones((1,(IH.shape[1])))).shape)
           # print("number: ", np.min((sum(np.abs(IH - IhMid.reshape(-1,1) @ np.ones((1,(IH.shape[1]))))))))
            #print("numberRRRR: ", np.sum((np.abs(IH-IhMid.T * np.ones((1,IH.shape[1]))  )), axis = 0) )
            # if np.min(sum((np.abs(IH - IhMid @ np.ones((1,(IH.shape[1]))))))) == 0:
            
            if np.min(np.sum((np.abs(IH-IhMid.T * np.ones((1,IH.shape[1])) )), axis = 0) ) == 0:
                CON = np.inf
            else:
                # print("IhMid: ", IhMid, IhMid.shape)
                # print(Xny[IhMid, :])
                CON = np.linalg.cond(Xny[(IhMid.flatten()), :])
        # the error is before this... print("IMgain: ", IMgain)
        s = s[IMgain[j - 1 + shifter]]
        q = q[IMgain[j - 1 + shifter]]
        cq = cq[IMgain[j - 1 + shifter]]
        alpha = alpha[IMgain[j - 1 + shifter]]
        #print("IH: ", IH.shape)
        #print("IHMid: ", IhMid.shape)
        IH = np.hstack((IH, IhMid.T)) # why should IH keep being bigger here.... TODO: check if this is correct 
        # IH = IH.reshape(2,-1)
        #print("IH post: ", IH.shape)
        h = h[:, IMgain[j - 1 + shifter]]
        gain = gain[IMgain[j - 1 + shifter]]
        md = md[IMgain[j - 1 + shifter]]

    return CON, s, q, gain, md, alpha, h, IH, cq

def rq_purify_final(xB, Ih, Ihc, P, K, Xny, yny):
    """
    This function takes care of infeasible points in a simplex formulation
    of a quantile regression problem. The underlying assumption in this
    function is that there are no restrictions in the problem.

    The updating can therefore be done by recalculating all residuals and
    coefficients.

    The assumption is further that we are in a position s.t.
    Xny*Xny(Ih)^(-1)*yny(Ih)=yny+residuals
    K=rank(Xny)

    Input as in rq_simplex_final
    """
    invXh = np.linalg.inv(Xny[Ih, :])
    xB = np.hstack((invXh @ yny[Ih], yny[Ihc] - Xny[Ihc, :] @ invXh @ yny[Ih]))
    P = np.sign(xB[K:])
    P[P == 0] = 1
    xB[K:] = np.abs(xB[K:])
    return xB, P

def rq_simplex_final(X, IX, Iy, Iex, r, beta, n, tau, bins, n_in_bin):
    """
    rq_simplex_final calculates the solution to an adaptive simplex
    algorithm for a quantile regression problem. The function uses
    knowledge of the solution at time t to calculate the solution at
    time t+1.

    The basic idea is that the solution to the quantile regression
    problem can be written as:
    y(t) = X(t)'*beta + r(t)

    where beta = X(h)^(-1)*y(h) for some index set h. Simplex algorithm
    is now used to calculate the optimal h at time t+1 based on the solution
    at time t. So basically, the function uses h(t) as a starting guess for
    the simplex algorithm to iterate to h(t+1). This function is described
    in the master thesis "Modelling of Uncertainty in Wind Energy Forecast"
    by Jan K. Moeller, URL http://www.imm.dtu.dk/pubdb/p.php?4428.

    Inputs to the function:
    X        : The design matrix for the linear quantile regression problem,
               or rather it contains the design matrix, see the three next
               input variables.
    IX       : An index set, it refers to the columns of X which is the
               design matrix.
    Iy       : One index refers to the matrix X, the column Iy of X contains
               the response corresponding to the explanatory variables in the
               design matrix.
    Iex      : An index referring to a grouping variable, refers to a column of
               X this may or may not be a part of the design matrix. This is
               used in the updating algorithm.
    r        : The residuals from a solution to the quantile regression
               problem based on the first n (see below) elements of X. Such a
               solution could be obtained by the "rq" method in "R". This is
               only used to initialize the solution.
    beta     : The solution corresponding to the residuals in the vector r
               above.
    n        : The number of elements in r, i.e., y(1:n) = X(1:n,IX)*beta + r(1:n).
    tau      : The required probability, the solution in r and beta should be
               based on this probability.
    bins     : A vector defining a partition of an interval that covers all
               elements in X(1:end,Iex). This is used for the updating
               procedure of the design matrix. If bins = [-Inf,Inf], then the
               updating procedure is on a gliding window.
    n_in_bin : Number of elements in the bins defined above, this is one
               number, so the number of elements will be the same in each of
               the bins after some time.

    Remarks:
    This implementation requires that the number of elements in each bin of
    the initial solution is less than or equal to n_in_bin.

    Output:
    N    : The number of simplex steps to get the solutions at time t+1
           given the solution at time t. This is a vector.
    BETA : A matrix with each column being the solution to the quantile
           regression problem, corresponding to the matrix X(n+1:end,1:end).
    GAIN : This is a parameter used only to analyze the method. It is the
           gain in the loss function in each simplex step. If this is very
           large at some points, it can be taken as a sign of the problem
           being ill-posed.
    Ld   : This is only used for analyzing the algorithm in its own right. It
           gives the number of descent directions in each simplex step.
    Rny  : This is the residual of the one-step-ahead prediction of the
           method.
    Mx   : This is only used for analyzing the algorithm in its own right. It
           is the minimum of the constraint solution to the simplex
           formulation. This should be larger than or equal to zero. If this
           becomes large in absolute value, then it is an indication of the
           algorithm having problems at this point.
    Re   : The reliability on the training set for each point along the
           solution. If everything is good, then this should follow Theorem
           2.3 in the reference mentioned above. I.e., this is close to tau at
           all points.
    CON1 : The condition number of the matrices X(h(t)).
    T    : The time used for each iteration.

    References:
    [1] J. K. Møller (2006), Modeling of Uncertainty in Wind Energy
           Forecast. Master Thesis, Informatics and Mathematical Modelling,
           Technical University of Denmark. Available at
           http://www.imm.dtu.dk/pubdb/p.php?4428.

    [2] H. B. Nielsen (1999), Algorithms for Linear Optimization, an
          Introduction. Course note for the DTU course "Optimization and Data
          Fitting 2". Available at http://www.imm.dtu.dk/courses/02611/
    """

    # Initialize output vectors and matrices
    GAIN = np.array([0])
    Rny = np.array([0])
    Ld = np.array([0])
    mx = 0
    T = np.zeros(len(X[:, Iy]))
    Mx = np.zeros(len(X[:, Iy]))
    N = np.zeros((100, 2))
    N_num_of_simplex_steps = []
    N_size_of_the_design_matrix_at_time_k = []
    CON1 = np.array([0])
    CON2 = np.array([[0, 0]])
    BETA = beta.reshape(1, -1)

    # Initialize internal variables for the function
    K = len(beta)  # K is the number of explanatory variables in X

    # The design matrix at the starting point for the algorithm.
    # This will be the observation matrix in the simulation.
    # So each solution will be based on Xny at the given time point.
    Xny = X[:(n), :] # maybe it shouldn't be n-1 here anyways.... 

    H = np.zeros((1, K))  # This is used to track solutions that the algorithm has visited

    tolmx = 1e-15  # This is a tolerance to determine when an infeasible
                   # point should be fixed and when it should be considered zero.

    j = 0  # Counter of the number of simplex steps in each iteration

    LX = len(X[:, Iy])  # The stop criterion of the simulation

    i = 2  # Internal counter used by the updating procedure described below

    k = 0  # Time counter for the algorithm, k will be the time from the beginning of the algorithm

    # The function rq_initialiser_final initializes the parameters needed for
    # the simplex algorithm based on the solution given as input to the function
    #print("n length", n, "r length", len(r))
    xB, Ih, Ihc, P = rq_initialiser_final(Xny[:, IX], r, beta, n)
    # pre allocate Re
    Re = np.zeros(LX - n)
    # The while loop makes the simulation through the data set, in each step
    # either i or n is updated. n is the total number of elements in the
    # design matrix, in the first iterations n will be updated later it will be i.
    # print("i, n, LX", i, n, LX)
    while i + n < LX:
        k += 1  # The time counter is updated
        t = time.time()  # Zero point for CPU time
        
        Re[k] = np.sum(P < 0) / n  # Reliability at time k
        mx = np.min(xB[K:])  # The minimum of the basic solution, this should be larger than zero

        # Infeasible points are taken care of by rq_purify_final
        if j > 0 and mx < -tolmx:
            xB, P = rq_purify_final(xB, Ih, Ihc, P, K, Xny[:, IX], Xny[:, Iy])
            mx = np.min(xB[K:])

        Mx[k] = mx  # The minimum of the basic solution is stored
        j = 0  # Simplex step counter is set to zero to start the new time step

        beta = xB[:K]  # The solution at time k is extracted and stored

        BETA = np.vstack((BETA, beta))

        # The design matrix and other variables needed for the simplex
        # procedure are updated
        Ih, Ihc, xB, Xny, Rny, P, n, i = opdatering_final(
            X, Xny, IX, Iy, Iex, Ih, Ihc, beta, Rny, K, n, xB, P, tau, i, bins, n_in_bin
        )

        # print("Ih after opdatering final: ", Ih)
        # Infeasible points are set equal to zero to avoid taking a simplex step in this direction
        # print("line 458: ", Ih.shape, IH.shape)
        IH = Ih.reshape(-1, 1)  # The index set h is stored in each simplex step, this is done
                                # to prevent the solution from going back and forth between
                                # equally good solutions
        # print("IH, Ih shapes line 458: ", IH.shape, Ih.shape) # think it is fine, just keep reshaping here^
        # Numbers needed to perform the simplex step are calculated in rq_simplex_alg_final
        CON, s, q, gain, md, alpha, h, IH, cq = rq_simplex_alg_final(
            Ih, Ihc, n, K, xB, Xny[:, IX], IH, P, tau
        )
        CON1 = np.append(CON1, CON)  # The condition number of the matrix X(h(t+1)) is stored

        # print("Time: ", time.time() - t, "s, k: ", k, "n: ", n, "i: ", i, "j: ", j, "Con: ", CON, "gain: ", gain, "md: ", md, "alpha: ", alpha, "mx: ", mx)
        # print(f"Time: {time.time() - t:.2f} s, k: {k}, n: {n}, i: {i}, j: {j}, Con: {CON:.2f} ,  gain: {gain:.2f}, md: {md:.2f}, alpha: {alpha:.4f}, mx: {mx:.2f}")
        # print("Ih, s: ", Ih, s)
        # The while loop continues until the optimal solution is reached or one
        # of the two stop criteria are violated. These are a maximal number of
        # simplex steps, and the condition number of the next solution.
        while gain <= 0 and md < 0 and j < 24 and CON < 1e6:
            GAIN = np.append(GAIN, gain)  # The gain for the step is stored
            j += 1  # The simplex counter is updated
            # print("line 416_ alpha: ", alpha, "q: ", q)
      
            z = xB - alpha * h  # z is the new solution to the problem..
            
            # The index sets defining the solution are updated
            # print("Ih, s, j: ", Ih, s, j)
            IhM = Ih[s]
            IhcM = Ihc[q]
            Ih[s] = IhcM
            Ihc[q] = IhM
            P[q] = cq  # The sign of the newest residual

            # The basic solution is updated
            xB = z
            xB[q + K] = alpha

            # The index sets are ordered s.t. they are in increasing order
            Ih = np.sort(Ih)
            Ihc, IndexIhc = np.sort(Ihc), np.argsort(Ihc)
            P = P[IndexIhc]
            xBm = xB[K:]
            xBm = xBm[IndexIhc]
            xB[K:] = xBm
           

            # rq_simplex_alg_final calculates the numbers needed to perform the next simplex step
            CON, s, q, gain, md, alpha, h, IH, cq = rq_simplex_alg_final( # 
                Ih, Ihc, n, K, xB, Xny[:, IX], IH, P, tau
            )
            # print("s, j : " , s, j)
            CON1 = np.append(CON1, CON)  # The condition number is stored

        # N[k, 0] = j  # Number of simplex steps at time k
        N_num_of_simplex_steps.append( j)
        N_size_of_the_design_matrix_at_time_k.append(n)
        # N[k, 1] = n  # Size of the design matrix at time k
        T[k] = time.time() - t  # CPU time used to update the solution
    
    N = np.hstack([N_num_of_simplex_steps, N_size_of_the_design_matrix_at_time_k])

    # return N, BETA[1:], GAIN[1:], Ld, Rny, Mx, Re, CON1[1:], T
    return N, BETA, GAIN[1:], Ld, Rny, Mx, Re, CON1[1:], T

def one_step_quantile_prediction(X_input, Y_input, n_init, n_full, quantile = 0.5, already_correct_size = False, n_in_X = 5000):
    
    '''
    As input, this function should take the entire training set, and based on the last n_init observations,
    calculate residuals and coefficients for the quantile regression.

    '''

    from functions import run_r_script

    # INPUTS THAT SHOULD BE OUTSIDE OF FUNCTION; BUT ARE STAYING INSIDE AS OF NOW - no premature optimization!!
    # solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    # model_QR = QuantileRegressor(quantile=quantile, solver = 'highs')
    
    # TODO add a check that r and n are of same length

    assert n_init <= n_full - 2, "n_init must be less than n_full" # should it be equal == ? Only if we want one step prediction... 11/6-24.

    if type(X_input) == pd.DataFrame:
        X_input = X_input.to_numpy()

    if type(Y_input) == pd.Series or type(Y_input) == pd.DataFrame:
        Y_input = Y_input.to_numpy()

    n,m = X_input.shape
    
    print("X_input shape: ", X_input.shape)   
        
    # get shapes
    full_length, p = X_input.shape

    # if not already_correct_size:
    # get input ready
    X = X_input[:n_full, :].copy()
    Y = Y_input[:n_full]
    # else:
    #     X = X_input
    #     Y = Y_input
    #     n_full = len(Y)

    X_for_residuals = X[:n_init, :]
    Y_for_residuals = Y[:n_init]

    # save them for to be used in rq... X_for_residuals and Y_for_residuals
    # X_for_residuals.to_csv("X_for_residuals.csv") #only use if pd dataframe
    np.savetxt("X_for_residuals.csv", X_for_residuals, delimiter=",")
    np.savetxt("Y_for_residuals.csv", Y_for_residuals, delimiter=",")
    # Y_for_residuals.to_csv("Y_for_residuals.csv") # --||--

    # calculate residuals
    run_r_script("X_for_residuals.csv", "Y_for_residuals.csv", tau = quantile)
    # quantile_fit = model_QR.fit(X_for_residuals, Y_for_residuals)
    # Y_predict = quantile_fit.predict(X_for_residuals)
    # residuals = Y_for_residuals - Y_predict

    # Define a converter function to ignore the first column
    def ignore_first_column(s):
        return float(s)

    # Read the CSV file
    residuals = np.genfromtxt(
        'rq_fit_residuals.csv', 
        delimiter=',', 
        skip_header=1, 
        usecols=(1,),  # Only read the second column
        converters={0: ignore_first_column}  # Ignore the first column
    )



    # residuals = np.loadtxt("rq_fit_residuals.csv", delimiter=",", skip_rows = 1)

    beta_init = np.genfromtxt(
        'rq_fit_coefficients.csv', 
        delimiter=',', 
        skip_header=1, 
        usecols=(1,),  # Only read the second column
        converters={0: ignore_first_column}  # Ignore the first column
    )

    # beta_init = np.loadtxt("rq_fit_coef.csv", delimiter=",", skip_header = 1)

    print("len of beta_init: ", len(beta_init))
    # print(beta_init)
    print("There is: ", sum(residuals == 0), "zeros in residuals", "and", sum(abs(residuals) < 1e-8), "close to zeroes")
    print("p: ", p)

    # add 1s to beta_init to match length of p
    beta_init = np.append(beta_init, np.ones(p-len(beta_init)))
    
    r_init = set_n_closest_to_zero(arr = residuals , n = len(beta_init))

    print(sum(r_init==0), "r_init zeros")

    # get the data ready
    # print("X shape: ", X.shape, "Y shape: ", Y.shape, "random choice shape: ", np.random.choice([1,1], size=n_full).shape)
    X_full = np.column_stack((X, Y, np.random.choice([1,1], size=n_full)))
    IX = np.arange(p)
    Iy = p
    Iex = p + 1
    bins = np.array([-np.inf , np.inf]) # rolling, Currently not active, since n_in_bin = full length...
    # beta_init = quantile_fit.coef_
    tau = quantile
    n_in_bin = int(1.0*full_length)
    print("n_in_bin", n_in_bin)


    # call the function
    n_input = n_in_X
    N, BETA, GAIN, Ld, Rny, Mx, Re, CON1, T = rq_simplex_final(X_full, IX, Iy, Iex, r_init, beta_init, n_input, tau, bins , n_in_bin ) # here we set n_init to 5000, to see what happens...
    # find the actual prediction
    # print(BETA.shape, "BETA SHAPE")
    # print(X_input.shape, "X_input full shape")
    # print(X_input[(n_init+1):(n_full), :].shape, "X_input shape") 
    y_pred = np.sum((X_input[(n_input+2):(n_full), :] * BETA[1:,:]), axis = 1) # TODO WHETHER IT IS +1, or 2 here or minus, should def. be checked
    y_actual = Y_input[(n_input):(n_full-2)]
    print(y_pred.shape, "y_pred shape")
    print(y_actual.shape, "y_actual shape")
    # plt.figure()
    # plt.plot(y_pred)
    # plt.plot(y_actual)
    # plt.show()
    y_actual_quantile = np.quantile(y_actual, quantile)
    #print("Quantile: ", quantile, "y_actual_quantile: ", y_actual_quantile)
    # return the prediction, the actual value and the coefficients
    return y_pred, y_actual, BETA

