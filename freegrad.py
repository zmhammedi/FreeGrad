# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:40:05 2021

@author: Zakaria Mhammedi
"""
import numpy as np

def loss(w,x,y):
    return  (y - np.dot(w,x))**2

def gradient(w,x,y):
    return -2* x * (y - np.dot(w,x))

def norm(x):
    return np.sqrt(np.dot(x,x))


def diagonal_freegrad_learn(X,Y,restart=False, project=False, autoradius=True,\
                   radius=1, epsilon=1):
    # The feature matrix X contains T rows and d columns
    # Y is a label vector of length T
    X =  np.array(X)
    Y = np.array(Y)
    T, d = X.shape
    
    # Initialize the "sufficient statistics"
    w = np.zeros(d)     # Prediction vector
    G = np.zeros(d)     # Sum of gradients 
    V = np.zeros(d)     # Sum of squared coordinate-wise gradients
    S = np.zeros(d)     # Sum of normalized coordinate-wise gradients (used for restarts)
    h1 = np.zeros(d)    # Absolute values of initial non-zero coordinate-wise gradients
    ht = np.zeros(d)    # Maximum absolute values of coordinate-wise gradients up to t
    Ht = 0
    sum_normalized_grad_norm = 1 # Sum of normalized gradients (used for projections)
    
    # Initializing the loss
    L = 0
    
    # Main loop 
    for t in range(T):
        # Get the feature vector and the label
        x = X[t]
        y = Y[t]
        
        # Norm of prediction 
        w_norm = norm(w)

        if autoradius:
            project_radius = epsilon * np.sqrt(sum_normalized_grad_norm)
        else:
            project_radius = radius
          
        
        w_project = w
        if project:
            if w_norm > project_radius:
                print('Projected')
                w_project = w * project_radius/w_norm
                
        # Get the loss
        L += loss(w_project,x,y)
        #print(loss(w,x,y))
        print(np.dot(w_project,x))
        
        # Compute gradient information
        g = gradient(w_project,x,y)
        
        # Cutkosky's varying constrains' reduction:
        # Alg. 1 in http://proceedings.mlr.press/v119/cutkosky20a/cutkosky20a.pdf with sphere sets
        if project and w_norm > project_radius and np.dot(g,w) < 0:
            tilde_g = g - np.dot(g,w/w_norm) * w/w_norm 
        else:
            tilde_g = g

        clipped_g = tilde_g
        
        # Update stuff
        for i in range(d):
            # Clip the gradient
            abs_tilde_g  = abs(tilde_g[i])
            
            # Only do something if non-zero grad observed
            if abs_tilde_g == 0:
                continue
            
            # Update the hints
            tmp_ht = ht[i]
            if h1[i]==0:
                h1[i]=abs_tilde_g
                ht[i]=abs_tilde_g
                V[i]+=tilde_g[i]**2
            elif abs_tilde_g > tmp_ht:
                clipped_g[i] *= tmp_ht/abs_tilde_g  
                ht[i] = abs_tilde_g
                
            # Check for restarts
            if restart and ht[i]/h1[i] > S[i]+2:
                print('Restarted')
                h1[i]=ht[i]
                G[i] = clipped_g[i]
                V[i] = clipped_g[i]**2
            else:
                G[i] += clipped_g[i]
                V[i] += clipped_g[i]**2
                
            # Check this is the same as the implementation (the tmp_ht part)
            if tmp_ht>0:
                S[i] += abs(clipped_g[i])/tmp_ht
                
        # Compute prediction
        absG = abs(G)
        w = - G * epsilon * h1**2 *(2*V+ht*absG)/(2*(V+ht*absG)**2 * np.sqrt(V))\
            * np.exp(absG**2/(2 *V + 2 * ht * absG))
       
        # Update statistics for projections
        norm_clipped_g = norm(clipped_g)
        if norm_clipped_g > Ht:
            Ht = norm_clipped_g
        if Ht > 0:
            sum_normalized_grad_norm +=  norm_clipped_g/Ht 
            
    return L/T, w


def freegrad_learn(X,Y,restart=False, project=False, autoradius=True,\
                   radius=1, epsilon=1):
    # The feature matrix X contains T rows and d columns
    # Y is a label vector of length T
    X =  np.array(X)
    Y = np.array(Y)
    T, d = X.shape
    
    # Initialize the "sufficient statistics"
    w = np.zeros(d)  # Prediction vector
    G = np.zeros(d)  # Sum of gradients 
    V = 0            # Sum of squared gradient norms 
    S = 0            # Sum of normalized gradients (used for restarts)
    h1 = 0           # Norm of initial non-zero gradient
    ht = 0           # Maximum norm of observed gradients up to t
    Ht = 0  
    sum_normalized_grad_norm = 1 # Sum of normalized gradients (used for projections)
    
    # Initializing the loss
    L = 0
    
    # Main loop 
    for t in range(T):
        # Get the feature vector and the label
        x = X[t]
        y = Y[t]
        
        # Norm of prediction 
        w_norm = norm(w)

        if autoradius:
            project_radius = epsilon * np.sqrt(sum_normalized_grad_norm)
        else:
            project_radius = radius
          
        w_project = w
        if project:
            if w_norm > project_radius:
                # TODO remove the print
                print('Projected')
                w_project = w * project_radius/w_norm
                
        # Get the loss
        L += loss(w_project,x,y)
        # TODO remove the print
        print(np.dot(w_project,x))
        
        # Compute gradient information
        g = gradient(w_project,x,y)
        
        # Cutkosky's varying constrains' reduction:
        # Alg. 1 in http://proceedings.mlr.press/v119/cutkosky20a/cutkosky20a.pdf with sphere sets
        if project and w_norm > project_radius and np.dot(g,w) < 0:
            tilde_g = g - np.dot(g,w/w_norm) * w/w_norm 
        else:
            tilde_g = g

        clipped_g = tilde_g
        
        ## Update stuff
        # Clip the gradient 
        norm_tilde_g  = norm(tilde_g)
        
        # Only do something if non-zero grad observed
        if norm_tilde_g == 0:
            continue
        
        # Update the hints
        tmp_ht = ht
        if h1==0:
            h1=norm_tilde_g
            ht=norm_tilde_g
            V+=norm_tilde_g**2
            norm_clipped_g = norm_tilde_g
        elif norm_tilde_g > tmp_ht:
            clipped_g *= tmp_ht/norm_tilde_g  
            ht = norm_tilde_g
            norm_clipped_g = tmp_ht
        
        # Check for restarts
        if restart and ht/h1 > S +2:
            # TODO remove the print
            print('Restarted')
            h1=ht
            G = clipped_g
            V = norm_clipped_g**2
        else:
            G += clipped_g
            V += norm_clipped_g**2
            
        # Check this is the same as the implementation (the tmp_ht part)
        if tmp_ht>0:
            S += norm_clipped_g/tmp_ht
                
        # Compute prediction
        normG = norm(G)
        w = - G * epsilon * h1**2 *(2*V+ht*normG)/(2*(V+ht*normG)**2 * np.sqrt(V))\
            * np.exp(normG**2/(2 *V + 2 * ht * normG))
       
        # Update statistics for projections  
        if norm_clipped_g > Ht:
            Ht = norm_clipped_g
        if Ht > 0:
            sum_normalized_grad_norm +=  norm_clipped_g/Ht 
            
    return L/T, w


if __name__ == '__main__':
    # For train5.txt
    X = [[1,-.6,-.2],[1,-.2,-.4],[1,-.4,-.8],[1,-.8,-.16],[1,-.16,-.32],[1,-.32,-.64]]
    Y = [1, 0.2, 3,1, 20, 0]
    
    # From train.txt
    #X = [[1,-1,-2],[1,-2,-4],[1,-2,-3],[1,-2,-3],[1,-2,-2],[1,-3,-2],[1,-2,-4],[1,-3,-2],[1,-2,-4],[1,-3,-6],[1,-1,-1],[1,-2,-3]]
    #Y = -1 * np.array([3,3,3,3,3,3,3,3,3,3,3,3])
    
    #L,w = freegrad_learn(X,Y)
    # L,w = freegrad_learn(X,Y, restart=True)
    # L,w = freegrad_learn(X,Y, project=True)
    #L,w = freegrad_learn(X,Y, restart=True, project=True, autoradius=False, radius=.20)
    L,w = diagonal_freegrad_learn(X,Y, restart=False, project=False, autoradius=False, radius=0.1, epsilon=1)
  
    print(f'Average loss = {L}, and final output w = {w}')