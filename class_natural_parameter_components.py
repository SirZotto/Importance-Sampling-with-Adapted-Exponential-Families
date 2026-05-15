import numpy as np
import random
from scipy.stats import norm
from scipy.stats import expon

def ReLU(x):
    return np.maximum(0, x)

def LReLU(x, beta):
    if x < 0:
        return x * beta
    else:
        return x

def shiftReLU(x, beta):
    if x < beta:
        return beta
    else:
        return x
    
def cappedReLU(x,beta):
    if x<beta:
        return np.maximum(0, x)
    else:
        return beta
    
def standardNormal(x,beta):
    if x >= -beta and x<= beta:
        return norm.pdf(x)
    else:
        return norm.pdf(beta)

def standardExp_dist(x,beta):
    if x >= -beta and x<= beta:
        return expon(scale=1).pdf(x)
    else:
        return expon(scale=1).pdf(beta)


class natural_parameter_definer(object):
    def __init__(self, typ=None, beta=None):
        def shiftReLUbeta(x): return shiftReLU(x, beta)
        def LReLUbeta(x): return LReLU(x, beta)
        def id(x): return x
        def const(x): return beta
        def cappedReLUbeta(x): return cappedReLU(x,beta)
        def stdNormal(x): return standardNormal(x,beta)
        def stdExp_dist(x): return standardExp_dist(x,beta)

        if typ == "ReLU":
            self.typ = ReLU

        elif typ == "LReLU":
            self.typ = LReLUbeta

        elif typ == "shiftReLU":
            self.typ = shiftReLUbeta

        elif typ == "id":
            self.typ = id

        elif typ == "constant":
            self.typ = const
            
        elif typ == "cappedReLU":
            self.typ =cappedReLUbeta
        
        elif typ == "stdNormal":
            self.typ = stdNormal
            
        elif typ == "abs":
            self.typ = abs
            
        elif typ == "stdExp_dist":
            self.typ = stdExp_dist
            
        else:
            self.typ = random.choice([ReLU, const, shiftReLUbeta, LReLUbeta,cappedReLUbeta, id, abs])  
            
    def create_z_exp_dist(self, S, lamb = None, combination_typ="mean_out_of_typ"):
        
        typ = self.typ
                
        if  lamb == None:
            lamb = 1
            
        def z0(list_empty):
            return -lamb
        
        list_of_z = [z0]  


        if combination_typ == "mean_out_of_typ":
        
            def combination(list_of_xs):
                list_of_xs = np.array(list_of_xs)
                sm1 = len(list_of_xs)
                typed_values = np.zeros(sm1)
                
                for s in range(sm1):
                    typed_values[s] = typ(list_of_xs[s])
                
                output = np.mean(typed_values)
        
                return -output
            

            combination_typ_for_USE = combination
            
        elif combination_typ == "mean_in_typ":
            
            def combination(list_of_xs):
                list_of_xs = np.array(list_of_xs)
                this_is_mean = np.mean(list_of_xs)
                output = typ(this_is_mean)

                return -output
            
            
            combination_typ_for_USE = combination
                
        else:
            raise ValueError(f"Invalid combination_typ '{combination_typ}'. Use 'mean_out_of_typ' or 'mean_in_typ'.")
        
        def make_z_s(comb_typ): 
            def z_s(list_of_xs):
                return comb_typ(list_of_xs)
            return z_s

        for _ in range(S - 1):
            list_of_z.append(make_z_s(combination_typ_for_USE)) 

        return list_of_z
            
            
    def create_z_normal(self, d, S, mu=None, SIG=None, combination_typ="mean_out_of_typ"):
        
        typ = self.typ
        
        if mu is None:
            mu = np.ones(d)
        if SIG is None:
            SIG = np.eye(d)   
        SIG_inv = np.linalg.inv(SIG)
        
        def z0(list_empty):
            return SIG_inv @ mu, -0.5 * SIG_inv
        
        list_of_z = [z0]  
        
        
        if combination_typ == "mean_out_of_typ":
            
            def combination(list_of_xs, sig_mat):
                list_of_xs = np.array(list_of_xs)
                sm1, d = list_of_xs.shape
                output_array = np.zeros(d)
                typed_values = np.zeros((sm1, d))

                for i in range(d):
                    for l in range(sm1):
                        typed_values[l, i] = typ(list_of_xs[l, i])
                    output_array[i] = np.mean(typed_values[:, i])
                    
                return output_array, -0.5 * sig_mat
            

            combination_typ_for_USE = combination
            
        elif combination_typ == "mean_in_typ":
            
            def combination(list_of_xs, sig_mat):
                list_of_xs = np.array(list_of_xs)
                sm1, d = list_of_xs.shape
                output_array = np.zeros(d)
                
                for i in range(d):
                    make_this_mean = np.zeros(sm1)
                    
                    for l in range(sm1):
                        make_this_mean[l] = list_of_xs[l, i]
                    
                    this_is_mean = np.mean(make_this_mean)
                    output_array[i] = typ(this_is_mean)

                return output_array, -0.5 * sig_mat
            
            
            combination_typ_for_USE = combination
                
        else:
            raise ValueError(f"Invalid combination_typ '{combination_typ}'. Use 'mean_out_of_typ' or 'mean_in_typ'.")
 
        def make_z_s(SIG_local, comb_typ):  
            def z_s(list_of_xs):
                lam_s, Lam_s = comb_typ(list_of_xs, SIG_local)
                return lam_s, Lam_s
            return z_s

        for _ in range(S - 1):
            list_of_z.append(make_z_s(SIG_inv, combination_typ_for_USE))  

        return list_of_z
    
        

        
        
            
        