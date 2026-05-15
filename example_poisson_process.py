from class_IS_optimizer import IS_optimizer
from class_natural_parameter_components import natural_parameter_definer
import numpy as np
from scipy.stats import expon

import matplotlib.pyplot as plt
from IS_evaluation import IS_evaluation, IS_evaluation2 

np.random.seed(34)

S = 100    
d=1
scale =1

def f(XX):
    return float(np.sum(XX))

def p_joint(XX):
    return float(np.prod(expon.pdf(XX, scale=scale)))

list_of_z = natural_parameter_definer(typ="stdExp_dist", beta=1).create_z_exp_dist(S=S,lamb = 1,combination_typ= "mean_in_typ")
#list_of_z = natural_parameter_definer(typ="constant", beta=1).create_z_normal(d, S, combination_typ="mean_in_typ")

training_size = 1000

samples_train = np.random.exponential(scale=scale, size=(training_size, S, d))
print(samples_train[0].shape)
print(p_joint(samples_train[0])*f(samples_train[0]))


opti = IS_optimizer(list_of_z, samples_train, p_joint, f)

#bounds = lambda x: [x>=0,x<=1]
q_sampler, q_alpha, alpha_opt, list_of_z_alpha = opti.exp_dist()
#q_sampler, q_alpha, alpha_opt, list_of_z_alpha = opti.normal()

print(alpha_opt)

M_eval = 3500

l = 1.96  # 95% confidence


if True:
    # === MC-Samples ===
    X_eval = np.random.exponential(scale=scale, size=(M_eval, S, d))

    # === IS-Samples ===
    X_evalIS = q_sampler(M_eval)

    list_of_timpoints = [100,500, 700, 1000,1500, 2000,2500,2700, 3000, M_eval]

    ##################################################################evaluation################################################
    if True:
        print("Evaluation 1 begins")
        IS_evaluation(
            f,
            p_joint,
            q_alpha,
            X_evalIS,
            X_eval,
            y_axes_lower_visual=85,
            y_axes_upper_visual=130,
            list_of_timepoints=list_of_timpoints,
            save_filename=r"C:\Studium\Semester_6\Bachelorarbeit\FINAL_CODE\IS_evaluation_results.csv",
            true_value=S,
            histogram_size= (14,7),  # NEW: controls figure size
            writin_size = 16,  
        )
    if False:
        print("Eval 2 begins")
        m_min = [1000,1000,1000,1000,1000,1000]
        m_max = [1500,2000,2500,2700,3000,M_eval]
        IS_evaluation2(
            f,
            p_joint,
            q_alpha,
            X_evalIS,
            X_eval,
            #y_axes_lower_visual=0.025,
            #y_axes_upper_visual=0.12,
            list_of_timepoints=list_of_timpoints,
            save_filename=r"C:\Studium\Semester_6\Bachelorarbeit\FINAL_CODE\IS_evaluation_results.csv",
            m_max=m_max,
            m_min=m_min,
            true_value= S
        )
