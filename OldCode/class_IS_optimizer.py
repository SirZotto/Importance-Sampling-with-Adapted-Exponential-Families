import numpy as np
import cvxpy as cp

def A_normal(lam_s, Lam_s, eps=1e-6, penalty=1e6):
    Lam_reg = Lam_s + eps * np.eye(Lam_s.shape[0])
    try:
        sol = np.linalg.solve(Lam_reg, lam_s)
    except np.linalg.LinAlgError:
        return penalty
    quad = -0.25 * np.dot(lam_s, sol)
    try:
        sign, logdet = np.linalg.slogdet(-2.0 * Lam_reg)
        if sign <= 0 or not np.isfinite(logdet):
            return penalty
    except np.linalg.LinAlgError:
        return penalty
    return quad - 0.5 * logdet

def block_diag(mats):
    sizes = [m.shape[0] for m in mats]
    total = sum(sizes)
    out = np.zeros((total, total), dtype=np.result_type(*mats))
    k = 0
    for m in mats:
        n = m.shape[0]
        out[k:k+n, k:k+n] = m
        k += n
    return out

class IS_optimizer(object):
    def __init__(self, list_of_z, samples, p_joint, f):
        self.list_of_z = list_of_z
        self.samples = samples
        self.p_joint = p_joint
        self.f = f
        
        sample0 = self.samples[0]
        assert sample0.ndim == 2, "Each sample must be a 2D array of shape (S, d)"
        self.S, self.d = sample0.shape
        
        
    def exp_dist(self, bounds=None, cvxpy_solver=cp.SCS, cvxpy_verbose=False, cvxpy_max_iters=50000, cvxpy_eps=1e-5, cvxpy_acceleration_lookback=10, cvxpy_scale=1.0):
            
            d, S = 1, self.S
            f = self.f
            p_joint = self.p_joint
            list_of_z = self.list_of_z      

            samples = self.samples
            samples = np.asarray(samples)                               
            if samples.ndim == 1:                                       
                samples = samples[None, :]
            assert samples.shape[1] == S, "Each sample must be a vector of length S."  
            M = samples.shape[0]                                       
            samples = samples.reshape(M, S)    
            
            def helping_z(XX):
                z_array = np.zeros(S)
                for s in range(S):
                    z_array[s] = list_of_z[s](XX[:s])
                return z_array

            if np.any(samples <= 0):                                   
                raise ValueError("Exponential distributions take no samples with values smaller or equal to 0")


            alpha = cp.Variable(S*d)

            cons = []
            eps_pos = 1e-5                               
            if bounds is not None:
                cons += list(bounds(alpha))
            else: 
                cons += [alpha >= eps_pos]              
                    
            z_T_all_XX = [None]*M
            z_all_XX = [None]*M
            sum_log_z_all_XX = [None]*M
            
            for i, XX in enumerate(samples):
                z_all_XX[i] = helping_z(XX)
                z_T_all_XX[i] = z_all_XX[i] * XX 
                sum_log_z_all_XX[i] = np.sum(np.log(- z_all_XX[i]))
                            
            z_T_matrix = np.array(z_T_all_XX)
                            
            weight = np.array([p_joint(XX) * (f(XX)**2) * np.exp(-sum_log_z_all_XX[i]) for XX in samples])

            alpha_sum_log = cp.sum(cp.log(alpha))
            log_weight = cp.Constant(np.log(np.maximum(weight, 1e-300)))             
            exponent_vec = -(z_T_matrix @ alpha) + log_weight                       
            expression = -alpha_sum_log + cp.log_sum_exp(exponent_vec)                

            obj = cp.Minimize(expression)
            prob = cp.Problem(obj, cons)
            prob.solve(solver=cvxpy_solver, verbose=cvxpy_verbose, max_iters=cvxpy_max_iters, eps=cvxpy_eps, acceleration_lookback=cvxpy_acceleration_lookback, scale=cvxpy_scale)
            
            if alpha.value is None:
                raise RuntimeError(f"Solver failed (status: {prob.status}).")

            alpha_opt = alpha.value.reshape(S, d)
            
            def create_list_of_z_alpha(alpha):
                alpha = alpha.reshape(S, d)
                list_of_z_alpha = []
                for s in range(S):
                    alpha_s = alpha[s]
                    z_s = list_of_z[s]
                    def z_alpha_s_factory(z_s=z_s, alpha_s=alpha_s):
                        def z_alpha_s(list_of_x_s):
                            eval_z_s = z_s(list_of_x_s)
                            return alpha_s * eval_z_s
                        return z_alpha_s
                    list_of_z_alpha.append(z_alpha_s_factory())
                return list_of_z_alpha

            list_of_z_alpha = create_list_of_z_alpha(alpha_opt)
            
            def q_sampler(sample_size):
                sample_set = [None]*sample_size
                eps = 1e-8
                for i in range(sample_size):
                    XX = []
                    for s in range(S):
                        z_alpha_s = list_of_z_alpha[s]
                        past_xs = [] if len(XX) == 0 else XX                                           
                        lam_s = abs(z_alpha_s(past_xs))                         
                        lam_s = float(max(lam_s, eps))                           
                        x_s = np.random.exponential(scale=1.0/lam_s, size=d)
                        XX.append(x_s)
                    XX_matrix = np.stack(XX, axis=0)
                    sample_set[i] = XX_matrix
                return sample_set

            def q_alpha(XX):
                if np.any(np.array(XX) <= 0):
                    raise ValueError("This exponential pdf take no samples with values smaller or equal to 0")
                list_of_x_s = []
                eval_z = np.zeros(S)
                eval_T = np.zeros(S)
                for s in range(S):
                    x_s = XX[s]                         
                    eval_z[s] =  list_of_z_alpha[s](list_of_x_s) 
                    eval_T[s] = x_s
                    list_of_x_s.append(x_s)
                return np.exp(np.clip(eval_z.T @ eval_T, -700, 700)) * np.prod(np.maximum(-eval_z, 1e-300))  

            return q_sampler, q_alpha, alpha_opt, list_of_z_alpha


    def normal(self, bounds=None, cvxpy_solver=cp.SCS, cvxpy_verbose=False, cvxpy_max_iters=20000, cvxpy_eps=1e-5, cvxpy_acceleration_lookback=10, cvxpy_scale=1.0):
        
        d, S = self.d, self.S
        f = self.f
        p_joint = self.p_joint
        q_0_inv = (((2 * np.pi) ** (-d / 2)) ** S)**(-1)
        list_of_z = self.list_of_z

        def T(x):
            return x, np.outer(x, x)

        samples = self.samples
        M = len(samples)


        alpha = cp.Variable(S*d)

        cons = []
        if bounds is not None:
            cons += list(bounds(alpha))
            
        frobenius_all_XX = [None]*M
        quad_term_matrix_all_XX = [None]*M
        log_det_all_XX = [None]*M
        lam_times_tau_all_XX = [None]*M
        exprs = [None]*M
        
        for i, XX in enumerate(samples):
            frob_sum = 0.0
            logdet_sum = 0.0
            Diag_lam_blocks = []
            Lam_inv_blocks = []
            lam_tau_segments = []

            for s in range(S):
                lam_s, Lam_s = list_of_z[s](list(XX[:s]))
                tau_s, Tau_s = T(XX[s])

                frob_sum += np.vdot(Lam_s, Tau_s)

                L = np.linalg.cholesky(-2.0 * Lam_s)
                logdet_sum +=  np.sum(np.log(np.diag(L)))

                Diag_lam_blocks.append(np.diag(lam_s))
                Lam_inv_blocks.append(np.linalg.inv(Lam_s))
                lam_tau_segments.append(lam_s * tau_s)

            frobenius_all_XX[i] = frob_sum
            log_det_all_XX[i] =  logdet_sum

            D_lambda = block_diag(Diag_lam_blocks)
            Lambda_inv = block_diag(Lam_inv_blocks)
            Q = (D_lambda @ Lambda_inv @ D_lambda)
            quad_term_matrix_all_XX[i] = -0.125 * (Q + Q.T)
            lam_times_tau_all_XX[i] = np.concatenate(lam_tau_segments)

            Q_i = quad_term_matrix_all_XX[i]
            b_i = lam_times_tau_all_XX[i]
            exprs[i] = cp.quad_form(alpha, Q_i) - (alpha @ b_i)
        
        weight = np.array([p_joint(XX) * (f(XX)**2) * q_0_inv * np.exp(-frobenius_all_XX[i] - log_det_all_XX[i]) for i, XX in enumerate(samples)])
        
        log_w = np.log(weight + 1e-300)                     
        m = float(np.max(log_w))                            

        exponent = cp.hstack(exprs)
        shifted = exponent + log_w - m                    
        obj = cp.Minimize(cp.sum(cp.exp(shifted)))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cvxpy_solver,verbose=cvxpy_verbose,max_iters=cvxpy_max_iters,eps=cvxpy_eps,acceleration_lookback=cvxpy_acceleration_lookback,scale=cvxpy_scale) 

        if alpha.value is None:                            
            raise RuntimeError(f"Solver failed (status: {prob.status}).") 

        alpha_opt = alpha.value.reshape(S, d)

        def create_list_of_z_alpha(alpha):
            alpha = alpha.reshape(S, d)
            list_of_z_alpha = []
            for s in range(S):
                alpha_s = alpha[s]
                z_s = list_of_z[s]
                def z_alpha_s_factory(z_s=z_s, alpha_s=alpha_s):
                    def z_alpha_s(list_of_x_s):
                        lam_s, Lam_s = z_s(list_of_x_s)
                        return alpha_s * lam_s, Lam_s
                    return z_alpha_s
                list_of_z_alpha.append(z_alpha_s_factory())
            return list_of_z_alpha

        list_of_z_alpha = create_list_of_z_alpha(alpha_opt)

        def q_sampler(sample_size):
            sample_set = []
            eps = 1e-6
            for _ in range(sample_size):
                XX = []
                for s in range(S):
                    z_alpha_s = list_of_z_alpha[s]
                    past_xs = [] if s == 0 else XX
                    lam_s, Lam_s = z_alpha_s(past_xs)
                    Lam_reg = -2 * Lam_s + eps * np.eye(d)
                    Sigma_s = np.linalg.inv(Lam_reg)
                    mu_s = Sigma_s @ lam_s
                    x_s = np.random.multivariate_normal(mu_s, Sigma_s)
                    XX.append(x_s)
                XX_matrix = np.stack(XX, axis=0)
                sample_set.append(XX_matrix)
            return sample_set
        
        def q_alpha(XX):
            q_0 = q_0_inv**(-1)
            list_of_x_s = []
            exp_XX = 0.0
            for s in range(S):
                x_s = XX[s]
                lam_s, Lam_s = list_of_z_alpha[s](list_of_x_s)
                tau_x_s, Tau_x_s = T(x_s)
                exp_XX += (np.dot(lam_s, tau_x_s) + np.vdot(Lam_s, Tau_x_s) - A_normal(lam_s, Lam_s))
                list_of_x_s.append(x_s)
            return q_0 * np.exp(np.clip(exp_XX, -700, 700))

        return q_sampler, q_alpha, alpha_opt, list_of_z_alpha 

