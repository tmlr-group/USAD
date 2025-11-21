import numpy as np
import torch
import time
import sys

sys.path.append('~/USAD')
from models import *
from exp.dataloader import load_data, check_device, load_train
from baselines.USAD.utils_USAD import *

def SAD(path, N1, N_ip, rs, check, model, n_test, n_per, alpha, ref, sigma_un, n_perturb, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD, reduction_method, reduction_dim, stat_names=["mean", "variance", "uncertainty"], adaptive=False, prop=None):
    device = check_device()
    model = model.to(device)
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    projection_matrix, target_dim = None, None
    if reduction_method is not None:
        try:
            # Try to load cached projection matrix (no calibration data needed)
            projection_matrix, target_dim = compute_projection_matrix(
                model, path, device, reduction_method, reduction_dim, n_perturb, sigma_un
            )
        except ValueError:
            # Cache doesn't exist, need to compute from calibration data
            log("Projection matrix cache not found, loading calibration data...")
            data_sample = load_train(path, rs)
            if data_sample is not None:
                projection_matrix, target_dim = compute_projection_matrix(
                    model, path, device, reduction_method, reduction_dim, n_perturb, sigma_un,
                    calibration_data=data_sample
                )
            else:
                log("Warning: No calibration data available, skipping dimension reduction")
                projection_matrix, target_dim = None, None

    (P, Q), (P_rep, Q_rep) = load_data(path, N1, rs, check, model, ref = ref, class_idx = None) #XUNYE: class_idx 从0-9，只load cifar10的10个类，输入None就是随机类
    P_sigma = sigma_from_perturb(P, sigma_un, n_perturb, model, path, device, "P",
                                 projection_matrix=projection_matrix, reduction_dim=target_dim,
                                 reduction_method=reduction_method)
    Q_sigma = sigma_from_perturb(Q, sigma_un, n_perturb, model, path, device, "Q",
                                 projection_matrix=projection_matrix, reduction_dim=target_dim,
                                 reduction_method=reduction_method)

    H_SAD = np.zeros(n_test)
    P_SAD = np.zeros(n_test)

    # # resample B times to get the covariance matrix
    B = 500
    # sigma_B = cov_from_B(B, P_rep, P_sigma, N1, N_ip, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD, stat_names, n_perturb, sigma_un)
    sigma_B = torch.eye(len(stat_names)).to(device)
    # sys.exit()
    
    np.random.seed(rs*1021 + N1)
    test_time = 0
    for k in range(n_test):

        P_idx = np.random.choice(len(P), N1, replace=False)
        # print(P_idx)
        # P_te = P[P_idx]
        Sigma_x = P_sigma[P_idx]

        Q_idx = np.random.choice(len(Q), N_ip, replace=False)
        # Q_te = Q[Q_idx]
        Sigma_y = Q_sigma[Q_idx]
        Q_te = Q_rep[Q_idx]

        if prop:
            n_from_P = int(round(prop * N_ip))
            n_from_Q = N_ip - n_from_P

            if n_from_P > 0:
                # Select from P and P_rep, not P_te/P_rep_te:
                P_indices = np.random.choice(len(P), n_from_P, replace=False)
            else:
                P_indices = np.array([], dtype=int)
            if n_from_Q > 0:
                Q_indices = np.random.choice(len(Q_te), n_from_Q, replace=False)
            else:
                Q_indices = np.array([], dtype=int)

            new_Q_te = []
            new_Sigma_y = []
            if n_from_P > 0:
                new_Q_te.append(P_rep[P_indices])          # use from P
                new_Sigma_y.append(P_sigma[P_indices])
            if n_from_Q > 0:
                new_Q_te.append(Q_te[Q_indices])
                new_Sigma_y.append(Sigma_y[Q_indices])
            if new_Q_te:
                Q_te = torch.cat(new_Q_te, dim=0)
                Sigma_y = torch.cat(new_Sigma_y, dim=0)

        Z_rep = torch.cat([P_rep[P_idx], Q_te], dim=0)
        # Z_rep = Z_rep @ projection_matrix  

        Z_sigma = torch.cat([Sigma_x, Sigma_y], dim=0)
        
        start_time = time.time()
        h, p_value = permutation_test(Z_rep, Z_sigma, sigma_B, N1, alpha, n_per, b_MMD, b_VD, b_CD, kernel_MMD, kernel_VD, kernel_CD, stat_names)
        test_time += time.time() - start_time

        H_SAD[k] = h
        P_SAD[k] = p_value
        # print("p_value: ", p_value)

    log("SAD avg test time: ", test_time/n_test, "s") # average 1s per test
    torch.cuda.empty_cache()
    return H_SAD, P_SAD, test_time/n_test