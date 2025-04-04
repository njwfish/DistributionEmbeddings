import numpy as np
import pandas as pd 
import torch
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import pearsonr

def generate_set_mean_predictions(encoder, sets, X, ctrl_key, pert_keys):
    encoder = encoder.to('cuda') 

    ctrl_X = torch.tensor(X[sets[ctrl_key]]).to('cuda')
    pert_X = {k: torch.tensor(X[sets[k]]).to('cuda') for k in pert_keys}

    ctrl_S = encoder(ctrl_X.unsqueeze(0))
    
    pert_S = {k: encoder(pert_X[k].unsqueeze(0)) for k in pert_keys}
    pert_S_delta = {k: pert_S[k] - ctrl_S for k in pert_keys}

    pert_S = torch.cat([pert_S[k] for k in pert_keys], dim=0)
    pert_S_delta = torch.cat([pert_S_delta[k] for k in pert_keys], dim=0)

    ctrl_X_mean = torch.mean(ctrl_X, dim=0)
    pert_X_mean = {k: torch.mean(pert_X[k], dim=0) for k in pert_keys}
    pert_X_mean = torch.cat([pert_X_mean[k].unsqueeze(0) for k in pert_keys], dim=0)
    pert_X_delta = pert_X_mean - ctrl_X_mean.unsqueeze(0)

    pert_X_delta_recon = encoder.mean_predict(pert_S) - ctrl_X_mean
    return ctrl_X_mean.cpu().detach().numpy(), ctrl_S.cpu().detach().numpy(), pert_X_delta.cpu().detach().numpy(), pert_S_delta.cpu().detach().numpy(), pert_X_delta_recon.cpu().detach().numpy()


def r2_score(y_true, y_pred):
    """Calculate R² using Pearson correlation."""
    r = pearsonr(y_true, y_pred, axis=1)
    return (r[0]**2).mean()

def generate_ood_eval_predictions(encoder, sc_dataset):
    metrics = {}
    for cell_type in sc_dataset.sets.keys():

        ctrl_key = sc_dataset.control_pert
        pert_keys = [k for k in sc_dataset.sets[cell_type] if k != ctrl_key and k in sc_dataset.pert_embeddings]
        eval_pert_keys = [k for k in sc_dataset.eval_sets[cell_type] if k != ctrl_key and k in sc_dataset.pert_embeddings]

        with torch.no_grad():
            ctrl_X, ctrl_S, X_delta, S_delta, _ = generate_set_mean_predictions(
                encoder, sc_dataset.sets[cell_type], sc_dataset.X, ctrl_key, pert_keys
            )
            _, _, X_delta_eval, S_delta_eval, X_delta_recon_eval = generate_set_mean_predictions(
                encoder, sc_dataset.eval_sets[cell_type], sc_dataset.X, ctrl_key, eval_pert_keys
            )

        Z = np.vstack([sc_dataset.pert_embeddings[k] for k in pert_keys])
        Z_eval = np.vstack([sc_dataset.pert_embeddings[k] for k in eval_pert_keys])

        # Stage 1
        reg = KernelRidge(kernel='polynomial', degree=2, alpha=.5)
        reg.fit(Z, S_delta)
        S_delta_pred_eval = reg.predict(Z_eval)
        S_pred_eval = S_delta_pred_eval + ctrl_S

        # Stage 2
        S_pred_eval_tensor = torch.tensor(S_pred_eval).to('cuda')
        X_delta_pred_set = encoder.mean_predict(S_pred_eval_tensor).cpu().detach().numpy() - ctrl_X

        # Baseline on full mean
        reg = KernelRidge(kernel='polynomial', degree=2, alpha=.5)
        reg.fit(Z, X_delta)
        X_delta_pred_full = reg.predict(Z_eval)

        r2 = r2_score(X_delta_eval, X_delta_pred_full)
        mse = mean_squared_error(X_delta_eval, X_delta_pred_full)

        r2_set_stage1 = r2_score(S_delta_eval, S_delta_pred_eval)
        mse_set_stage1 = mean_squared_error(S_delta_eval, S_delta_pred_eval)

        r2_set_stage2 = r2_score(X_delta_eval, X_delta_pred_set)
        mse_set_stage2 = mean_squared_error(X_delta_eval, X_delta_pred_set)

        r2_recon = r2_score(X_delta_eval, X_delta_recon_eval)
        mse_recon = mean_squared_error(X_delta_eval, X_delta_recon_eval)

        metrics[cell_type] = {
            'set_r2_stage1': r2_set_stage1, 'set_mse_stage1': mse_set_stage1,
            'set_r2_stage2': r2_set_stage2, 'set_mse_stage2': mse_set_stage2,
            'full_r2': r2, 'full_mse': mse,
            'recon_r2': r2_recon, 'recon_mse': mse_recon
        }
    return metrics
