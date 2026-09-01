from __future__ import annotations
import numpy as np


def add_intercept(X):
    X=np.asarray(X,dtype=float)
    return np.column_stack([np.ones(X.shape[0]),X])


def validate_coords(coords):
    a=np.asarray(coords,dtype=float)
    if a.ndim!=2 or a.shape[0]==0 or a.shape[1]<1:
        raise ValueError('coords must be a non-empty 2D array')
    if not np.all(np.isfinite(a)):
        raise ValueError('coords contains NaN or infinite values')
    return a


def compute_diagnostics(y_true,y_pred,hat_matrix=None,compute_gwr_stats=False):
    y=np.asarray(y_true,dtype=float).reshape(-1); f=np.asarray(y_pred,dtype=float).reshape(-1)
    r=y-f; rss=float(r@r); n=y.size
    tss=float(((y-y.mean())**2).sum()); r2=1-rss/tss if tss>0 else (1.0 if rss==0 else 0.0)
    out={'r2':float(r2),'rss':rss,'rmse':float(np.sqrt(np.mean(r*r))),'mae':float(np.mean(np.abs(r)))}
    if hat_matrix is None:
        out.update({'effective_params':np.nan,'adj_r2':np.nan,'aic':np.nan,'aicc':np.nan,'bic':np.nan})
        return out
    S=np.asarray(hat_matrix,dtype=float); tr=float(np.trace(S)); trss=float(np.sum(S*S))
    edf=float(n-2*tr+trss); enp=float(2*tr-trss)
    safe=max(rss,np.finfo(float).tiny); den=n-2-tr
    aic=float(n*np.log(safe/n)+n*np.log(2*np.pi)+n+2*(tr+1))
    aicc=float(np.inf if den<=0 else n*np.log(safe/n)+n*np.log(2*np.pi)+n*(n+tr)/den)
    bic=float(n*np.log(safe/n)+n*np.log(2*np.pi)+n+np.log(n)*(tr+1))
    adj=float(1-(1-r2)*(n-1)/(edf-1)) if edf>1 else np.nan
    out.update({'effective_params':tr,'adj_r2':adj,'aic':aic,'aicc':aicc,'bic':bic})
    if compute_gwr_stats:
        out.update({'trace_S':tr,'trace_StS':trss,'enp_v1':tr,'edf_v1':float(n-tr),'enp_v2':enp,'edf_v2':edf,'enp':enp,'edf':edf})
    return out


def format_summary(title, values):
    lines=[title, '='*len(title)]
    lines.extend(f'{k}: {v}' for k,v in values.items())
    return '\n'.join(lines)
