from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.spatial.distance import cdist
from .core import compute_diagnostics

@dataclass
class GWRResult:
    parameters: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    hat_matrix: np.ndarray

class BasicGWR:
    """Minimal standard GWR baseline for LG-GWR research."""
    def __init__(self, bandwidth='auto', kernel='bisquare', fit_intercept=True, *, adaptive=None):
        if kernel not in {'bisquare','gaussian','exponential'}: raise ValueError('invalid kernel')
        self.bandwidth=bandwidth; self.kernel=kernel; self.fit_intercept=bool(fit_intercept); self.adaptive=adaptive
    def _is_adaptive(self):
        if self.adaptive is not None: return bool(self.adaptive)
        return isinstance(self.bandwidth,Integral) and not isinstance(self.bandwidth,(bool,np.bool_)) or self.bandwidth=='auto'
    def _kernel(self,d,bw):
        v=d/bw
        if self.kernel=='bisquare': return np.where(v<1,(1-v*v)**2,0.0)
        if self.kernel=='gaussian': return np.exp(-0.5*v*v)
        return np.exp(-v)
    @staticmethod
    def _solve(X,y,w):
        Xw=X*w[:,None]; M=Xw.T@X
        try: C=np.linalg.solve(M,Xw.T)
        except np.linalg.LinAlgError: C=np.linalg.pinv(M)@Xw.T
        return C@y,C
    def _fit_at(self,Xd,y,D,bw,adaptive):
        n,p=Xd.shape; bet=np.zeros((n,p)); hat=np.zeros((n,n))
        for i in range(n):
            if adaptive:
                k=min(int(bw),n); b=float(np.partition(D[i],k-1)[k-1]); b=float(np.nextafter(max(b,1e-12),np.inf))
            else: b=float(bw)
            w=self._kernel(D[i],b); bet[i],C=self._solve(Xd,y,w); hat[i]=Xd[i]@C
        fit=np.einsum('ij,ij->i',Xd,bet)
        return bet,fit,hat
    def fit(self,X,y,coords):
        X=np.asarray(X,float); y=np.asarray(y,float).reshape(-1); coords=np.asarray(coords,float)
        if X.ndim==1:X=X[:,None]
        if X.shape[0]!=y.size or coords.shape[0]!=y.size: raise ValueError('same rows required')
        Xd=np.column_stack([np.ones(y.size),X]) if self.fit_intercept else X.copy(); D=cdist(coords,coords); adaptive=self._is_adaptive()
        if self.bandwidth=='auto':
            if adaptive:
                low=max(Xd.shape[1]+1,2,int(np.ceil(.05*y.size))); candidates=range(low,y.size+1)
            else:
                pos=D[D>0]; candidates=np.geomspace(max(pos.min()/2,1e-6),max(D.max()*2,1e-5),24)
            best=None
            for bw in candidates:
                bet,fit,hat=self._fit_at(Xd,y,D,bw,adaptive); score=compute_diagnostics(y,fit,hat,True)['aicc']
                if np.isfinite(score) and (best is None or score<best[0]): best=(score,bw,bet,fit,hat)
            if best is None: raise RuntimeError('bandwidth selection failed')
            _,bw,bet,fit,hat=best; self.bandwidth_=int(bw) if adaptive else float(bw)
        else:
            self.bandwidth_=int(self.bandwidth) if adaptive else float(self.bandwidth); bet,fit,hat=self._fit_at(Xd,y,D,self.bandwidth_,adaptive)
        self.adaptive_=adaptive; self.parameters_=bet; self.fitted_values_=fit; self.residuals_=y-fit; self.hat_matrix_=hat
        self.diagnostics_=compute_diagnostics(y,fit,hat,True); self.result_=GWRResult(bet,fit,self.residuals_,hat); return self
