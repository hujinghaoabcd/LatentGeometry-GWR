from __future__ import annotations
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from .core import add_intercept, validate_coords, compute_diagnostics, format_summary

ArrayLike=Union[np.ndarray,pd.DataFrame]
VectorLike=Union[np.ndarray,pd.Series]
BandwidthLike=Union[float,int,Tuple[float,float],None]

@dataclass(frozen=True)
class LGGWRPredictionResult:
    predictions: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    coords: np.ndarray
    latent_coords: np.ndarray
    feature_names: Tuple[str,...]
    def to_frame(self):
        d={'coord_0':self.coords[:,0],'coord_1':self.coords[:,1],'prediction':self.predictions,'intercept':self.intercepts}
        for i in range(self.latent_coords.shape[1]): d[f'latent_{i}']=self.latent_coords[:,i]
        for i,n in enumerate(self.feature_names): d[f'coef_{n}']=self.coefficients[:,i]
        return pd.DataFrame(d)

@dataclass(frozen=True)
class _OptimisationResult:
    matrix: np.ndarray; loss_history: Tuple[float,...]; best_loss: float; final_loss: float; n_iter: int; converged: bool; stop_reason: str

class LGGWR:
    """Research baseline for Latent-Geometry Geographically Weighted Regression."""
    _KERNELS={'gaussian','bisquare','exponential'}; _GEOMETRIES={'joint','separable'}; _INITIALISATIONS={'coordinate','random','pca'}; _SCALE_CONSTRAINTS={'frobenius','orthogonal','none'}
    def __init__(self,latent_dim=2,bandwidth=None,adaptive=False,kernel='gaussian',geometry='joint',learning_rate=.05,max_iter=100,tol=1e-6,lambda_reg=0.0,orthogonal_constraint=None,grad_clip=10.0,patience=20,select_bandwidth=True,random_state=None,verbose=False,*,fit_intercept=True,standardize_geometry=True,initialization='coordinate',n_restarts=1,scale_constraint='frobenius',bandwidth_updates=1):
        self.latent_dim=self._positive_int(latent_dim,'latent_dim'); self.bandwidth=bandwidth; self.adaptive=self._boolean(adaptive,'adaptive')
        self.kernel=self._choice(kernel,'kernel',self._KERNELS); self.geometry=self._choice(geometry,'geometry',self._GEOMETRIES)
        self.learning_rate=self._nonnegative_float(learning_rate,'learning_rate'); self.max_iter=self._nonnegative_int(max_iter,'max_iter'); self.tol=self._positive_float(tol,'tol'); self.lambda_reg=self._nonnegative_float(lambda_reg,'lambda_reg')
        self.grad_clip=self._positive_float(grad_clip,'grad_clip'); self.patience=self._positive_int(patience,'patience'); self.select_bandwidth=self._boolean(select_bandwidth,'select_bandwidth'); self.random_state=random_state; self.verbose=self._boolean(verbose,'verbose')
        self.fit_intercept=self._boolean(fit_intercept,'fit_intercept'); self.standardize_geometry=self._boolean(standardize_geometry,'standardize_geometry'); self.initialization=self._choice(initialization,'initialization',self._INITIALISATIONS); self.n_restarts=self._positive_int(n_restarts,'n_restarts'); self.bandwidth_updates=self._nonnegative_int(bandwidth_updates,'bandwidth_updates')
        if orthogonal_constraint is not None:
            if self._boolean(orthogonal_constraint,'orthogonal_constraint'): scale_constraint='orthogonal'
            warnings.warn('orthogonal_constraint is deprecated; use scale_constraint instead.',DeprecationWarning,stacklevel=2)
        self.scale_constraint=self._choice(scale_constraint,'scale_constraint',self._SCALE_CONSTRAINTS); self.orthogonal_constraint=self.scale_constraint=='orthogonal'
        if self.lambda_reg>0 and self.scale_constraint!='none': raise ValueError("lambda_reg must be 0 when scale_constraint fixes the matrix norm. Use scale_constraint='none' for ordinary L2 regularisation.")
        if self.scale_constraint=='none' and self.lambda_reg==0: warnings.warn("An unconstrained latent map without regularisation has an unidentified scale; consider scale_constraint='frobenius'.",RuntimeWarning,stacklevel=2)
        self._validate_bandwidth_spec(); self._reset_fit_state()
    @staticmethod
    def _boolean(v,n):
        if not isinstance(v,(bool,np.bool_)): raise TypeError(f'{n} must be boolean.')
        return bool(v)
    @staticmethod
    def _positive_int(v,n):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral): raise TypeError(f'{n} must be a positive integer.')
        v=int(v)
        if v<=0: raise ValueError(f'{n} must be greater than zero.')
        return v
    @staticmethod
    def _nonnegative_int(v,n):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral): raise TypeError(f'{n} must be a non-negative integer.')
        v=int(v)
        if v<0: raise ValueError(f'{n} must be non-negative.')
        return v
    @staticmethod
    def _positive_float(v,n):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Real): raise TypeError(f'{n} must be a positive real scalar.')
        v=float(v)
        if not np.isfinite(v) or v<=0: raise ValueError(f'{n} must be finite and greater than zero.')
        return v
    @staticmethod
    def _nonnegative_float(v,n):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Real): raise TypeError(f'{n} must be a non-negative real scalar.')
        v=float(v)
        if not np.isfinite(v) or v<0: raise ValueError(f'{n} must be finite and non-negative.')
        return v
    @staticmethod
    def _choice(v,n,c):
        if not isinstance(v,str): raise TypeError(f'{n} must be a string.')
        v=v.strip().lower()
        if v not in c: raise ValueError(f"{n} must be one of: {', '.join(sorted(c))}.")
        return v
    def _validate_bandwidth_spec(self):
        v=self.bandwidth
        if v is None:return
        if isinstance(v,tuple):
            if self.geometry!='separable' or len(v)!=2: raise ValueError('A bandwidth tuple is supported only in separable mode and must contain (h_g, h_a).')
            for x in v:
                if not np.isinf(x): self._positive_float(x,'bandwidth component')
            return
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Real): raise TypeError('bandwidth must be a positive scalar, a two-item tuple, or None.')
        self._positive_int(v,'adaptive bandwidth') if self.adaptive else self._positive_float(v,'bandwidth')
    def _reset_fit_state(self):
        for n in ['A_','B_','metric_matrix_','metric_contributions_','bandwidth_','latent_coords_','coefficients_','coef_','intercept_','fitted_values_','residuals_','hat_matrix_','diagnostics_','best_loss_','final_loo_loss_','stop_reason_','X_train_','X_design_','y_train_','coords_train_','attrs_train_','coords_geometry_','attrs_geometry_','u_train_','feature_names_in_','n_features_in_','coord_center_','coord_scale_','attr_center_','attr_scale_','constant_attribute_mask_']: setattr(self,n,None)
        self.bandwidth_history_=[]; self.restart_scores_=[]; self.loss_history_=[]; self.n_iter_=0; self.converged_=False; self.feature_names_=(); self.geometry_feature_names_=(); self._legacy_intercept_input_=False; self._is_fitted=False
    @staticmethod
    def _numeric_2d(v,n):
        raw=v.to_numpy() if isinstance(v,pd.DataFrame) else v; a=np.asarray(raw,float)
        if a.ndim==1:a=a[:,None]
        if a.ndim!=2 or not a.size: raise ValueError(f'{n} must be a non-empty two-dimensional array.')
        if not np.all(np.isfinite(a)): raise ValueError(f'{n} contains NaN or infinite values.')
        return a
    @staticmethod
    def _numeric_y(v):
        raw=v.to_numpy() if isinstance(v,pd.Series) else v; a=np.asarray(raw,float)
        if a.ndim==2 and 1 in a.shape:a=a.reshape(-1)
        if a.ndim!=1 or not a.size: raise ValueError('y must be a non-empty one-dimensional vector.')
        if not np.all(np.isfinite(a)): raise ValueError('y contains NaN or infinite values.')
        return a
    def _coerce_X_fit(self,X):
        a=self._numeric_2d(X,'X'); names=tuple(str(c) for c in X.columns) if isinstance(X,pd.DataFrame) else tuple(f'x{i}' for i in range(a.shape[1])); self._legacy_intercept_input_=False
        if self.fit_intercept and a.shape[1] and np.allclose(a[:,0],1):
            self._legacy_intercept_input_=True; a=a[:,1:]; names=names[1:]
            if not a.shape[1]: raise ValueError('X must contain at least one non-intercept predictor.')
            warnings.warn('A leading all-ones column was detected and removed because fit_intercept=True.',UserWarning,stacklevel=3)
        return a,names
    def _coerce_X_predict(self,X):
        a=self._numeric_2d(X,'X'); names=tuple(str(c) for c in X.columns) if isinstance(X,pd.DataFrame) else None
        if self.fit_intercept and a.shape[1]==(self.n_features_in_ or 0)+1 and np.allclose(a[:,0],1): a=a[:,1:]; names=names[1:] if names else None
        if a.shape[1]!=self.n_features_in_: raise ValueError(f'X must contain {self.n_features_in_} predictors; got {a.shape[1]}.')
        if names is not None and tuple(names)!=self.feature_names_: raise ValueError('Prediction DataFrame columns must match training columns in the same order.')
        return a
    @staticmethod
    def _input_names(v,prefix,n): return tuple(str(c) for c in v.columns) if isinstance(v,pd.DataFrame) else tuple(f'{prefix}_{i}' for i in range(n))
    def _fit_geometry_scaler(self,coords,attrs):
        self.coord_center_=coords.mean(0); cc=coords-self.coord_center_; ss=cc.std(0); pos=ss[ss>np.finfo(float).eps]; self.coord_scale_=float(np.sqrt(np.mean(pos**2))) if pos.size else 1.0
        self.attr_center_=attrs.mean(0) if attrs.shape[1] else np.zeros(0); self.attr_scale_=attrs.std(0) if attrs.shape[1] else np.zeros(0); self.constant_attribute_mask_=self.attr_scale_<=np.finfo(float).eps if attrs.shape[1] else np.zeros(0,dtype=bool)
        if attrs.shape[1]: self.attr_scale_=self.attr_scale_.copy(); self.attr_scale_[self.constant_attribute_mask_]=1.0
        return self._transform_geometry(coords,attrs)
    def _transform_geometry(self,coords,attrs):
        if self.coord_center_ is None or self.attr_center_ is None: raise RuntimeError('Geometry scaler is not fitted.')
        if coords.shape[1]!=self.coord_center_.size or attrs.shape[1]!=self.attr_center_.size: raise ValueError('Prediction geometry dimension mismatch.')
        if not self.standardize_geometry:return coords.copy(),attrs.copy()
        return (coords-self.coord_center_)/self.coord_scale_, ((attrs-self.attr_center_)/self.attr_scale_ if attrs.shape[1] else attrs.copy())
    def _prepare_fit_inputs(self,X,y,coords,attributes):
        Xr,names=self._coerce_X_fit(X); ya=self._numeric_y(y); ca=validate_coords(coords); aa=np.zeros((ca.shape[0],0)) if attributes is None else self._numeric_2d(attributes,'attributes'); an=() if attributes is None else self._input_names(attributes,'attr',aa.shape[1])
        n=Xr.shape[0]
        if ya.size!=n or ca.shape[0]!=n or aa.shape[0]!=n: raise ValueError('X, y, coords and attributes must contain the same rows.')
        Xd=add_intercept(Xr) if self.fit_intercept else Xr.copy()
        if n<=Xd.shape[1]+1: raise ValueError('LG-GWR needs more observations than local design parameters plus one.')
        if self.scale_constraint=='orthogonal' and self.latent_dim>(ca.shape[1]+aa.shape[1]): raise ValueError('orthogonal scale_constraint requires latent_dim <= geometry input dimension.')
        cn=self._input_names(coords,'coord',ca.shape[1]); self.feature_names_in_=np.asarray(names,dtype=object); self.feature_names_=names; self.geometry_feature_names_=cn+an; self.n_features_in_=Xr.shape[1]; self.X_train_=Xr.copy(); self.X_design_=Xd.copy(); self.y_train_=ya.copy(); self.coords_train_=ca.copy(); self.attrs_train_=aa.copy(); self.coords_geometry_,self.attrs_geometry_=self._fit_geometry_scaler(ca,aa); self.u_train_=np.hstack([self.coords_geometry_,self.attrs_geometry_]); return Xd,ya,self.coords_geometry_,self.attrs_geometry_
    def _prepare_prediction_inputs(self,X,coords,attributes):
        self._require_fitted(); Xr=self._coerce_X_predict(X); Xd=add_intercept(Xr) if self.fit_intercept else Xr.copy(); ca=validate_coords(coords); aa=np.zeros((ca.shape[0],0)) if attributes is None else self._numeric_2d(attributes,'attributes')
        if Xr.shape[0]!=ca.shape[0] or aa.shape[0]!=Xr.shape[0]: raise ValueError('X, coords and attributes must contain the same rows.')
        cg,ag=self._transform_geometry(ca,aa); return Xd,ca,cg,ag
    def _require_fitted(self):
        if not self._is_fitted: raise ValueError('LGGWR is not fitted. Call fit() first.')
    def _kernel_weights(self,d,h):
        if np.isinf(h):return np.ones_like(d,float)
        v=d/h
        if self.kernel=='gaussian':return np.exp(-.5*v*v)
        if self.kernel=='bisquare':return np.where(v<1,(1-v*v)**2,0.0)
        return np.exp(-v)
    def _kernel_deriv_over_d(self,d,h):
        if np.isinf(h):return np.zeros_like(d,float)
        v=d/h
        if self.kernel=='gaussian':return -(1/h**2)*np.exp(-.5*v*v)
        if self.kernel=='bisquare':return np.where(v<1,-(4/h**2)*(1-v*v),0.0)
        out=np.zeros_like(d); m=d>1e-12; out[m]=-(1/h)*np.exp(-v[m])/d[m]; return out
    def _project_matrix(self,M,target):
        if not M.size:return M
        if self.scale_constraint=='orthogonal':u,_,vt=np.linalg.svd(M,full_matrices=False); return u@vt
        if self.scale_constraint=='frobenius':
            n=np.linalg.norm(M,'fro'); return M*(target/n) if n>1e-12 else M
        return M
    def _initialize_A(self,input_dim,rng,coord_dim=None,u=None,mode=None):
        mode=self.initialization if mode is None else mode; scale=np.sqrt(2/(input_dim+self.latent_dim))
        if mode=='coordinate' and coord_dim is not None:
            A=np.zeros((self.latent_dim,input_dim)); k=min(self.latent_dim,coord_dim); A[np.arange(k),np.arange(k)]=1
            if self.latent_dim>k:A[k:]=rng.standard_normal((self.latent_dim-k,input_dim))*scale
        elif mode=='pca' and u is not None:
            _,_,vt=np.linalg.svd(u-u.mean(0),full_matrices=False); k=min(self.latent_dim,vt.shape[0]); A=np.zeros((self.latent_dim,input_dim)); A[:k]=vt[:k]
            if k<self.latent_dim:A[k:]=rng.standard_normal((self.latent_dim-k,input_dim))*scale
        else:A=rng.standard_normal((self.latent_dim,input_dim))*scale
        return self._project_matrix(A,np.linalg.norm(A,'fro'))
    def _initialize_B(self,q,rng,attrs=None,mode=None):
        if q==0:return np.zeros((self.latent_dim,0))
        mode=self.initialization if mode is None else mode; scale=np.sqrt(2/(q+self.latent_dim))
        if mode=='pca' and attrs is not None:
            _,_,vt=np.linalg.svd(attrs-attrs.mean(0),full_matrices=False); k=min(self.latent_dim,vt.shape[0]); B=np.zeros((self.latent_dim,q)); B[:k]=vt[:k]
            if k<self.latent_dim:B[k:]=rng.standard_normal((self.latent_dim-k,q))*scale
        else:B=rng.standard_normal((self.latent_dim,q))*scale
        return self._project_matrix(B,np.linalg.norm(B,'fro'))
    @staticmethod
    def _auto_distance_bandwidth(D,n_parameters):
        n=D.shape[0]; s=np.sort(D,axis=1); k=min(max(n_parameters+2,int(np.sqrt(n))),n-1); v=float(np.median(s[:,k]));
        if not np.isfinite(v) or v<=1e-12:
            p=D[D>1e-12]; v=float(np.median(p)) if p.size else 1.0
        return max(v,1e-6)
    def _resolve_bandwidth(self,z,n_features):
        D=cdist(z,z)
        if self.bandwidth is not None and not isinstance(self.bandwidth,tuple):
            if not self.adaptive:return float(self.bandwidth)
            k=min(int(self.bandwidth),z.shape[0]-1); return max(float(np.median(np.sort(D,axis=1)[:,k])),1e-6)
        return self._auto_distance_bandwidth(D,n_features)
    @staticmethod
    def _solve_wls(M,b):
        p=M.shape[0]
        try:
            x=np.linalg.solve(M,b)
            if np.all(np.isfinite(x)):return x
        except np.linalg.LinAlgError:pass
        ridge=1e-6*(np.trace(M)/max(p,1)+1e-12)+1e-12
        try:
            x=np.linalg.solve(M+ridge*np.eye(p),b)
            if np.all(np.isfinite(x)):return x
        except np.linalg.LinAlgError:pass
        return np.linalg.pinv(M)@b
    @classmethod
    def _hat_solution(cls,X,y,w,xq):
        Xw=X*w[:,None]; M=Xw.T@X; C=cls._solve_wls(M,Xw.T); return C@y,xq@C
    def _forward_loo(self,X,y,z,h):
        n,p=X.shape; D=cdist(z,z); W=self._kernel_weights(D,h); np.fill_diagonal(W,0); beta=np.zeros((n,p)); g=np.zeros((n,p)); yh=np.zeros(n)
        for i in range(n): Xw=X*W[i,:,None]; M=Xw.T@X; beta[i]=self._solve_wls(M,Xw.T@y); g[i]=self._solve_wls(M,X[i]); yh[i]=X[i]@beta[i]
        return {'d':D,'W':W,'beta':beta,'g':g,'yhat':yh}
    def _compute_loss(self,y,yh): return float(np.mean((y-yh)**2)+self.lambda_reg*np.sum(self.A_**2))
    def _compute_gradient(self,X,y,u,z,h,c):
        n=X.shape[0]; res=y-c['yhat']; ratio=self._kernel_deriv_over_d(c['d'],h); np.fill_diagonal(ratio,0); grad=np.zeros_like(self.A_)
        for i in range(n):
            sens=X@c['g'][i]; le=y-X@c['beta'][i]; coef=res[i]*sens*le*ratio[i]; grad+=((z[i]-z)*coef[:,None]).T@(u[i]-u)
        return -(2/n)*grad+2*self.lambda_reg*self.A_
    def _forward_loo_sep(self,X,y,Kg,zeta,h_a):
        n,p=X.shape; da=cdist(zeta,zeta) if zeta.shape[1] else np.zeros((n,n)); Ka=self._kernel_weights(da,h_a) if zeta.shape[1] else np.ones((n,n)); W=Kg*Ka; np.fill_diagonal(W,0); beta=np.zeros((n,p)); g=np.zeros((n,p)); yh=np.zeros(n)
        for i in range(n): Xw=X*W[i,:,None]; M=Xw.T@X; beta[i]=self._solve_wls(M,Xw.T@y); g[i]=self._solve_wls(M,X[i]); yh[i]=X[i]@beta[i]
        return {'da':da,'Kg':Kg,'beta':beta,'g':g,'yhat':yh}
    def _compute_gradient_sep(self,X,y,a,z,h,c):
        n=X.shape[0]; res=y-c['yhat']; ratio=self._kernel_deriv_over_d(c['da'],h); np.fill_diagonal(ratio,0); grad=np.zeros_like(self.B_)
        for i in range(n):
            sens=X@c['g'][i]; le=y-X@c['beta'][i]; coef=res[i]*sens*le*c['Kg'][i]*ratio[i]; grad+=((z[i]-z)*coef[:,None]).T@(a[i]-a)
        return -(2/n)*grad+2*self.lambda_reg*self.B_
    def _local_fit_with_hat(self,X,y,z,h):
        W=self._kernel_weights(cdist(z,z),h); n,p=X.shape; b=np.zeros((n,p)); S=np.zeros((n,n))
        for i in range(n):b[i],S[i]=self._hat_solution(X,y,W[i],X[i])
        return b,S
    def _local_fit_with_hat_sep(self,X,y,Dg,z,hg,ha):
        W=self._kernel_weights(Dg,hg)*(self._kernel_weights(cdist(z,z),ha) if z.shape[1] else np.ones_like(Dg)); n,p=X.shape; b=np.zeros((n,p)); S=np.zeros((n,n))
        for i in range(n):b[i],S[i]=self._hat_solution(X,y,W[i],X[i])
        return b,S
    def _optimise_joint(self,X,y,u,initial,h):
        self.A_=initial.copy(); target=np.linalg.norm(initial,'fro'); best=np.inf; bm=initial.copy(); hist=[]; m=np.zeros_like(initial);v=np.zeros_like(initial);prev=np.inf;stall=0;conv=False;reason='max_iter'
        for it in range(self.max_iter):
            z=u@self.A_.T;c=self._forward_loo(X,y,z,h);loss=self._compute_loss(y,c['yhat'])
            if not np.isfinite(loss):reason='nonfinite_loss';break
            hist.append(float(loss))
            if loss<best-self.tol:best=loss;bm=self.A_.copy();stall=0
            else:stall+=1
            if stall>=self.patience:reason='patience';conv=True;break
            if it>0 and abs(prev-loss)<self.tol:reason='tolerance';conv=True;break
            prev=loss;g=self._compute_gradient(X,y,u,z,h,c)
            if not np.all(np.isfinite(g)):reason='nonfinite_gradient';break
            gn=np.linalg.norm(g); g*=self.grad_clip/gn if gn>self.grad_clip else 1; t=it+1;m=.9*m+.1*g;v=.999*v+.001*g*g; self.A_-=self.learning_rate*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8); self.A_=self._project_matrix(self.A_,target)
        self.A_=bm; final=self._compute_loss(y,self._forward_loo(X,y,u@self.A_.T,h)['yhat'])
        if not hist:hist=[final];best=final
        return _OptimisationResult(self.A_.copy(),tuple(hist),float(best),float(final),len(hist),conv,reason)
    def _optimise_separable(self,X,y,a,initial,Kg,ha):
        self.B_=initial.copy()
        if not initial.size:
            loss=float(np.mean((y-self._forward_loo_sep(X,y,Kg,np.zeros((len(y),0)),ha)['yhat'])**2)); return _OptimisationResult(initial.copy(),(loss,),loss,loss,0,True,'no_attributes')
        target=np.linalg.norm(initial,'fro');best=np.inf;bm=initial.copy();hist=[];m=np.zeros_like(initial);v=np.zeros_like(initial);prev=np.inf;stall=0;conv=False;reason='max_iter'
        for it in range(self.max_iter):
            z=a@self.B_.T;c=self._forward_loo_sep(X,y,Kg,z,ha);loss=float(np.mean((y-c['yhat'])**2)+self.lambda_reg*np.sum(self.B_**2))
            if not np.isfinite(loss):reason='nonfinite_loss';break
            hist.append(loss)
            if loss<best-self.tol:best=loss;bm=self.B_.copy();stall=0
            else:stall+=1
            if stall>=self.patience:reason='patience';conv=True;break
            if it>0 and abs(prev-loss)<self.tol:reason='tolerance';conv=True;break
            prev=loss;g=self._compute_gradient_sep(X,y,a,z,ha,c);gn=np.linalg.norm(g);g*=self.grad_clip/gn if gn>self.grad_clip else 1;t=it+1;m=.9*m+.1*g;v=.999*v+.001*g*g;self.B_-=self.learning_rate*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8);self.B_=self._project_matrix(self.B_,target)
        self.B_=bm; final=float(np.mean((y-self._forward_loo_sep(X,y,Kg,a@self.B_.T,ha)['yhat'])**2)+self.lambda_reg*np.sum(self.B_**2))
        if not hist:hist=[final];best=final
        return _OptimisationResult(self.B_.copy(),tuple(hist),float(best),float(final),len(hist),conv,reason)
    def _select_bandwidth_aicc(self,X,y,z,n_grid=12):
        D=cdist(z,z);n,p=X.shape;s=np.sort(D,axis=1);lo=max(float(np.median(s[:,min(p+2,n-1)])),1e-6);hi=max(float(D.max()),lo*4);cand=list(np.geomspace(lo,hi,n_grid));
        if isinstance(self.bandwidth_,Real):cand.append(float(self.bandwidth_))
        best=(np.inf,cand[0])
        for h in cand:
            b,S=self._local_fit_with_hat(X,y,z,float(h));f=np.einsum('ij,ij->i',X,b);a=compute_diagnostics(y,f,S,True)['aicc'];
            if np.isfinite(a) and a<best[0]:best=(a,float(h))
        return best[1]
    def _select_bandwidths_aicc(self,X,y,Dg,z,current=None,n_grid=6):
        n,p=X.shape
        def bounds(D): s=np.sort(D,axis=1);lo=max(float(np.median(s[:,min(p+2,n-1)])),1e-6);return lo,max(float(D.max()),lo*4)
        gl,gu=bounds(Dg);gg=list(np.geomspace(gl,gu,n_grid));ag=[np.inf]
        if z.shape[1]:al,au=bounds(cdist(z,z));ag=list(np.geomspace(al,au,n_grid))+[np.inf]
        if current:gg.append(float(current[0]));ag.append(float(current[1]))
        def score(hg,ha):b,S=self._local_fit_with_hat_sep(X,y,Dg,z,hg,ha);f=np.einsum('ij,ij->i',X,b);return compute_diagnostics(y,f,S,True)['aicc']
        hg=gg[len(gg)//2] if current is None else float(current[0]);ha=np.inf if current is None else float(current[1])
        for _ in range(2):hg=min(gg,key=lambda x:score(float(x),ha));ha=min(ag,key=lambda x:score(float(hg),float(x)))
        return float(hg),float(ha)
    def fit(self,X,y,coords,attributes=None):
        self._reset_fit_state()
        try:
            Xd,ya,cg,ag=self._prepare_fit_inputs(X,y,coords,attributes); self._fit_separable(Xd,ya,cg,ag) if self.geometry=='separable' else self._fit_joint(Xd,ya,cg,ag); self._finalise_public_parameters(); self._is_fitted=True; return self
        except Exception:self._reset_fit_state();raise
    def _fit_joint(self,X,y,coords,attrs):
        u=np.hstack([coords,attrs]); rec=[]; seed=0 if self.random_state is None else int(self.random_state)
        for r in range(self.n_restarts):
            rng=np.random.default_rng(seed+r); mode=self.initialization if r==0 else 'random'; self.A_=self._initialize_A(u.shape[1],rng,coords.shape[1],u,mode);h=self._resolve_bandwidth(u@self.A_.T,X.shape[1]);bh=[float(h)];hist=[];stages=self.bandwidth_updates+1 if self.select_bandwidth else 1;sr=None
            for stage in range(stages):
                sr=self._optimise_joint(X,y,u,self.A_,h);self.A_=sr.matrix.copy();hist.extend(sr.loss_history)
                if self.select_bandwidth:self.bandwidth_=h;h=self._select_bandwidth_aicc(X,y,u@self.A_.T);bh.append(float(h))
            z=u@self.A_.T;b,S=self._local_fit_with_hat(X,y,z,h);f=np.einsum('ij,ij->i',X,b);d=compute_diagnostics(y,f,S,True);loo=self._compute_loss(y,self._forward_loo(X,y,z,h)['yhat']);rec.append(dict(matrix=self.A_.copy(),bandwidth=h,bh=bh,hist=hist,best=min(hist) if hist else loo,loo=loo,n=len(hist),conv=sr.converged,reason=sr.stop_reason,b=b,S=S,f=f,d=d))
        best=min(rec,key=lambda x:(x['d']['aicc'],x['loo']));self.restart_scores_=[{'restart':float(i),'aicc':float(x['d']['aicc']),'final_loo_loss':float(x['loo'])} for i,x in enumerate(rec)];self.A_=best['matrix'];self.B_=None;self.bandwidth_=float(best['bandwidth']);self.bandwidth_history_=list(best['bh']);self.loss_history_=list(best['hist']);self.best_loss_=float(best['best']);self.final_loo_loss_=float(best['loo']);self.n_iter_=best['n'];self.converged_=best['conv'];self.stop_reason_=best['reason'];self.latent_coords_=u@self.A_.T;self.coefficients_=best['b'];self.hat_matrix_=best['S'];self.fitted_values_=best['f'];self.residuals_=y-best['f'];self.diagnostics_=best['d'];self._set_metric_outputs(self.A_)
    def _fit_separable(self,X,y,coords,attrs):
        Dg=cdist(coords,coords);hg,ha=(float(self.bandwidth[0]),float(self.bandwidth[1])) if isinstance(self.bandwidth,tuple) else (float(self.bandwidth) if self.bandwidth is not None else self._auto_distance_bandwidth(Dg,X.shape[1]),np.inf);rec=[];seed=0 if self.random_state is None else int(self.random_state)
        for r in range(self.n_restarts):
            rng=np.random.default_rng(seed+r);mode=self.initialization if r==0 else 'random';self.B_=self._initialize_B(attrs.shape[1],rng,attrs,mode);hgr,har=hg,ha
            if attrs.shape[1] and np.isinf(har):har=self._auto_distance_bandwidth(cdist(attrs@self.B_.T,attrs@self.B_.T),X.shape[1])
            bh=[(float(hgr),float(har))];hist=[];stages=self.bandwidth_updates+1 if self.select_bandwidth else 1;sr=None
            for stage in range(stages):
                sr=self._optimise_separable(X,y,attrs,self.B_,self._kernel_weights(Dg,hgr),har);self.B_=sr.matrix.copy();hist.extend(sr.loss_history);z=attrs@self.B_.T if attrs.shape[1] else np.zeros((len(y),0))
                if self.select_bandwidth:hgr,har=self._select_bandwidths_aicc(X,y,Dg,z,(hgr,har));bh.append((float(hgr),float(har)))
            z=attrs@self.B_.T if attrs.shape[1] else np.zeros((len(y),0));b,S=self._local_fit_with_hat_sep(X,y,Dg,z,hgr,har);f=np.einsum('ij,ij->i',X,b);d=compute_diagnostics(y,f,S,True);loo=float(np.mean((y-self._forward_loo_sep(X,y,self._kernel_weights(Dg,hgr),z,har)['yhat'])**2)+self.lambda_reg*np.sum(self.B_**2));rec.append(dict(matrix=self.B_.copy(),bandwidth=(hgr,har),bh=bh,hist=hist,best=min(hist) if hist else loo,loo=loo,n=len(hist),conv=sr.converged,reason=sr.stop_reason,b=b,S=S,f=f,d=d,z=z))
        best=min(rec,key=lambda x:(x['d']['aicc'],x['loo']));self.restart_scores_=[{'restart':float(i),'aicc':float(x['d']['aicc']),'final_loo_loss':float(x['loo'])} for i,x in enumerate(rec)];self.A_=None;self.B_=best['matrix'];self.bandwidth_=best['bandwidth'];self.bandwidth_history_=list(best['bh']);self.loss_history_=list(best['hist']);self.best_loss_=float(best['best']);self.final_loo_loss_=float(best['loo']);self.n_iter_=best['n'];self.converged_=best['conv'];self.stop_reason_=best['reason'];self.latent_coords_=best['z'];self.coefficients_=best['b'];self.hat_matrix_=best['S'];self.fitted_values_=best['f'];self.residuals_=y-best['f'];self.diagnostics_=best['d'];self._set_metric_outputs(self.B_)
    def _set_metric_outputs(self,M):self.metric_matrix_=M.T@M;d=np.clip(np.diag(self.metric_matrix_),0,np.inf);s=float(d.sum());self.metric_contributions_=d/s if s>0 else np.zeros_like(d)
    def _finalise_public_parameters(self):
        self.intercept_=self.coefficients_[:,0].copy() if self.fit_intercept else np.zeros(self.coefficients_.shape[0]);self.coef_=self.coefficients_[:,1:].copy() if self.fit_intercept else self.coefficients_.copy()
    def _local_fit(self,X,y,zt,zq,h,Xq):
        W=self._kernel_weights(cdist(zq,zt),h);gb=np.linalg.lstsq(X,y,rcond=None)[0];out=np.zeros((len(zq),X.shape[1]))
        for i in range(len(zq)):
            if np.sum(W[i]>1e-8)<X.shape[1]:out[i]=gb
            else:out[i],_=self._hat_solution(X,y,W[i],Xq[i])
        return out
    def _local_fit_sep(self,X,y,ct,zt,cq,zq,hg,ha,Xq):
        W=self._kernel_weights(cdist(cq,ct),hg)*(self._kernel_weights(cdist(zq,zt),ha) if zt.shape[1] else np.ones((len(cq),len(ct))));gb=np.linalg.lstsq(X,y,rcond=None)[0];out=np.zeros((len(cq),X.shape[1]))
        for i in range(len(cq)):
            if np.sum(W[i]>1e-8)<X.shape[1]:out[i]=gb
            else:out[i],_=self._hat_solution(X,y,W[i],Xq[i])
        return out
    def predict_result(self,X,coords,attributes=None):
        Xd,cr,cg,ag=self._prepare_prediction_inputs(X,coords,attributes)
        if self.geometry=='separable':
            zt=self.attrs_geometry_@self.B_.T if self.attrs_geometry_.shape[1] else np.zeros((len(self.y_train_),0));zq=ag@self.B_.T if ag.shape[1] else np.zeros((len(Xd),0));b=self._local_fit_sep(self.X_design_,self.y_train_,self.coords_geometry_,zt,cg,zq,float(self.bandwidth_[0]),float(self.bandwidth_[1]),Xd);latent=zq
        else:
            uq=np.hstack([cg,ag]);latent=uq@self.A_.T;b=self._local_fit(self.X_design_,self.y_train_,self.latent_coords_,latent,float(self.bandwidth_),Xd)
        p=np.einsum('ij,ij->i',Xd,b);inter=b[:,0] if self.fit_intercept else np.zeros(len(p));coef=b[:,1:] if self.fit_intercept else b;return LGGWRPredictionResult(p,coef,inter,cr.copy(),latent.copy(),self.feature_names_)
    def predict(self,X,coords,attributes=None):return self.predict_result(X,coords,attributes).predictions
    def results_frame(self):
        self._require_fitted();d={'coord_0':self.coords_train_[:,0],'coord_1':self.coords_train_[:,1],'fitted':self.fitted_values_,'residual':self.residuals_,'intercept':self.intercept_}
        for i in range(self.latent_coords_.shape[1]):d[f'latent_{i}']=self.latent_coords_[:,i]
        for i,n in enumerate(self.feature_names_):d[f'coef_{n}']=self.coef_[:,i]
        return pd.DataFrame(d)
    to_frame=results_frame
    def metric_frame(self):
        self._require_fitted();names=self.geometry_feature_names_ if self.geometry=='joint' else self.geometry_feature_names_[self.coord_center_.size:];return pd.DataFrame({'geometry_feature':list(names),'metric_diagonal':np.diag(self.metric_matrix_),'metric_contribution':self.metric_contributions_})
    def summary(self):
        self._require_fitted();M=self.A_ if self.geometry=='joint' else self.B_;return format_summary('LG-GWR Summary',{'geometry':self.geometry,'n_samples':len(self.y_train_),'latent_dim':self.latent_dim,'bandwidth':self.bandwidth_,'matrix_norm':float(np.linalg.norm(M,'fro')),'r2':self.diagnostics_['r2'],'aicc':self.diagnostics_['aicc']})

__all__=['LGGWR','LGGWRPredictionResult']
