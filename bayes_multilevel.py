# -*- coding: utf-8 -*-
"""Bayesian analysis engine (sarcoid + IPF parity). Two-level design:
exposure = tau^2-marginalized Bayesian RE (4-prior sweep); overall = Bayesian multilevel
(exposure + study random intercepts). Frequentist DL/REML + CHE(CR2+Satterthwaite) as sensitivity.
Effect scale log-OR; SE=(lnUCL-lnLCL)/(2z), z=1.959964. numpy only."""
import math, numpy as np
Z=1.959964
_T={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
    11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,20:2.086,30:2.042,60:2.000,120:1.980}
def tcrit(df):
    df=max(df,1e-6); ks=sorted(_T)
    if df>=ks[-1]: return 1.96
    for a,b in zip(ks,ks[1:]):
        if a<=df<=b: return _T[a]+(_T[b]-_T[a])*(df-a)/(b-a)
    return _T[ks[0]]
def lv(e,l,u):
    y=math.log(e); s=(math.log(u)-math.log(l))/(2*Z); return y,s*s

def dl(y,v):
    y=np.asarray(y); v=np.asarray(v); k=len(y); w=1/v
    Q=(w*(y-(w*y).sum()/w.sum())**2).sum(); C=w.sum()-(w**2).sum()/w.sum(); t2=max(0.,(Q-(k-1))/C) if C>0 else 0.
    s2=(k-1)*w.sum()/(w.sum()**2-(w**2).sum()); I2=100*t2/(t2+s2) if t2+s2>0 else 0.
    w2=1/(v+t2); mu=(w2*y).sum()/w2.sum(); se=math.sqrt(1/w2.sum())
    return dict(OR=math.exp(mu),lcl=math.exp(mu-Z*se),ucl=math.exp(mu+Z*se),I2=I2)
def reml(y,v):
    y=np.asarray(y); v=np.asarray(v); best=(-1e18,0.)
    for t in np.linspace(0,3,30001):
        t2=t*t; w=1/(v+t2); mu=(w*y).sum()/w.sum()
        ll=-0.5*(np.sum(np.log(v+t2))+math.log(w.sum())+np.sum(w*(y-mu)**2))
        if ll>best[0]: best=(ll,t2)
    t2=best[1]; w=1/(v+t2); mu=(w*y).sum()/w.sum(); se=math.sqrt(1/w.sum())
    return dict(OR=math.exp(mu),lcl=math.exp(mu-Z*se),ucl=math.exp(mu+Z*se),tau2=t2)

def bayes_pool(y,v,prior="HN",scale=1.0,ntau=4000,nmu=4000,taumax=6.0):
    y=np.asarray(y,float); v=np.asarray(v,float)
    lpri=(lambda t:-0.5*(t/scale)**2) if prior=="HN" else (lambda t:-math.log(1+(t/scale)**2))
    taus=np.linspace(1e-5,taumax,ntau); lp=np.empty(ntau); mh=np.empty(ntau); Vm=np.empty(ntau)
    for j,t in enumerate(taus):
        w=1/(v+t*t); mu=(w*y).sum()/w.sum(); Vv=1/w.sum()
        lp[j]=(-0.5*np.sum(np.log(2*math.pi*(v+t*t)))-0.5*np.sum((y-mu)**2*w)+0.5*math.log(2*math.pi*Vv)+lpri(t))
        mh[j]=mu; Vm[j]=Vv
    lp-=lp.max(); pw=np.exp(lp); pw/=pw.sum()
    sp=8*math.sqrt(Vm.max()+taumax**2); mg=np.linspace(mh.min()-sp,mh.max()+sp,nmu); dens=np.zeros_like(mg)
    for j in range(ntau): dens+=pw[j]*np.exp(-0.5*(mg-mh[j])**2/Vm[j])/math.sqrt(2*math.pi*Vm[j])
    dens/=np.trapz(dens,mg); cdf=np.concatenate([[0],np.cumsum((dens[1:]+dens[:-1])/2*np.diff(mg))]); cdf/=cdf[-1]
    q=lambda p: float(np.interp(p,cdf,mg))
    return dict(OR=math.exp(q(0.5)),lcl=math.exp(q(0.025)),ucl=math.exp(q(0.975)),P=float(1-np.interp(0.,mg,cdf)))
def bayes_sweep(y,v):
    res=[bayes_pool(y,v,p,s) for p,s in [("HN",0.5),("HN",1.0),("HN",2.0),("HC",1.0)]]
    return bayes_pool(y,v),(min(r["P"] for r in res),max(r["P"] for r in res))

def bayes_multilevel(y,v,expo,study,n_iter=40000,warm=12000,seeds=(3,13,23,33)):
    y=np.asarray(y,float); v=np.asarray(v,float); n=len(y)
    exps=sorted(set(expo)); J=len(exps); ei=np.array([exps.index(e) for e in expo])
    studs=sorted(set(study)); M=len(studs); si=np.array([studs.index(s) for s in study])
    p=1+J+M; X=np.zeros((n,p)); X[:,0]=1
    for i in range(n): X[i,1+ei[i]]=1; X[i,1+J+si[i]]=1
    iA,iB=1,1+J
    def hn(t): return -0.5*t*t if t>0 else -np.inf
    def loglik(coef,t2): s2=v+t2; eta=X@coef; return -0.5*np.sum(np.log(2*np.pi*s2)+(y-eta)**2/s2)
    chains=[]
    for sd in seeds:
        rng=np.random.default_rng(sd)
        coef=np.concatenate([[rng.normal(0,.3)],rng.normal(0,.2,p-1)])
        tau,se,ss=.2,.2,.2; st=dict(t=.25,e=.4,s=.4); keep=[]
        for it in range(n_iter):
            t2=tau*tau; W=1/(v+t2)
            P0=np.zeros(p); P0[0]=1/100; P0[iA:iB]=1/se**2; P0[iB:]=1/ss**2
            Lam=X.T@(W[:,None]*X)+np.diag(P0); L=np.linalg.cholesky(Lam)
            mnt=np.linalg.solve(Lam,X.T@(W*y)); coef=mnt+np.linalg.solve(L.T,rng.normal(size=p))
            lt=math.log(tau); ltp=lt+rng.normal(0,st['t']); tp=math.exp(ltp)
            if math.log(rng.random())<(loglik(coef,tp**2)+hn(tp)+ltp)-(loglik(coef,t2)+hn(tau)+lt): tau=tp
            def mh(sd0,blk,key):
                l=math.log(sd0); lp2=l+rng.normal(0,st[key]); spx=math.exp(lp2)
                new=-0.5*np.sum(np.log(2*np.pi*spx**2)+blk**2/spx**2)+hn(spx)+lp2
                old=-0.5*np.sum(np.log(2*np.pi*sd0**2)+blk**2/sd0**2)+hn(sd0)+l
                return spx if math.log(rng.random())<(new-old) else sd0
            se=mh(se,coef[iA:iB],'e'); ss=mh(ss,coef[iB:],'s')
            if it>=warm: keep.append(np.concatenate([[coef[0]],coef[iA:iB],[tau,se,ss]]))
        chains.append(np.array(keep))
    S=np.stack(chains); flat=S.reshape(-1,S.shape[2])
    def rhat(a):
        m,nn=a.shape; hh=nn//2; sub=np.concatenate([a[:,:hh],a[:,hh:2*hh]],0)
        B=hh*sub.mean(1).var(ddof=1); Wv=sub.var(1,ddof=1).mean()
        return math.sqrt(((hh-1)/hh*Wv+B/hh)/Wv) if Wv>0 else float('nan')
    mu=flat[:,0]
    return dict(overall=(math.exp(mu.mean()),math.exp(np.percentile(mu,2.5)),math.exp(np.percentile(mu,97.5)),float(np.mean(mu>0))),
                rhat_mu=rhat(S[:,:,0]),exps=exps)

def _msqrt(M,inv=False):
    w,V=np.linalg.eigh((M+M.T)/2); w=np.clip(w,1e-12,None); d=1/np.sqrt(w) if inv else np.sqrt(w); return (V*d)@V.T
def che_overall(y,v,study,r=0.6):
    y=np.asarray(y,float); v=np.asarray(v,float); n=len(y); X=np.ones((n,1))
    same=np.array([[1.0 if study[a]==study[b] else 0.0 for b in range(n)] for a in range(n)])
    S=np.array([[(v[a] if a==b else (r*math.sqrt(v[a]*v[b]) if study[a]==study[b] else 0.0)) for b in range(n)] for a in range(n)])
    def negRLL(t2,o2):
        V=S+t2*same+o2*np.eye(n)
        try: Vi=np.linalg.inv(V)
        except: return 1e18
        XtViX=X.T@Vi@X
        try: P=Vi-Vi@X@np.linalg.inv(XtViX)@X.T@Vi
        except: return 1e18
        _,ld=np.linalg.slogdet(V); _,ld2=np.linalg.slogdet(XtViX); return 0.5*(ld+ld2+y@P@y)
    grid=np.concatenate([[0.0],np.geomspace(1e-4,4,40)]); best=(1e18,0,0)
    for t2 in grid:
        for o2 in grid:
            rr=negRLL(t2,o2)
            if rr<best[0]: best=(rr,t2,o2)
    _,t2,o2=best; V=S+t2*same+o2*np.eye(n); Vi=np.linalg.inv(V)
    M=np.linalg.inv(X.T@Vi@X); beta=M@X.T@Vi@y; e=y-X@beta
    meat=np.zeros_like(M); g=[]
    for s in list(dict.fromkeys(study)):
        idx=[i for i in range(n) if study[i]==s]; Xi=X[idx]; Wi=Vi[np.ix_(idx,idx)]; ei=e[idx]; Psi=V[np.ix_(idx,idx)]
        Ai=_msqrt(Psi)@_msqrt(Psi-Xi@M@Xi.T,inv=True); adj=Xi.T@Wi@Ai@ei.reshape(-1,1); meat+=adj@adj.T; g.append((Xi,Wi,Ai,Psi))
    VR=M@meat@M
    bv=np.array([float((M@Xi.T@Wi@Ai)@Psi@(M@Xi.T@Wi@Ai).T) for Xi,Wi,Ai,Psi in g]); df=(bv.sum()**2)/(bv**2).sum()
    se=math.sqrt(VR[0,0]); tc=tcrit(df); b=beta[0]
    return dict(OR=math.exp(b),lcl=math.exp(b-tc*se),ucl=math.exp(b+tc*se),df=df)
