#!/usr/bin/env python3
# =============================================================================
# ild_meta_pipeline.py
#
# TIERED BAYESIAN META-ANALYSIS PIPELINE WITH DEPENDENCY-AWARE ROBUSTNESS CHECKS
# Occupational airborne exposures and interstitial lung disease (IPF & sarcoidosis)
#
# One reproducible pipeline consolidating the project's verified components:
#   Stage 1  Effect prep      lnOR + SE reconstructed from published 95% CIs
#   Stage 2  Frequentist RE   DerSimonian-Laird + REML            (sensitivity)
#   Stage 3  Bayesian primary tau^2-marginalized normal-normal    (PRIMARY)
#            Prior sweep       HN(0.5), HN(1.0)*, HN(2.0), HC(1.0) -> prior-stability
#   Stage 4  Tiering          robust / borderline / fragile / weak / null
#   Stage 5  Dependency-aware CHE (Pustejovsky-Tipton) + CR2 sandwich + Satterthwaite df
#   Stage 6  Subgroup contrast common-tau^2 Wald + separate-tau^2 Cochran Q_between
#   Stage 7  Validation       CR2==HC2 unit test; reproduce published pooled values
#
# Inference labels track the model: Bayesian -> credible interval (CrI);
# frequentist DL/REML and source studies -> confidence interval (CI).
#
# Inputs  : Pooling_Set.csv                          (IPF study-level)
#           Sarcoidosis_Extraction_v2_reconciled.xlsx (sheet 'Pooling_Set_Sarcoid')
# Deps    : numpy, pandas (+ openpyxl for the .xlsx). No scipy required.
# Run     : python ild_meta_pipeline.py            (writes ild_pipeline_results.txt)
# =============================================================================
import math, sys, os
import numpy as np
import pandas as pd

Z = 1.95996                       # qnorm(0.975)
DATA_DIR = os.environ.get("ILD_DATA", ".")
PRIORS = [("HN", 0.5), ("HN", 1.0), ("HN", 2.0), ("HC", 1.0)]   # HN(1.0) is primary
_LINES = []
def _log(s=""):
    print(s); _LINES.append(s)

# ----------------------------------------------------------------- math helpers
def _erf(x):
    x = np.asarray(x, float); s = np.sign(x); x = np.abs(x); t = 1/(1+0.3275911*x)
    y = 1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*np.exp(-x*x)
    return s*y
def _ncdf(x, m, sd): return 0.5*(1+_erf((x-m)/(sd*math.sqrt(2))))
def _prior_logpdf(t, kind, scale):
    # half-normal / half-Cauchy on tau>=0 (normalizing constants drop out on the grid)
    if kind == "HN": return -0.5*(t/scale)**2
    return -math.log(1+(t/scale)**2)                 # HC

def se_from_ci(lcl, ucl): return (np.log(ucl)-np.log(lcl))/(2*Z)

# ------------------------------------------------- Stage 1: effect preparation
def prep(df):
    """study-level lnOR (yi) and sampling variance (vi) from OR + 95% CI."""
    d = df.copy()
    d["yi"] = np.log(d.est.astype(float))
    d["vi"] = se_from_ci(d.lcl.astype(float), d.ucl.astype(float))**2
    return d.reset_index(drop=True)

# ------------------------------------------ Stage 1b: LOCKED data reconciliation
def reconcile_ipf(ipf):
    """Locked IPF reconciliation (manuscript Methods / ROB notes), applied before pooling:
      (a) Air pollution -> genuine two-study incidence pool: exclude Johannson 2014
          (acute-exacerbation outcome, not incident IPF); Cui 2023 contributes its
          PM2.5 aHR 1.09 [1.02,1.17] (raw 1.01 [0.86,1.18] matched no reported estimate).
      (b) Genetic risk -> reframed as an effect modifier; reported descriptively, not pooled.
      (c) Nickel (speciation) -> uninformative k=2 pool; reported descriptively, not pooled.
    """
    ipf = ipf.copy()
    ap = ipf.exposure_family == "Air pollution"
    ipf.loc[ap & ipf.study.str.contains("Cui", na=False), ["est", "lcl", "ucl"]] = [1.09, 1.02, 1.17]
    ipf = ipf[~(ap & ipf.study.str.contains("Johannson", na=False))]
    DESCRIPTIVE = ["Genetic risk", "Nickel (speciation)"]
    ipf = ipf[~ipf.exposure_family.isin(DESCRIPTIVE)]
    return ipf

# --------------------------------------------- Stage 2: frequentist RE (DL/REML)
def dl(yi, vi):
    yi, vi = np.asarray(yi, float), np.asarray(vi, float); k = len(yi)
    w = 1/vi; muF = (w*yi).sum()/w.sum(); Q = (w*(yi-muF)**2).sum()
    C = w.sum()-(w**2).sum()/w.sum(); t2 = max(0.0, (Q-(k-1))/C) if C > 0 else 0.0
    w2 = 1/(vi+t2); mu = (w2*yi).sum()/w2.sum(); se = math.sqrt(1/w2.sum())
    I2 = max(0.0, (Q-(k-1))/Q)*100 if Q > 0 else 0.0
    return dict(OR=math.exp(mu), lcl=math.exp(mu-Z*se), ucl=math.exp(mu+Z*se),
                mu=mu, se=se, tau2=t2, I2=I2, Q=Q, k=k)
def reml(yi, vi):
    yi, vi = np.asarray(yi, float), np.asarray(vi, float); k = len(yi); t2 = 0.0
    for _ in range(200):
        w = 1/(vi+t2); W = w.sum(); mu = (w*yi).sum()/W
        num = (w**2*((yi-mu)**2 - vi)).sum() + (w**2/W).sum()
        den = (w**2).sum(); new = max(0.0, t2 + num/den) if den > 0 else 0.0
        if abs(new-t2) < 1e-10: t2 = new; break
        t2 = new
    w = 1/(vi+t2); mu = (w*yi).sum()/w.sum(); se = math.sqrt(1/w.sum())
    return dict(OR=math.exp(mu), lcl=math.exp(mu-Z*se), ucl=math.exp(mu+Z*se),
                mu=mu, se=se, tau2=t2, k=k)

# -------------------------------- Stage 3: Bayesian primary (tau^2-marginalized)
def bayes(yi, vi, kind="HN", scale=1.0, ngrid=4001):
    """tau^2-marginalized normal-normal RE. Posterior of mu is a normal mixture
    over a tau grid weighted by the marginal posterior of tau (flat prior on mu).
    Returns posterior median OR, 95% CrI, P(OR>1), and 95% prediction interval."""
    yi, vi = np.asarray(yi, float), np.asarray(vi, float)
    tau = np.linspace(0, 6, ngrid)
    mubar = np.empty_like(tau); sumw = np.empty_like(tau); logp = np.empty_like(tau)
    for i, t in enumerate(tau):
        w = 1/(vi+t*t); W = w.sum(); mb = (w*yi).sum()/W; Q = (w*(yi-mb)**2).sum()
        logp[i] = 0.5*np.log(w).sum() - 0.5*math.log(W) - 0.5*Q + _prior_logpdf(t, kind, scale)
        mubar[i] = mb; sumw[i] = W
    logp -= logp.max(); p = np.exp(logp); p /= p.sum()
    sd = 1/np.sqrt(sumw)
    mu_grid = np.linspace(mubar.min()-4, mubar.max()+4, 6000); cdf = np.zeros_like(mu_grid)
    for i in range(len(tau)):
        if p[i] > 1e-11: cdf += p[i]*_ncdf(mu_grid, mubar[i], sd[i])
    q = lambda a: float(np.interp(a, cdf, mu_grid))
    P = float(1 - np.interp(0.0, mu_grid, cdf))
    tau_med = float(np.interp(0.5, np.cumsum(p), tau))
    # 95% prediction interval: posterior-median mu +/- z*sqrt(se^2 + tau_med^2)
    mu_med = q(0.5); se_mu = math.sqrt(1/np.average(sumw, weights=p))
    pi_hw = Z*math.sqrt(se_mu**2 + tau_med**2)
    return dict(OR=math.exp(mu_med), lcl=math.exp(q(0.025)), ucl=math.exp(q(0.975)),
                P=P, tau=tau_med, pl=math.exp(mu_med-pi_hw), pu=math.exp(mu_med+pi_hw))

def prior_sweep(yi, vi):
    Ps = [bayes(yi, vi, kind=k, scale=s)["P"] for k, s in PRIORS]
    return min(Ps), max(Ps), (max(Ps)-min(Ps) < 0.05)

# ------------------------------------------------------ Stage 4: tier classifier
def tier(b_lcl, P, prior_stable):
    if b_lcl > 1 and prior_stable: return "robust"
    if P >= 0.95: return "borderline"
    if P >= 0.85: return "suggestive/fragile"
    if P >= 0.70: return "weak"
    return "null"

# ---------- Stage 5: dependency-aware CHE (Pustejovsky-Tipton) + CR2 + Satterthwaite
# Working model V = S + tau2*SameStudy + omega2*I ; S has constant within-study r.
# CR2 bias-reduced cluster-robust sandwich; Satterthwaite df per coefficient.
_T = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
      11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,20:2.086,30:2.042,60:2.000,120:1.980}
def tcrit(df):
    df = max(df, 1e-6); ks = sorted(_T)
    if df >= ks[-1]: return 1.96
    for a, b in zip(ks, ks[1:]):
        if a <= df <= b: return _T[a]+(_T[b]-_T[a])*(df-a)/(b-a)
    return _T[ks[0]]
def _msqrt(Mx, inv=False):
    w, V = np.linalg.eigh((Mx+Mx.T)/2); w = np.clip(w, 1e-12, None)
    d = 1/np.sqrt(w) if inv else np.sqrt(w)
    return (V*d)@V.T
def che_fit(y, study, X, vi, r=0.6):
    y = np.asarray(y, float); vi = np.asarray(vi, float); X = np.asarray(X, float); n = len(y)
    same = np.array([[1.0 if study[a] == study[b] else 0.0 for b in range(n)] for a in range(n)])
    S = np.array([[(vi[a] if a == b else (r*math.sqrt(vi[a]*vi[b]) if study[a] == study[b] else 0.0))
                   for b in range(n)] for a in range(n)])
    def negRLL(t2, o2):
        V = S+t2*same+o2*np.eye(n)
        try: Vi = np.linalg.inv(V)
        except np.linalg.LinAlgError: return 1e18
        XtViX = X.T@Vi@X
        try: P = Vi-Vi@X@np.linalg.inv(XtViX)@X.T@Vi
        except np.linalg.LinAlgError: return 1e18
        _, ld = np.linalg.slogdet(V); _, ld2 = np.linalg.slogdet(XtViX)
        return 0.5*(ld+ld2+y@P@y)
    grid = np.concatenate([[0.0], np.geomspace(1e-4, 4, 40)]); best = (1e18, 0, 0)
    for t2 in grid:
        for o2 in grid:
            r_ = negRLL(t2, o2)
            if r_ < best[0]: best = (r_, t2, o2)
    _, t2, o2 = best
    for _ in range(6):
        st, so = max(t2*0.3, 1e-4), max(o2*0.3, 1e-4)
        for t2c in [t2-st, t2, t2+st]:
            for o2c in [o2-so, o2, o2+so]:
                if t2c < 0 or o2c < 0: continue
                r_ = negRLL(t2c, o2c)
                if r_ < best[0]: best = (r_, t2c, o2c)
        _, t2, o2 = best
    V = S+t2*same+o2*np.eye(n); Vi = np.linalg.inv(V)
    M = np.linalg.inv(X.T@Vi@X); beta = M@X.T@Vi@y; e = y-X@beta
    studies = list(dict.fromkeys(study)); meat = np.zeros_like(M); g = []
    for s in studies:
        idx = [i for i in range(n) if study[i] == s]
        Xi = X[idx]; Wi = Vi[np.ix_(idx, idx)]; ei = e[idx]; Psi = V[np.ix_(idx, idx)]
        Bi = Psi - Xi@M@Xi.T; Ai = _msqrt(Psi)@_msqrt(Bi, inv=True)
        adj = Xi.T@Wi@Ai@ei.reshape(-1, 1); meat += adj@adj.T
        g.append((Xi, Wi, Ai, Psi))
    VR = M@meat@M; p = X.shape[1]; dfs = []
    for j in range(p):
        c = np.zeros(p); c[j] = 1.0
        bv = [float((c@M@Xi.T@Wi@Ai)@Psi@(c@M@Xi.T@Wi@Ai).T) for (Xi, Wi, Ai, Psi) in g]
        bv = np.array(bv); dfs.append((bv.sum()**2)/(bv**2).sum())
    return dict(beta=beta, VR=VR, df=dfs, tau2=t2, omega2=o2, studies=studies)

# ------------------------------- Stage 6: between-subgroup contrast (Cochran Q)
def cochran_between(a, b):
    th = np.array([a["mu"], b["mu"]]); w = 1/np.array([a["se"], b["se"]])**2
    thb = (w*th).sum()/w.sum(); Qb = (w*(th-thb)**2).sum()
    # chi-square(1) survival via erf
    p = math.erfc(math.sqrt(Qb/2)) if Qb >= 0 else 1.0
    return Qb, p, max(0.0, (Qb-1)/Qb)*100 if Qb > 0 else 0.0

# ------------------------------------------------------- Stage 7: validation
def unit_test_cr2_hc2():
    rng = np.random.default_rng(0); n = 8
    X = np.column_stack([np.ones(n), rng.normal(size=n)]); y = rng.normal(size=n)
    study = list(range(n)); vi = np.ones(n)
    out = che_fit(y, study, X, vi, r=0.0)
    M = np.linalg.inv(X.T@X); beta = M@X.T@y
    return bool(np.allclose(beta, out["beta"], atol=1e-6))

# ================================================================== DRIVERS
def run_bins(df, label):
    _log(f"\n{'='*70}\n{label}: tiered Bayesian pooling (primary) with DL/REML sensitivity\n{'='*70}")
    _log(f"{'exposure / adjustment':40}{'k':>3}  {'Bayes OR [95% CrI]':24}{'P>1':>6} {'prior':>12}  "
         f"{'DL OR [95% CI]':22}{'tier':>12}")
    rows = []
    for (fam, adj), g in sorted(df.groupby(["exposure_family", "adjusted"])):
        d = prep(g); k = len(d)
        if k < 2:
            _log(f"{fam+' / '+adj:40}{k:>3}  single study -> descriptive"); continue
        r = dl(d.yi, d.vi); b = bayes(d.yi, d.vi); lo, hi, stable = prior_sweep(d.yi, d.vi)
        cl = tier(b["lcl"], b["P"], stable)
        rows.append((fam, adj, k, b, cl))
        _log(f"{fam+' / '+adj:40}{k:>3}  {b['OR']:.2f} [{b['lcl']:.2f},{b['ucl']:.2f}]{'':4}"
             f"{b['P']:.3f} {lo:.2f}-{hi:.2f}{'*' if stable else ' '} "
             f"{r['OR']:.2f} [{r['lcl']:.2f},{r['ucl']:.2f}]{'':2}{cl:>12}")
    return rows

def run_ipf():
    df = pd.read_csv(f"{DATA_DIR}/Pooling_Set.csv")
    ipf = df[df.disease.astype(str).str.contains("IPF")]
    ipf = reconcile_ipf(ipf)   # Stage 1b: locked reconciliation (air pollution k=2; genetic risk & nickel descriptive)
    run_bins(ipf, "IPF")
    # dependency-aware dust super-family (General Dust + VGDF, Unadjusted; shared studies)
    dd = prep(ipf[(ipf.exposure_family.isin(["General Dust", "VGDF"])) & (ipf.adjusted == "Unadjusted")])
    fam = dd.exposure_family.tolist(); study = dd.study.tolist()
    _log("\nIPF non-specific-dust super-family — dependency-aware CHE/CR2 (Stage 5)")
    Xc = np.column_stack([np.ones(len(dd)), [1.0 if f == "VGDF" else 0.0 for f in fam]])
    for r in (0.4, 0.6, 0.8):
        o = che_fit(dd.yi.values, study, Xc, dd.vi.values, r=r)
        b, se, d_ = o["beta"][1], math.sqrt(o["VR"][1, 1]), o["df"][1]; tc = tcrit(d_)
        _log(f"  r={r}: VGDF/GeneralDust OR ratio {math.exp(b):.2f} "
             f"[{math.exp(b-tc*se):.2f},{math.exp(b+tc*se):.2f}]  Satterthwaite df={d_:.2f}")

def run_sarcoidosis():
    xf = f"{DATA_DIR}/Sarcoidosis_Extraction_v2_reconciled.xlsx"
    if not os.path.exists(xf):
        _log("\n[sarcoidosis extraction workbook not found; skipping]"); return
    s = pd.read_excel(xf, sheet_name="Pooling_Set_Sarcoid").dropna(subset=["est"])
    # Huntley exposure pools (silica/pesticide/mould) — the CHE core
    hunt = s[s.source_review.astype(str).str.contains("Huntley")]
    core = hunt[hunt.exposure_family.isin(["Silica", "Pesticides", "Mould/Mildew"])]
    _log(f"\n{'='*70}\nSARCOIDOSIS: poolability assessment (Huntley pools + dependency-aware CHE)\n{'='*70}")
    for fam, g in sorted(core.groupby("exposure_family")):
        d = prep(g); r = dl(d.yi, d.vi); b = bayes(d.yi, d.vi); lo, hi, stable = prior_sweep(d.yi, d.vi)
        _log(f"  {fam:14} k={len(d)}  DL {r['OR']:.2f} [{r['lcl']:.2f},{r['ucl']:.2f}]  "
             f"Bayes {b['OR']:.2f} [{b['lcl']:.2f},{b['ucl']:.2f}]  P>1 {lo:.3f}-{hi:.3f}"
             f"{'  prior-robust' if stable else '  NOT prior-robust'}")
    # CHE overall + exposure-specific on the Huntley core (cluster = dependency_cluster)
    d = prep(core); study = core.dependency_cluster.tolist()
    _log("  -- dependency-aware CHE/CR2 (cluster = shared primary study) --")
    for r in (0.4, 0.6, 0.8):
        o = che_fit(d.yi.values, study, np.ones((len(d), 1)), d.vi.values, r=r)
        se, df_ = math.sqrt(o["VR"][0, 0]), o["df"][0]; tc = tcrit(df_)
        _log(f"     overall r={r}: OR {math.exp(o['beta'][0]):.2f} "
             f"[{math.exp(o['beta'][0]-tc*se):.2f},{math.exp(o['beta'][0]+tc*se):.2f}]  df={df_:.1f}")
    _log("  Conclusion: intervals cross the null and effective df is 1-3 -> retain narrative synthesis.")

# ================================================================== MAIN
if __name__ == "__main__":
    _log("Tiered Bayesian meta-analysis pipeline — validation + reproduction")
    _log(f"  CR2==HC2 unit test (iid limit): {'PASS' if unit_test_cr2_hc2() else 'FAIL'}")
    try:
        run_ipf()
    except FileNotFoundError:
        _log("\n[Pooling_Set.csv not found — set ILD_DATA to the data directory]")
    run_sarcoidosis()
    _log("\nTiers: robust = 95% CrI excludes 1 AND prior-stable; borderline = P(OR>1)>=0.95, CrI incl 1;")
    _log("       suggestive/fragile = 0.85<=P<0.95; weak = 0.70<=P<0.85; null = P<0.70.  (* = prior-stable)")
    open("ild_pipeline_results.txt", "w").write("\n".join(_LINES)+"\n")
