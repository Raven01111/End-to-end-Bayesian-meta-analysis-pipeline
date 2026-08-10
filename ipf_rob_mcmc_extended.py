#!/usr/bin/env python3
"""
IPF ROB meta-regression — EXTENDED MCMC.
Adds to the base model (i) a RANDOM ROB SLOPE by exposure (ROB x exposure interaction,
so the bias effect can differ by exposure — e.g. silica) and (ii) a STUDY random effect
(shared-study dependence; a study contributing to several exposures is correlated).

    y[i] ~ Normal( mu + a[e_i] + s[study_i] + (beta + b[e_i]) * x_i ,  v[i] + tau^2 )
    a[e]     ~ Normal(0, sigma_exp^2)      # exposure intercept (partial pooling)
    b[e]     ~ Normal(0, sigma_slope^2)    # exposure-specific ROB slope deviation
    s[study] ~ Normal(0, sigma_study^2)    # study random effect (dependence)
    mu ~ N(0,10^2); beta ~ N(0,1); tau, sigma_* ~ HalfNormal(1)

Coefficient block (mu, beta, a[.], b[.], s[.]) drawn jointly/exactly from its Gaussian
full conditional; the four SDs use random-walk Metropolis. numpy only.
"""
import math, re
import numpy as np, pandas as pd, openpyxl

UP = "/sessions/bold-pensive-shannon/mnt/uploads"; Z = 1.95996
wb = openpyxl.load_workbook(f"{UP}/Civ Primary ROB.xlsx", data_only=True)
rows = list(wb["Sheet1"].iter_rows(values_only=True)); CRIT = [17, 21, 25]
ALLD = [13, 17, 19, 21, 23, 25, 27, 29, 31, 33]
def risk_level(r):
    crit = [str(r[c]).strip() if r[c] else "" for c in CRIT]; alld = [str(r[c]).strip() if r[c] else "" for c in ALLD]
    if any(v == "High risk of bias" for v in crit): return "High"
    if any(v == "Probably High risk of bias" for v in crit) or any(v == "High risk of bias" for v in alld): return "Moderate"
    return "Low"
rob_full = [dict(auth=str(r[8] or ""), year=str(r[9]).strip() if r[9] else "", risk=risk_level(r)) for r in rows[1:]]
DRAFT = {("Cui", "2023"): "Low", ("Wang", "2023"): "Low", ("Berge", "2003"): "Moderate", ("Kitamura", "2007"): "Moderate"}
ps = pd.read_csv(f"{UP}/Pooling_Set.csv"); ps = ps[ps.disease.str.contains("IPF")].copy()
ps["surname"] = ps.study.map(lambda s: str(s).split()[0])
ps["year"] = ps.study.map(lambda s: (re.search(r"(19|20)\d\d", str(s)) or [""])[0] if re.search(r"(19|20)\d\d", str(s)) else "")
def match_risk(sur, yr):
    if (sur, yr) in DRAFT: return DRAFT[(sur, yr)]
    pat = re.compile(r"\b"+re.escape(sur)+r"\b", re.I)
    for rr in rob_full:
        if rr["year"] == yr and pat.search(rr["auth"]): return rr["risk"]
    return None
ps["risk"] = ps.apply(lambda r: match_risk(r.surname, r.year), axis=1)
ps.loc[(ps.study.str.contains("Cui", na=False)) & (ps.exposure_family == "Air pollution"), ["est", "lcl", "ucl"]] = [1.09, 1.02, 1.17]
ps = ps[~((ps.study.str.contains("Johannson", na=False)) & (ps.exposure_family == "Air pollution"))]  # exacerbation outcome, not incident IPF
ps = ps[ps.exposure_family != "Genetic risk"]  # reframed as effect modifier, not an exposure
ps = ps.dropna(subset=["risk"]); ps["adj_rank"] = (ps.adjusted == "Adjusted").astype(int)
ps = ps.sort_values("adj_rank", ascending=False).drop_duplicates(["study", "exposure_family"]).reset_index(drop=True)
ps["x"] = (ps.risk != "Low").astype(float); ps["y"] = np.log(ps.est); ps["v"] = ((np.log(ps.ucl)-np.log(ps.lcl))/(2*Z))**2
counts = ps.exposure_family.value_counts(); ps = ps[ps.exposure_family.isin(counts[counts >= 2].index)].reset_index(drop=True)
exps = sorted(ps.exposure_family.unique()); J = len(exps); ei = ps.exposure_family.map({e: j for j, e in enumerate(exps)}).values
studs = sorted(ps.study.unique()); M = len(studs); si = ps.study.map({s: k for k, s in enumerate(studs)}).values
y, v, x, n = ps.y.values, ps.v.values, ps.x.values, len(ps)
print(f"n={n}, exposures J={J}, studies M={M}, >Low fraction={x.mean():.2f}")

# design: [1 | x | a(J) | b(J: exp*x) | s(M)]
Xd = np.column_stack([np.ones(n), x, np.eye(J)[ei], np.eye(J)[ei]*x[:, None], np.eye(M)[si]])
p = Xd.shape[1]; iA, iB, iS = 2, 2+J, 2+2*J
def hn(t): return -0.5*t*t if t > 0 else -np.inf
def loglik(coef, tau2):
    eta = Xd@coef; s2 = v+tau2; return -0.5*np.sum(np.log(2*np.pi*s2)+(y-eta)**2/s2)
def run(seed, n_iter=26000, warm=9000):
    rng = np.random.default_rng(seed)
    coef = np.concatenate([[rng.normal(0, .3), rng.normal(0, .3)], rng.normal(0, .2, p-2)])
    tau, sx, sb, ss = .25, .2, .15, .15; st = dict(t=.22, x=.4, b=.4, s=.4); keep = []
    for it in range(n_iter):
        tau2 = tau*tau; W = 1.0/(v+tau2)
        P0 = np.zeros(p); P0[0] = 1/100; P0[1] = 1/1
        P0[iA:iB] = 1/sx**2; P0[iB:iS] = 1/sb**2; P0[iS:] = 1/ss**2
        Lam = Xd.T@(W[:, None]*Xd) + np.diag(P0); L = np.linalg.cholesky(Lam)
        mnt = np.linalg.solve(Lam, Xd.T@(W*y)); coef = mnt + np.linalg.solve(L.T, rng.normal(size=p))
        # MH tau
        for _ in range(1):
            lt = math.log(tau); ltp = lt+rng.normal(0, st['t']); tp = math.exp(ltp)
            if math.log(rng.random()) < (loglik(coef, tp**2)+hn(tp)+ltp)-(loglik(coef, tau2)+hn(tau)+lt): tau = tp
        # MH sigma_exp / slope / study given their coeff blocks
        def mh_sd(sd, block, key):
            l = math.log(sd); lp = l+rng.normal(0, st[key]); sp = math.exp(lp)
            new = -0.5*np.sum(np.log(2*np.pi*sp**2)+block**2/sp**2)+hn(sp)+lp
            old = -0.5*np.sum(np.log(2*np.pi*sd**2)+block**2/sd**2)+hn(sd)+l
            return sp if math.log(rng.random()) < (new-old) else sd
        sx = mh_sd(sx, coef[iA:iB], 'x'); sb = mh_sd(sb, coef[iB:iS], 'b'); ss = mh_sd(ss, coef[iS:], 's')
        if it >= warm: keep.append(np.concatenate([[coef[0], coef[1], tau, sx, sb, ss], coef[iB:iS]]))
    return np.array(keep)

chains = [run(s) for s in [7, 17, 27, 37]]
S = np.stack(chains); flat = S.reshape(-1, S.shape[2])
names = ["mu", "beta", "tau", "sigma_exp", "sigma_slope", "sigma_study"] + [f"bROB[{e}]" for e in exps]
def rhat(a):
    m, nn = a.shape; h = nn//2; sub = np.concatenate([a[:, :h], a[:, h:2*h]], 0)
    B = h*sub.mean(1).var(ddof=1); Wv = sub.var(1, ddof=1).mean()
    return math.sqrt(((h-1)/h*Wv+B/h)/Wv) if Wv > 0 else float("nan")
out = []
def w(s=""): print(s); out.append(s)
w("IPF ROB meta-regression — EXTENDED (random ROB slope by exposure + study random effect)")
w(f"4 chains x {S.shape[1]} draws. n={n}, exposures={J}, studies={M}.\n")
w(f"{'param':22}{'mean':>8}{'2.5%':>9}{'97.5%':>9}{'Rhat':>7}")
for i, nm in enumerate(names):
    q = np.percentile(flat[:, i], [2.5, 97.5])
    w(f"{nm:22}{flat[:,i].mean():8.3f}{q[0]:9.3f}{q[1]:9.3f}{rhat(S[:,:,i]):7.3f}")
b = flat[:, 1]
w(f"\nAverage ROB effect exp(beta): {math.exp(b.mean()):.2f} [{math.exp(np.percentile(b,2.5)):.2f}, {math.exp(np.percentile(b,97.5)):.2f}], P(beta>0)={np.mean(b>0):.3f}")
w("Exposure-specific ROB effect  exp(beta + bROB[e])  (>Low vs Low, within exposure):")
for j, e in enumerate(exps):
    tot = flat[:, 1] + flat[:, 6+j]
    w(f"  {e:26} x{math.exp(tot.mean()):.2f} [{math.exp(np.percentile(tot,2.5)):.2f}, {math.exp(np.percentile(tot,97.5)):.2f}]  P(>0)={np.mean(tot>0):.2f}")
maxr = max(rhat(S[:, :, i]) for i in range(len(names)))
w(f"\nMax split-Rhat = {maxr:.3f} -> {'RESOLVED' if maxr < 1.01 else 'variance components mix slowly (expected; see note)'}")
open("ipf_rob_mcmc_extended_results.txt", "w").write("\n".join(out)+"\n")
