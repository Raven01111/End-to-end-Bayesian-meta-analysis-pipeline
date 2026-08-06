# ipf_all_exposures_aligned.py — IPF exposure-specific tiered Bayesian pooling (aligned to final results)
import math, os
import numpy as np, pandas as pd
from statistics import NormalDist
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

_N = NormalDist(); Z = _N.inv_cdf(0.975)
XLSX = "ILD_Meta-analysis_Reshaped.xlsx"
FDIR = "ipf_forests"; os.makedirs(FDIR, exist_ok=True)

def chi2_sf(x, df):
    a, xx = df/2., x/2.
    if xx < a+1:
        ap, s, d = a, 1./a, 1./a
        for _ in range(2000):
            ap += 1; d *= xx/ap; s += d
            if abs(d) < abs(s)*1e-15: break
        return 1 - s*math.exp(-xx + a*math.log(xx) - math.lgamma(a))
    tiny = 1e-30; b = xx+1-a; c = 1/tiny; d = 1/b; h = d
    for i in range(1, 2000):
        an = -i*(i-a); b += 2; d = an*d + b
        if abs(d) < tiny: d = tiny
        c = b + an/c
        if abs(c) < tiny: c = tiny
        d = 1/d; h *= d*c
    return math.exp(-xx + a*math.log(xx) - math.lgamma(a))*h

def prep(g):
    g = g.copy()
    for c in ["est", "lcl", "ucl"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["est", "lcl", "ucl"])
    zm = g.study.str.contains("Zubairi")                      # >>> C2
    g.loc[zm, ["est", "lcl", "ucl"]] = [1.11, 0.65, 1.90]
    g["study"] = g.study.str.replace("Zubairi 2021", "Zubairi 2023")
    g["yi"] = np.log(g.est); g["sei"] = (np.log(g.ucl)-np.log(g.lcl))/(2*Z); g["vi"] = g.sei**2
    g["a1"] = g.study.str.split().str[0]
    g = g.sort_values("study").reset_index(drop=True)
    keep, seen = [], set()                                    # >>> C1
    for _, r in g.iterrows():
        key = (round(r.est, 4), round(r.lcl, 4), round(r.ucl, 4), r.a1)
        if key in seen: continue
        seen.add(key); keep.append(r.name)
    return g.loc[keep].reset_index(drop=True)

def dl(yi, vi):
    yi = np.asarray(yi, float); vi = np.asarray(vi, float); k = len(yi)
    wi = 1/vi; yF = (wi*yi).sum()/wi.sum(); Q = (wi*(yi-yF)**2).sum(); df = k-1
    C = wi.sum()-(wi**2).sum()/wi.sum(); t2 = max(0., (Q-df)/C) if C > 0 else 0.
    wr = 1/(vi+t2); mu = (wr*yi).sum()/wr.sum(); se = math.sqrt(1/wr.sum())
    s2 = (k-1)*wi.sum()/(wi.sum()**2-(wi**2).sum()); I2 = 100*t2/(t2+s2) if t2+s2 > 0 else 0
    return dict(k=k, mu=mu, se=se, OR=math.exp(mu), lcl=math.exp(mu-Z*se), ucl=math.exp(mu+Z*se),
                I2=I2, p=2*(1-_N.cdf(abs(mu/se))))

def reml(yi, vi):
    yi = np.asarray(yi, float); vi = np.asarray(vi, float); k = len(yi)
    taus = np.linspace(0, 3, 20001); best = (-1e18, 0.)
    for t in taus:
        t2 = t*t; w = 1/(vi+t2); mu = (w*yi).sum()/w.sum()
        ll = -0.5*(np.sum(np.log(vi+t2)) + math.log(w.sum()) + np.sum(w*(yi-mu)**2))
        if ll > best[0]: best = (ll, t2)
    t2 = best[1]; w = 1/(vi+t2); mu = (w*yi).sum()/w.sum(); se = math.sqrt(1/w.sum())
    wi = 1/vi; s2 = (k-1)*wi.sum()/(wi.sum()**2-(wi**2).sum()); I2 = 100*t2/(t2+s2) if t2+s2 > 0 else 0
    return dict(OR=math.exp(mu), lcl=math.exp(mu-Z*se), ucl=math.exp(mu+Z*se), I2=I2, p=2*(1-_N.cdf(abs(mu/se))))

def bayesmeta(yi, vi, prior="HN", scale=1.0, ntau=3000, nmu=4000, taumax=4.):
    yi = np.asarray(yi, float); vi = np.asarray(vi, float)
    lpri = (lambda t: -0.5*(t/scale)**2) if prior == "HN" else (lambda t: -math.log(1+(t/scale)**2))
    taus = np.linspace(1e-5, taumax, ntau); lp = np.empty(ntau); mh = np.empty(ntau); Vm = np.empty(ntau)
    for j, t in enumerate(taus):
        t2 = t*t; w = 1/(vi+t2); mu = (w*yi).sum()/w.sum(); Vv = 1/w.sum()
        lp[j] = (-0.5*np.sum(np.log(2*math.pi*(vi+t2))) - 0.5*np.sum((yi-mu)**2*w)
                 + 0.5*math.log(2*math.pi*Vv) + lpri(t))
        mh[j] = mu; Vm[j] = Vv
    lp -= lp.max(); pw = np.exp(lp); pw /= pw.sum(); tau_med = float(np.interp(0.5, np.cumsum(pw), taus))
    sp = 8*math.sqrt(Vm.max()+taumax**2); mg = np.linspace(mh.min()-sp, mh.max()+sp, nmu)
    dens = np.zeros_like(mg); pdens = np.zeros_like(mg)
    for j in range(ntau):
        dens  += pw[j]*np.exp(-0.5*(mg-mh[j])**2/Vm[j])/math.sqrt(2*math.pi*Vm[j])
        pv = Vm[j]+taus[j]**2
        pdens += pw[j]*np.exp(-0.5*(mg-mh[j])**2/pv)/math.sqrt(2*math.pi*pv)
    def qf(den):
        den = den/np.trapezoid(den, mg)
        cdf = np.concatenate([[0], np.cumsum((den[1:]+den[:-1])/2*np.diff(mg))]); cdf /= cdf[-1]
        return (lambda p: float(np.interp(p, cdf, mg))), cdf
    qm, cdfm = qf(dens); qp, _ = qf(pdens)
    return dict(OR=math.exp(qm(0.5)), lcl=math.exp(qm(0.025)), ucl=math.exp(qm(0.975)),
                P=float(1-np.interp(0., mg, cdfm)), tau_med=tau_med,
                pl=math.exp(qp(0.025)), pu=math.exp(qp(0.975)))

def classify(bl, P):
    if bl > 1: return "robust"
    if P >= 0.95: return "borderline"
    if P >= 0.85: return "fragile"
    if P >= 0.70: return "weak"
    return "null"

def forest(d, dd, b, cls, title, path):
    col = {"robust": "#1D7A1F", "borderline": "#BA7517", "fragile": "#C0392B", "weak": "#7F7F7F", "null": "#7F7F7F"}[cls]
    st = sorted([[s, e, l, u] for s, e, l, u in zip(d.study, d.est, d.lcl, d.ucl)], key=lambda x: x[1])
    n = len(st); fig, ax = plt.subplots(figsize=(7.2, 1.0+0.42*(n+3))); y = np.arange(n)[::-1]+3
    for i, (lab, orr, lcl, ucl) in enumerate(st):
        ax.plot([lcl, ucl], [y[i], y[i]], color="#555", lw=1.2); ax.plot(orr, y[i], 's', color="#333", ms=5)
        ax.text(0.012, y[i], lab, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8)
    l, u, x = b['lcl'], b['ucl'], b['OR']
    ax.plot([l, u], [1.4, 1.4], color=col, lw=2.4)
    ax.fill([l, x, u, x], [1.4, 1.68, 1.4, 1.12], color=col, alpha=0.35)
    ax.text(0.012, 1.4, "Bayesian pooled (τ² marg.)", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, fontweight="bold")
    ax.plot([b['pl'], b['pu']], [0.4, 0.4], color=col, lw=1.0, ls=(0, (4, 2)))
    ax.text(0.012, 0.4, "95% prediction interval", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, style="italic")
    ax.axvline(1, color="#C0392B", ls="--", lw=1); ax.set_xscale("log")
    lo = min([s[2] for s in st]+[b['pl']])*0.85; hi = max([s[3] for s in st]+[b['pu']])*1.15
    ax.set_xlim(lo, hi); ax.set_ylim(-0.2, y.max()+0.8); ax.set_yticks([]); ax.set_xlabel("Odds ratio (log scale)", fontsize=9)
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(title, fontsize=9.5)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

PRIORS = [("HN", 0.5), ("HN", 1.0), ("HN", 2.0), ("HC", 1.0)]

if __name__ == "__main__":
    ipf = pd.read_excel(XLSX, sheet_name="Pooling_Set")
    ipf = ipf[ipf.disease == "IPF/Pulmonary Fibrosis"].copy()
    # >>> C3  Air pollution -> two-study incidence pool: drop Johannson 2014 (exacerbation, not
    #         incident IPF); Cui 2023 uses its PM2.5 aHR 1.09 [1.02,1.17].
    # >>> C4  Genetic risk -> effect modifier, reported descriptively (not pooled).
    # >>> C5  Nickel (speciation) -> uninformative k=2 pool, reported descriptively (not pooled).
    _ap = ipf.exposure_family == "Air pollution"
    ipf.loc[_ap & ipf.study.str.contains("Cui", na=False), ["est", "lcl", "ucl"]] = [1.09, 1.02, 1.17]
    ipf = ipf[~(_ap & ipf.study.str.contains("Johannson", na=False))]
    ipf = ipf[~ipf.exposure_family.isin(["Genetic risk", "Nickel (speciation)"])]
    def fn(fam, adj): return "".join(c for c in f"{fam}_{adj}" if c.isalnum() or c == "_").replace(" ", "")
    print(f"{'exposure / adj':40}{'k':>3}  {'DL OR[CI]':22}{'Bayes OR[CrI]':24}{'P>1':>6} {'class':>11}")
    for (fam, adj), g in sorted(ipf.groupby(["exposure_family", "adjusted"])):
        d = prep(g); k = len(d)
        if k < 2:
            print(f"{fam+' / '+adj:40}{k:>3}  singleton -> descriptive"); continue
        r = dl(d.yi, d.vi); rr = reml(d.yi, d.vi); b = bayesmeta(d.yi, d.vi)
        Ps = [bayesmeta(d.yi, d.vi, prior=p, scale=s)["P"] for p, s in PRIORS]
        cls = classify(b["lcl"], b["P"])
        forest(d, r, b, cls, f"IPF · {fam} ({adj}) — k={k}, I²={r['I2']:.0f}%, P(OR>1)={b['P']:.3f}  [{cls}]",
               f"{FDIR}/forest_{fn(fam, adj)}.png")
        print(f"{fam+' / '+adj:40}{k:>3}  {r['OR']:.2f}[{r['lcl']:.2f},{r['ucl']:.2f}]{'':6}"
              f"{b['OR']:.2f}[{b['lcl']:.2f},{b['ucl']:.2f}]{'':6}{b['P']:.3f} {cls:>11}"
              f"  (pred [{b['pl']:.2f},{b['pu']:.2f}], prior {min(Ps):.2f}-{max(Ps):.2f})")

    def cochran(a, b):
        th = np.array([a['mu'], b['mu']]); se = np.array([a['se'], b['se']]); wg = 1/se**2
        thb = (wg*th).sum()/wg.sum(); Qb = (wg*(th-thb)**2).sum()
        return Qb, chi2_sf(Qb, 1), max(0, (Qb-1)/Qb)*100
    gd = prep(ipf[(ipf.exposure_family == "General Dust") & (ipf.adjusted == "Unadjusted")])
    vg = prep(ipf[(ipf.exposure_family == "VGDF") & (ipf.adjusted == "Unadjusted")])
    md = prep(ipf[(ipf.exposure_family == "Metal Dust (occupational)") & (ipf.adjusted == "Adjusted")])
    ni = prep(ipf[(ipf.exposure_family == "Nickel (speciation)") & (ipf.adjusted == "Adjusted")])
    dQ = cochran(dl(gd.yi, gd.vi), dl(vg.yi, vg.vi))
    print(f"\nDust GD(k={len(gd)}) vs VGDF(k={len(vg)}): Cochran Q_between={dQ[0]:.3f} p={dQ[1]:.4f} I2b={dQ[2]:.1f}%  "
          f"(GD {dl(gd.yi,gd.vi)['OR']:.2f} vs VGDF {dl(vg.yi,vg.vi)['OR']:.2f}); RVE underpowered (~1-2 df, see dust re-analysis)")
    if len(ni) >= 2:
        mQ = cochran(dl(md.yi, md.vi), dl(ni.yi, ni.vi))
        print(f"Metals MD(k={len(md)}) vs Ni(k={len(ni)}): Cochran Q_between={mQ[0]:.3f} p={mQ[1]:.4f} I2b={mQ[2]:.1f}%  (disjoint; no RVE)")
    else:
        print("Metals MD vs Ni: not run — Nickel (speciation) is reported descriptively, not pooled (C5).")
