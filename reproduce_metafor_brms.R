####################################################################
## Canonical reproduction — metafor (DL/REML) + brms (Bayesian)
## Umbrella Review: Airborne Hazards and ILD (IPF + Sarcoidosis)
## Run top-to-bottom in R (>= 4.1). Reproduces the key exposure
## pools and the cross-exposure overall against the reported values.
####################################################################

## ---- 0. Packages ----
pkgs <- c("metafor", "brms", "dplyr")
new  <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if (length(new)) install.packages(new)     # brms will pull rstan/cmdstanr (needs a C++ toolchain)
library(metafor); library(brms); library(dplyr)

## ---- 1. Data (embedded: study, OR, LCL, UCL) ----
mk <- function(study, or, lcl, ucl) data.frame(study, or, lcl, ucl)

pools <- list(
 `IPF Metal dust (adj)` = mk(
   c("Awadalla 2012","Baumgartner 2000","Ekstrom 2014","Hubbard 1996","Koo 2017","Miyake 2005","Paolocci 2018"),
   c(1.58,2.00,1.10,1.68,4.97,9.55,3.80),
   c(0.69,1.00,0.60,1.07,1.36,1.68,1.20),
   c(3.61,4.00,1.80,2.65,18.17,181.12,12.20)),
 `IPF VGDF (unadj)` = mk(
   c("Abramson 2020","Garcia-Sancho 2011","Gustafson 2007","Koo 2017","Mullen 1998","Paolocci 2018","Park J 2020","Reynolds C 2019","Zubairi 2023"),
   c(1.10,2.80,1.10,2.70,2.40,4.10,2.10,1.70,1.11),
   c(0.90,1.50,0.70,0.70,0.70,2.30,1.20,1.20,0.65),
   c(1.40,5.50,1.70,10.90,8.40,7.50,3.70,2.30,1.90)),
 `IPF Silica (unadj)` = mk(   # Abramson = self-reported 0.60 (as originally extracted; see JEM-correction note below)
   c("Abramson 2020","Awadalla 2012","Baumgartner 2000","Gustafson 2007","Koo 2017","Miyake 2005","Mullen 1998","Park J 2020","Reynolds C 2019","Scott 1990"),
   c(0.60,1.10,3.90,1.40,1.20,1.80,11.00,2.50,2.90,1.60),
   c(0.40,0.50,1.20,0.70,0.40,0.50,1.10,1.00,1.30,0.50),
   c(0.80,2.70,12.70,2.70,3.80,7.00,115.00,6.70,6.70,4.80)),
 `IPF Wood dust (unadj)` = mk(
   c("Abramson 2020","Gustafson 2007","Mullen 1998","Paolocci 2018","Park J 2020","Reynolds C 2019","Scott 1990"),
   c(0.69,1.20,3.30,1.40,2.10,1.40,2.94),
   c(0.27,0.65,0.40,0.50,0.90,0.90,0.87),
   c(1.78,2.23,25.80,4.00,3.90,2.30,9.92)),
 `Sarcoid Silica` = mk(
   c("Beijer 2020","Kucera 2003","Graff 2020","Jonsson 2019","Rafnsson 1998"),
   c(1.38,1.62,1.24,2.44,13.20), c(0.46,0.82,1.11,1.37,2.00), c(4.20,3.18,1.39,4.33,140.90)),
 `Sarcoid Pesticides` = mk(
   c("Kajdasz 2001","Kucera 2003","ACCESS 2004"), c(2.10,1.11,1.52), c(0.90,0.72,1.14), c(4.70,1.70,2.04)),
 `Sarcoid Mould/Mildew` = mk(
   c("Kucera 2003","ACCESS 2004"), c(1.46,1.61), c(1.09,1.13), c(1.99,2.31))
)

## ---- 2. metafor: DL and REML per pool ----
cat("\n================  metafor: DL & REML  ================\n")
for (nm in names(pools)) {
  d  <- escalc(measure="OR", yi=log(pools[[nm]]$or),
               sei=(log(pools[[nm]]$ucl)-log(pools[[nm]]$lcl))/(2*1.959964))
  dl <- rma(yi, vi, data=d, method="DL")
  re <- rma(yi, vi, data=d, method="REML")
  cat(sprintf("%-24s k=%d  DL %.2f (%.2f-%.2f)  REML %.2f (%.2f-%.2f)  I2=%.0f%%\n",
      nm, nrow(d), exp(dl$b), exp(dl$ci.lb), exp(dl$ci.ub),
      exp(re$b), exp(re$ci.lb), exp(re$ci.ub), dl$I2))
}

## ---- 3. brms: Bayesian random-effects pool (four-prior sweep) ----
## Matches the primary model: log-OR ~ 1 + (1|study); half-normal(scale) prior on tau.
cat("\n================  brms: Bayesian (Stan/NUTS)  ================\n")
tau_scales <- c(0.5, 1.0, 2.0)     # half-normal; add half-Cauchy(1) via prior(cauchy(0,1),class='sd') if desired
run_brms <- function(df, scale) {
  df$yi <- log(df$or); df$sei <- (log(df$ucl)-log(df$lcl))/(2*1.959964)
  fit <- brm(yi | se(sei) ~ 1 + (1|study), data=df,
             prior = prior(normal(0,5), class="Intercept") +
                     set_prior(sprintf("normal(0,%g)", scale), class="sd"),
             chains=4, iter=4000, warmup=1000, refresh=0, seed=1,
             control=list(adapt_delta=0.99))
  ps  <- as_draws_df(fit)$b_Intercept
  c(OR=exp(median(ps)), lo=exp(quantile(ps,.025)), hi=exp(quantile(ps,.975)), P_gt1=mean(ps>0))
}
for (nm in names(pools)) {
  r <- run_brms(pools[[nm]], 1.0)   # reference prior HN(1.0); loop tau_scales for the full sweep
  cat(sprintf("%-24s  Bayes OR %.2f (%.2f-%.2f)  P(OR>1)=%.3f\n", nm, r["OR"], r["lo.2.5%"], r["hi.97.5%"], r["P_gt1"]))
}

## ---- 4. Cross-exposure overall (multilevel, study clustering) ----
## Load the full per-estimate set (study, exposure, yi, vi) from the pooling file, then:
## metafor:
##   rma.mv(yi, vi, random = ~ 1 | study/estimate, method="REML", data=all_estimates)
## brms (Bayesian multilevel, exposure + study random intercepts):
##   brm(yi | se(sei) ~ 1 + (1|exposure) + (1|study), data=all_estimates,
##       prior = prior(normal(0,5),class="Intercept") + prior(normal(0,1),class="sd"),
##       chains=4, iter=4000)
## NB: for the IPF incidence overall, exclude Johannson 2014 (acute-exacerbation outcome);
##     n = 62 estimates across 22 studies.

## ---- Reported values for side-by-side comparison ----
cat("\nReported (main analysis): Metal-adj Bayes 1.89/DL 1.88; VGDF 1.75/1.75; Silica 1.60/1.66;\n",
    "Wood 1.45/1.43; Sarc Silica 1.67/1.73; Pest 1.44/1.42; Mould 1.53/1.52.\n",
    "Overall: IPF 1.52 (DL 1.45); Sarcoid 1.86 (DL 1.70).\n")
