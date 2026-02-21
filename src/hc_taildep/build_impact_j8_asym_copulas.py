# src/hc_taildep/build_impact_j8_asym_copulas.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau

from hc_taildep.impact.var_es import build_empirical_quantile, compute_var_es, sample_copula
from hc_taildep.impact.j8_pairwise import ensure_dir, detect_date_col, normalize_date, rolling_rv, finite_quantile, bucket_from_x


def sha12_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def resolve_vars(cfg: Dict[str, Any]) -> Dict[str, Any]:
    s = yaml.safe_dump(cfg, sort_keys=False)
    dv = cfg.get("dataset_version", "")
    s = s.replace("${dataset_version}", str(dv))
    return yaml.safe_load(s)


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    y = x[m]
    n = y.size
    out = np.full_like(x, np.nan, dtype=float)
    if n < 20:
        return out
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    u = ranks / (n + 1.0)
    out[m] = np.clip(u, 1e-12, 1 - 1e-12)
    return out


def fit_theta_clayton_from_tau(tau: float) -> float | None:
    # theta = 2 tau / (1-tau), valid tau in (0,1)
    if not np.isfinite(tau) or tau <= 1e-6 or tau >= 0.999:
        return None
    th = 2.0 * tau / (1.0 - tau)
    return float(th) if (np.isfinite(th) and th > 0) else None


def fit_theta_gumbel_from_tau(tau: float) -> float | None:
    # theta = 1/(1-tau), valid tau in (0,1)
    if not np.isfinite(tau) or tau <= 1e-6 or tau >= 0.999:
        return None
    th = 1.0 / (1.0 - tau)
    th = max(th, 1.0)
    return float(th) if (np.isfinite(th) and th >= 1.0) else None


def tail_dependence_mc(u: np.ndarray, v: np.ndarray, q: float = 0.05) -> Dict[str, float]:
    u = np.clip(u, 1e-12, 1 - 1e-12)
    v = np.clip(v, 1e-12, 1 - 1e-12)
    q = float(q)
    lam_l = float(((u < q) & (v < q)).mean() / q)
    lam_u = float(((u > 1 - q) & (v > 1 - q)).mean() / q)
    return {"lambda_L": lam_l, "lambda_U": lam_u}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_raw = load_yaml(Path(args.config))
    cfg = resolve_vars(cfg_raw)

    dataset_version = str(cfg["dataset_version"])
    returns_path = Path(str(cfg["inputs"]["returns_path"]))

    a1 = str(cfg["pair"]["asset1"])
    a2 = str(cfg["pair"]["asset2"])

    refit_every = int(cfg.get("refit_every", 40))
    rv_window = int(cfg["stress"]["window"])
    calm_q = float(cfg["stress"]["calm_q"])
    stress_q = float(cfg["stress"]["stress_q"])

    n_scenarios = int(cfg.get("n_scenarios", 20000))
    alphas = [float(x) for x in cfg.get("alphas", [0.95, 0.99])]
    seed = int(cfg.get("seed", 123))
    rho_clamp = float(cfg.get("rho_clamp", 1e-6))

    out_name = str(cfg.get("out_name", "j8_asym_copulas")).strip()
    out_dir = Path(f"data/processed/{dataset_version}/impact/j8_asym_copulas/{out_name}")
    ensure_dir(out_dir)
    ensure_dir(out_dir / "tables")
    ensure_dir(out_dir / "figures")

    df = pd.read_csv(returns_path)
    dcol = detect_date_col(df)
    df = normalize_date(df, dcol)
    df = df.sort_values("date").reset_index(drop=True)

    if a1 not in df.columns or a2 not in df.columns:
        raise ValueError(f"returns missing columns for pair: {a1},{a2}")

    dates = df["date"].astype(str).to_numpy()
    r1 = df[a1].to_numpy(float)
    r2 = df[a2].to_numpy(float)

    r_port = 0.5 * r1 + 0.5 * r2
    loss_real = -r_port
    x_rv = rolling_rv(r_port, window=rv_window)

    out_cols = ["date", "bucket", "x_rv", "r_port_real", "loss_real"]
    models = ["static_gauss", "static_clayton", "static_gumbel", "thr_clayton", "thr_gumbel"]
    for m in models:
        for a in alphas:
            tag = int(round(a * 100))
            out_cols += [f"VaR{tag}_{m}", f"ES{tag}_{m}", f"exceed{tag}_{m}"]
    out = pd.DataFrame(index=np.arange(len(dates)), columns=out_cols)
    out["date"] = dates
    out["x_rv"] = x_rv
    out["r_port_real"] = r_port
    out["loss_real"] = loss_real
    out["bucket"] = "mid"

    tail_rows = []

    MIN_TRAIN = int(cfg.get("min_train", 250))
    for start_i in range(0, len(dates), refit_every):
        train_end_i = start_i - 1
        if train_end_i < MIN_TRAIN:
            continue
        end_i = min(len(dates), start_i + refit_every)

        tr = slice(0, train_end_i + 1)
        r1_tr = r1[tr]
        r2_tr = r2[tr]
        x_tr = x_rv[tr]

        thr_stress = finite_quantile(x_tr, stress_q)
        thr_calm = finite_quantile(x_tr, calm_q)

        Q1 = build_empirical_quantile(r1_tr)
        Q2 = build_empirical_quantile(r2_tr)

        # pseudo obs on TRAIN for tau fitting
        u_tr = pseudo_obs(r1_tr)
        v_tr = pseudo_obs(r2_tr)
        m = np.isfinite(u_tr) & np.isfinite(v_tr)
        tau_all = float(kendalltau(u_tr[m], v_tr[m], nan_policy="omit").correlation) if int(m.sum()) > 50 else 0.0

        # baseline gauss rho from tau (elliptical)
        rho = float(np.sin(np.pi * 0.5 * np.clip(tau_all, -0.999, 0.999)))
        rho = float(np.clip(rho, -1 + rho_clamp, 1 - rho_clamp))
        static_gauss = {"rho": rho}

        # asym static
        th_c = fit_theta_clayton_from_tau(tau_all)
        th_g = fit_theta_gumbel_from_tau(tau_all)

        # threshold tau fits on calm/stress subsets
        m_calm = np.isfinite(x_tr) & (x_tr <= thr_calm)
        m_stress = np.isfinite(x_tr) & (x_tr >= thr_stress)

        tau_calm = tau_all
        tau_stress = tau_all
        if int(m_calm.sum()) >= 150:
            uc = pseudo_obs(r1_tr[m_calm])
            vc = pseudo_obs(r2_tr[m_calm])
            mc = np.isfinite(uc) & np.isfinite(vc)
            if int(mc.sum()) >= 80:
                tau_calm = float(kendalltau(uc[mc], vc[mc], nan_policy="omit").correlation)
        if int(m_stress.sum()) >= 150:
            us = pseudo_obs(r1_tr[m_stress])
            vs = pseudo_obs(r2_tr[m_stress])
            ms = np.isfinite(us) & np.isfinite(vs)
            if int(ms.sum()) >= 80:
                tau_stress = float(kendalltau(us[ms], vs[ms], nan_policy="omit").correlation)

        th_c_calm = fit_theta_clayton_from_tau(tau_calm)
        th_c_stress = fit_theta_clayton_from_tau(tau_stress)
        th_g_calm = fit_theta_gumbel_from_tau(tau_calm)
        th_g_stress = fit_theta_gumbel_from_tau(tau_stress)

        # tail dependence diagnostics (MC) per bucket, per family (ANNEX)
        for bname, thC, thG in [("calm", th_c_calm, th_g_calm), ("stress", th_c_stress, th_g_stress)]:
            rng = np.random.default_rng(seed + 1000 + start_i + (0 if bname == "calm" else 1))
            if thC is not None:
                u_s, v_s = sample_copula("clayton", {"theta": thC}, 20000, rng, rho_clamp=rho_clamp)
            else:
                u_s, v_s = sample_copula("gauss", static_gauss, 20000, rng, rho_clamp=rho_clamp)
            td = tail_dependence_mc(u_s, v_s, q=0.05)
            tail_rows.append({"refit_index": int(start_i), "bucket": bname, "family": "clayton", "theta": thC, **td})

            rng = np.random.default_rng(seed + 2000 + start_i + (0 if bname == "calm" else 1))
            if thG is not None:
                u_s, v_s = sample_copula("gumbel", {"theta": thG}, 20000, rng, rho_clamp=rho_clamp)
            else:
                u_s, v_s = sample_copula("gauss", static_gauss, 20000, rng, rho_clamp=rho_clamp)
            td = tail_dependence_mc(u_s, v_s, q=0.05)
            tail_rows.append({"refit_index": int(start_i), "bucket": bname, "family": "gumbel", "theta": thG, **td})

        # score dates in block
        for i in range(start_i, end_i):
            bucket = bucket_from_x(float(x_rv[i]), thr_calm=thr_calm, thr_stress=thr_stress)
            out.loc[i, "bucket"] = bucket

            rng = np.random.default_rng(int(hashlib.sha256((str(seed) + "|" + str(dates[i])).encode()).hexdigest()[:8], 16))

            # static models
            u_g, v_g = sample_copula("gauss", static_gauss, n_scenarios, rng, rho_clamp=rho_clamp)
            if th_c is not None:
                u_c, v_c = sample_copula("clayton", {"theta": th_c}, n_scenarios, rng, rho_clamp=rho_clamp)
            else:
                u_c, v_c = sample_copula("gauss", static_gauss, n_scenarios, rng, rho_clamp=rho_clamp)

            if th_g is not None:
                u_u, v_u = sample_copula("gumbel", {"theta": th_g}, n_scenarios, rng, rho_clamp=rho_clamp)
            else:
                u_u, v_u = sample_copula("gauss", static_gauss, n_scenarios, rng, rho_clamp=rho_clamp)

            # threshold models
            thC = th_c_stress if bucket == "stress" else th_c_calm
            thG = th_g_stress if bucket == "stress" else th_g_calm

            if thC is not None:
                u_ct, v_ct = sample_copula("clayton", {"theta": thC}, n_scenarios, rng, rho_clamp=rho_clamp)
            else:
                u_ct, v_ct = sample_copula("gauss", static_gauss, n_scenarios, rng, rho_clamp=rho_clamp)

            if thG is not None:
                u_gt, v_gt = sample_copula("gumbel", {"theta": thG}, n_scenarios, rng, rho_clamp=rho_clamp)
            else:
                u_gt, v_gt = sample_copula("gauss", static_gauss, n_scenarios, rng, rho_clamp=rho_clamp)

            sims = {
                "static_gauss": (u_g, v_g),
                "static_clayton": (u_c, v_c),
                "static_gumbel": (u_u, v_u),
                "thr_clayton": (u_ct, v_ct),
                "thr_gumbel": (u_gt, v_gt),
            }

            for name, (uu, vv) in sims.items():
                losses = -(0.5 * Q1(uu) + 0.5 * Q2(vv))
                for a in alphas:
                    tag = int(round(a * 100))
                    var, es = compute_var_es(losses, a)
                    out.loc[i, f"VaR{tag}_{name}"] = var
                    out.loc[i, f"ES{tag}_{name}"] = es
                    lr = float(out.loc[i, "loss_real"])
                    out.loc[i, f"exceed{tag}_{name}"] = 1.0 if (np.isfinite(lr) and np.isfinite(var) and (lr > var)) else 0.0

    pred_path = out_dir / "var_es_predictions.csv"
    out.to_csv(pred_path, index=False)

    tail_df = pd.DataFrame(tail_rows)
    tail_path = out_dir / "tables" / "tail_dependence_mc.csv"
    tail_df.to_csv(tail_path, index=False)

    report = []
    report.append(f"# ANNEXE — J8 Asymmetric copulas — {dataset_version}\n")
    report.append("Clayton (lower tail) vs Gumbel (upper tail). Tau inversion (train-only). Fallback to Gauss if tau<=0.\n\n")
    report.append(f"- pair: {a1}/{a2}\n")
    report.append(f"- refit_every: {refit_every}\n")
    report.append(f"- rv_window: {rv_window}\n")
    report.append(f"- n_scenarios: {n_scenarios}\n")
    report.append(f"- alphas: {alphas}\n")
    report.append("Outputs:\n")
    report.append("- var_es_predictions.csv\n")
    report.append("- tables/tail_dependence_mc.csv\n")
    rep_path = out_dir / "report.md"
    rep_path.write_text("".join(report), encoding="utf-8")

    prov = {
        "config_path": str(Path(args.config)),
        "dataset_version": dataset_version,
        "inputs": {"returns_path": str(returns_path)},
        "outputs": {"var_es_predictions.csv": str(pred_path), "tail_dependence_mc.csv": str(tail_path), "report.md": str(rep_path)},
        "hashes": {"var_es_predictions.csv": sha12_file(pred_path), "tail_dependence_mc.csv": sha12_file(tail_path), "report.md": sha12_file(rep_path)},
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    write_yaml(out_dir / "config.resolved.yaml", cfg)

    print(f"[OK] J8 asym copulas done: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())