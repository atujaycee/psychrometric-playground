import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# =========================
# Constants (global)
# =========================
C8 = -5.8002206e3
C9 = 1.3914993
C10 = -4.8640239e-2
C11 = 4.1764768e-5
C12 = -1.4452093e-8
C13 = 6.5459673

P_atm = 101325.0  # Pa

# Wet-bulb correlation constants
R = 22105649.25
AA = -27405.526
BB = 97.5413
CC = -0.146244
DD = 0.12558e-3
EE = -0.48502e-7
FF = 4.34903
GG = 0.39381e-2


# =========================
# Saturation pressure helper
# =========================
def psat_pa_from_Tc(T_c: float) -> float:
    """Saturation pressure (Pa) at temperature T (°C) using your correlation style."""
    T_k = T_c + 273.16
    zz = (C8 / (T_k + 1e-5)) + C9 + (C10 * T_k) + (C11 * T_k**2) + (C12 * T_k**3) + (C13 * np.log(T_k + 1e-5))
    return float(np.exp(zz))


# =========================
# Psychrometric direct relations
# =========================
def pv_pa_from_W(W: float) -> float:
    """Vapour partial pressure (Pa) from humidity ratio W at fixed P_atm."""
    return float((W * P_atm) / (0.62198 + W))

def W_from_pv(pv: float) -> float:
    """Humidity ratio W from vapour partial pressure pv (Pa)."""
    return float(0.62198 * pv / (P_atm - pv))

def W_from_Tdpt(Tdpt_C: float) -> float:
    """Humidity ratio W from dew point (°C) at fixed pressure."""
    pv = psat_pa_from_Tc(Tdpt_C)  # at dew point, pv = psat(Tdpt)
    return W_from_pv(pv)

def RH_from_T_W(T_C: float, W: float) -> float:
    """Relative humidity (0–1) from (T, W)."""
    pv = pv_pa_from_W(W)
    psat = psat_pa_from_Tc(T_C)
    return float(pv / psat)

def RHchart(T_a: float, RH: float) -> float:
    """Humidity ratio W (kg/kg) from dry-bulb temperature T_a (°C) and RH (0–1)."""
    P_sat = psat_pa_from_Tc(T_a)
    Pv = P_sat * RH
    return float(0.62198 * Pv / (P_atm - Pv))


# =========================
# Wet-bulb (fsolve)
# =========================
def wetbulb_residual(T_wb_K, Tain_C, Xain_W):
    """
    Residual function for wet-bulb root solve.
    T_wb_K is in Kelvin
    Tain_C in °C, Xain_W is humidity ratio W (kg/kg)
    """
    T_k = Tain_C + 273.16  # dry bulb in K

    P = R * np.exp((AA + BB * T_wb_K + CC * T_wb_K**2 + DD * T_wb_K**3 + EE * T_wb_K**4) / (FF * T_wb_K - GG * T_wb_K**2))
    P_v = (Xain_W * P_atm) / (0.62198 + Xain_W)

    h_fg = 2502535.259 - 2385.76424 * (T_wb_K - 255.38)
    B = (1006.9254 * (P - P_atm) * (1 + 0.1577 * P_v / P_atm)) / (0.62194 * h_fg)

    return P - P_v - B * (T_wb_K - T_k)

def calculate_wetbulb(Tain_C: float, W: float) -> float:
    """Returns wet-bulb temperature in °C."""
    T_wb_guess_K = (Tain_C + 273.16) - 5.0
    T_wb_solution_K = fsolve(wetbulb_residual, T_wb_guess_K, args=(Tain_C, W))[0]
    return float(T_wb_solution_K - 273.16)


# =========================
# Dew-point (fsolve)
# =========================
def dewpt_residual(T_d_C, T_a_C, W):
    """Residual for dew point solve: Pv - Psat(Td) = 0"""
    T_j = T_d_C + 273.16
    yy = (C8 / T_j) + C9 + (C10 * T_j) + (C11 * T_j**2) + (C12 * T_j**3) + (C13 * np.log(T_j))
    Psat_Td = np.exp(yy)
    Pv = pv_pa_from_W(W)
    return Pv - Psat_Td

def calculate_dewpoint(T_a_C: float, W: float) -> float:
    """Returns dew-point temperature in °C."""
    T_d_guess = T_a_C - 5.0
    T_d_solution = fsolve(dewpt_residual, T_d_guess, args=(T_a_C, W))[0]
    if T_d_solution > T_a_C:
        raise ValueError("Air humidity ratio too high: Dew point exceeds dry bulb temperature.")
    return float(T_d_solution)


# =========================
# Enthalpy and specific volume
# =========================
def ENchart(T_a_C: float, W: float) -> float:
    """Enthalpy (kJ/kg dry air) from T (°C) and W (kg/kg)."""
    Cpa = 1.006
    Cpw = 1.86
    Hv0 = 2501.0
    return float(((Cpa + (W * Cpw)) * T_a_C) + (W * Hv0))

def SPVchart(T_a_C: float, W: float) -> float:
    """Specific volume (m^3/kg dry air) from T (°C) and W (kg/kg)."""
    Rda = 286.9
    T_k = T_a_C + 273.16
    return float(((W + 0.62198) * (Rda * T_k)) / (0.62198 * P_atm))


# =========================
# Canonical state builder
# =========================
def state_from_T_W(T_C: float, W: float) -> dict:
    RH = RH_from_T_W(T_C, W)
    Twb = calculate_wetbulb(T_C, W)
    Tdpt = calculate_dewpoint(T_C, W)
    h = ENchart(T_C, W)
    v = SPVchart(T_C, W)

    Pv = pv_pa_from_W(W)
    Psat = psat_pa_from_Tc(T_C)

    return {
        "T_C": float(T_C),
        "W": float(W),
        "RH": float(RH),
        "Twb_C": float(Twb),
        "Tdpt_C": float(Tdpt),
        "h_kJkg": float(h),
        "v_m3kg": float(v),
        "Pv_Pa": float(Pv),
        "Psat_Pa": float(Psat),
    }


# =========================
# 12-pair solver
# =========================
def psychro_state(given: dict, tol_W: float = 1e-6) -> dict:
    """
    given: dict with exactly two keys from:
      "T_C", "W", "Twb_C", "Tdpt_C", "RH", "h_kJkg"
    """
    if len(given) != 2:
        raise ValueError("Provide exactly two properties from: T_C, W, Twb_C, Tdpt_C, RH, h_kJkg")

    keys = set(given.keys())

    if keys == {"T_C", "RH"}:
        T = float(given["T_C"]); RH = float(given["RH"])
        W = RHchart(T, RH)
        return state_from_T_W(T, W)

    if keys == {"T_C", "W"}:
        return state_from_T_W(float(given["T_C"]), float(given["W"]))

    if keys == {"T_C", "Tdpt_C"}:
        T = float(given["T_C"]); Tdpt = float(given["Tdpt_C"])
        W = W_from_Tdpt(Tdpt)
        return state_from_T_W(T, W)

    if keys == {"T_C", "Twb_C"}:
        T = float(given["T_C"]); Twb_target = float(given["Twb_C"])

        def res(W):
            return calculate_wetbulb(T, float(W)) - Twb_target

        W0 = max(1e-6, RHchart(T, 0.5))
        W = fsolve(res, W0)[0]
        return state_from_T_W(T, float(W))

    if keys == {"T_C", "h_kJkg"}:
        T = float(given["T_C"]); h_target = float(given["h_kJkg"])

        def res(W):
            return ENchart(T, float(W)) - h_target

        W0 = max(1e-6, RHchart(T, 0.5))
        W = fsolve(res, W0)[0]
        return state_from_T_W(T, float(W))

    if keys == {"W", "h_kJkg"}:
        W = float(given["W"]); h_target = float(given["h_kJkg"])

        def res(T):
            return ENchart(float(T), W) - h_target

        T = fsolve(res, 25.0)[0]
        return state_from_T_W(float(T), W)

    if keys == {"RH", "Tdpt_C"}:
        RH = float(given["RH"]); Tdpt = float(given["Tdpt_C"])
        pv = psat_pa_from_Tc(Tdpt)
        W = W_from_pv(pv)

        def res(T):
            return RH - (pv / psat_pa_from_Tc(float(T)))

        T = fsolve(res, Tdpt + 5.0)[0]
        return state_from_T_W(float(T), W)

    if keys == {"RH", "Twb_C"}:
        RH_target = float(given["RH"]); Twb_target = float(given["Twb_C"])

        def F(x):
            T, W = float(x[0]), float(x[1])
            return [
                RH_from_T_W(T, W) - RH_target,
                calculate_wetbulb(T, W) - Twb_target
            ]

        x0 = [25.0, RHchart(25.0, max(1e-3, min(0.999, RH_target)))]
        T, W = fsolve(F, x0)
        return state_from_T_W(float(T), float(W))

    if keys == {"W", "Tdpt_C"}:
        W_given = float(given["W"]); Tdpt = float(given["Tdpt_C"])
        W_implied = W_from_Tdpt(Tdpt)
        if abs(W_given - W_implied) > tol_W:
            raise ValueError(f"Inconsistent inputs: Tdpt implies W≈{W_implied:.6g}, but you gave W={W_given:.6g}.")
        return state_from_T_W(Tdpt, W_given)

    if keys == {"h_kJkg", "Tdpt_C"}:
        h_target = float(given["h_kJkg"]); Tdpt = float(given["Tdpt_C"])
        W = W_from_Tdpt(Tdpt)

        def res(T):
            return ENchart(float(T), W) - h_target

        T = fsolve(res, Tdpt + 10.0)[0]
        return state_from_T_W(float(T), W)

    if keys == {"h_kJkg", "RH"}:
        h_target = float(given["h_kJkg"]); RH_target = float(given["RH"])

        def F(x):
            T, W = float(x[0]), float(x[1])
            return [
                ENchart(T, W) - h_target,
                RH_from_T_W(T, W) - RH_target
            ]

        x0 = [25.0, RHchart(25.0, max(1e-3, min(0.999, RH_target)))]
        T, W = fsolve(F, x0)
        return state_from_T_W(float(T), float(W))

    if keys == {"h_kJkg", "Twb_C"}:
        h_target = float(given["h_kJkg"]); Twb_target = float(given["Twb_C"])

        def F(x):
            T, W = float(x[0]), float(x[1])
            return [
                ENchart(T, W) - h_target,
                calculate_wetbulb(T, W) - Twb_target
            ]

        x0 = [25.0, RHchart(25.0, 0.5)]
        T, W = fsolve(F, x0)
        return state_from_T_W(float(T), float(W))

    raise ValueError(f"Pair {keys} not implemented in the 12-case solver.")

def load_twb_library(csv_filename="twb_precomputed.csv") -> "pd.DataFrame":
    """Load precomputed Twb isolines from CSV stored alongside this module."""
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    return pd.read_csv(csv_path)


# =========================
# Plotting
# =========================
def plot_psychrometric_chart(
    states=None,
    T_min=-10, T_max=80,
    W_min=0.0, W_max=0.05,  # kg/kg (internal)
    RH_lines=(0.2, 0.4, 0.6, 0.8, 1.0),
    h_lines=(20, 40, 60, 80, 100, 120),
    Twb_lines=tuple(range(10, 41, 10)),
    v_lines=(0.80, 0.85, 0.90, 0.95),
    label_v=True,
    label_v_offset_px=(6, 6),

    nT=800,
    nT_twb=250,
    figsize=(12, 7),
    dpi=130,
    font_base=12,
    label_RH=True,
    label_h=True,
    label_Twb=True,
    label_spread=(0.20, 0.85),
    label_offset_px=(6, 6),
    label_h_offset_px=(6, -10),
):
    """
    Psychrometric chart with:
    - x-axis: T (°C)
    - y-axis: W (g/kg dry air)  [internally W is kg/kg]
    - RH curves (angled labels)
    - Enthalpy lines (left-side labels)
    - Wet-bulb lines (right-side labels + aesthetic extension to y-axis)
    """

    plt.rcParams.update({
        "font.size": font_base,
        "axes.titlesize": font_base + 2,
        "axes.labelsize": font_base + 1,
        "xtick.labelsize": font_base,
        "ytick.labelsize": font_base,
    })

    T = np.linspace(T_min, T_max, nT)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title("Psychrometric Chart (W vs T)")
    ax.set_xlabel("Dry-bulb temperature, T (°C)")
    ax.set_ylabel("Humidity ratio, W (g/kg dry air)")

    W_min_g = W_min * 1000.0
    W_max_g = W_max * 1000.0
    ax.set_xlim(T_min, T_max)
    ax.set_ylim(W_min_g, W_max_g)

    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.2)
    ax.minorticks_on()

    def _angle_on_screen(ax, x1, y1, x2, y2):
        p1 = ax.transData.transform((x1, y1))
        p2 = ax.transData.transform((x2, y2))
        dx, dy = (p2 - p1)
        return np.degrees(np.arctan2(dy, dx))

    # Spread RH labels across x-axis
    if label_RH and len(RH_lines) > 1:
        fracs = np.linspace(label_spread[0], label_spread[1], len(RH_lines))
    else:
        fracs = np.array([0.6] * len(RH_lines))

    # ---- RH curves ----
    for i, RH in enumerate(RH_lines):
        W_curve = np.array([RHchart(t, RH) for t in T])     # kg/kg
        W_curve_g = W_curve * 1000.0                        # g/kg

        mask = np.isfinite(W_curve_g) & (W_curve_g >= W_min_g) & (W_curve_g <= W_max_g)
        ax.plot(T[mask], W_curve_g[mask], linewidth=(2.2 if abs(RH - 1.0) < 1e-12 else 1.4))

        if label_RH:
            # planned label position
            t_lab = T_min + fracs[i] * (T_max - T_min)
            w_lab_g = RHchart(t_lab, RH) * 1000.0

            # fallback to visible segment
            if not (np.isfinite(w_lab_g) and (W_min_g <= w_lab_g <= W_max_g)):
                valid_idx = np.where(mask)[0]
                if len(valid_idx) == 0:
                    continue
                j = valid_idx[int(0.75 * (len(valid_idx) - 1))]
                t_lab = T[j]
                w_lab_g = W_curve_g[j]

            dt = 0.5
            w1_g = RHchart(t_lab - dt, RH) * 1000.0
            w2_g = RHchart(t_lab + dt, RH) * 1000.0
            angle = _angle_on_screen(ax, t_lab - dt, w1_g, t_lab + dt, w2_g) if (np.isfinite(w1_g) and np.isfinite(w2_g)) else 0.0

            ax.annotate(
                f"RH={int(RH*100)}%",
                xy=(t_lab, w_lab_g),
                xytext=label_offset_px,
                textcoords="offset pixels",
                rotation=angle,
                rotation_mode="anchor",
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
            )

    # ---- Enthalpy lines ----
    Cpa, Cpw, Hv0 = 1.006, 1.86, 2501.0
    for h in h_lines:
        denom = (Hv0 + Cpw * T)
        W_h = (h - Cpa * T) / denom         # kg/kg
        W_h_g = W_h * 1000.0                # g/kg

        mask_h = np.isfinite(W_h_g) & (W_h_g >= W_min_g) & (W_h_g <= W_max_g)
        if np.any(mask_h):
            ax.plot(T[mask_h], W_h_g[mask_h], linestyle="--", linewidth=1.0)

            if label_h:
                idx = np.where(mask_h)[0]
                j = idx[int(0.15 * (len(idx) - 1))]  # left-side label
                t_lab = T[j]
                w_lab_g = W_h_g[j]

                dt = 0.5
                def W_from_h_T(hh, TT):
                    return ((hh - Cpa * TT) / (Hv0 + Cpw * TT)) * 1000.0

                w1_g = W_from_h_T(h, t_lab - dt)
                w2_g = W_from_h_T(h, t_lab + dt)
                angle = _angle_on_screen(ax, t_lab - dt, w1_g, t_lab + dt, w2_g) if (np.isfinite(w1_g) and np.isfinite(w2_g)) else 0.0

                ax.annotate(
                    f"h={h}",
                    xy=(t_lab, w_lab_g),
                    xytext=label_h_offset_px,
                    textcoords="offset pixels",
                    rotation=angle,
                    rotation_mode="anchor",
                    va="center",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.65),
                )


        # ---- Specific volume lines (optional) ----
    # v = ((W + 0.62198) * (Rda * T_k)) / (0.62198 * P_atm)
    # Solve for W at constant v:
    # W = (v * 0.62198 * P_atm) / (Rda * T_k) - 0.62198

    Rda = 286.9
    if v_lines:
        T_k = T + 273.16

        for v in v_lines:
            W_v = (v * 0.62198 * P_atm) / (Rda * T_k) - 0.62198   # kg/kg
            W_v_g = W_v * 1000.0                                  # g/kg

            mask_v = np.isfinite(W_v_g) & (W_v_g >= W_min_g) & (W_v_g <= W_max_g)

            if np.any(mask_v):
                ax.plot(T[mask_v], W_v_g[mask_v], linestyle="-.", linewidth=1.0, alpha=0.55)
                
                if label_v:
                    # Fixed label anchor temperatures (°C) to avoid RH labels
                    v_label_T = {
                        0.80: 6.0,
                        0.85: 14.0,
                        0.90: 22.0,
                        0.95: 30.0,
                    }

                    if v in v_label_T:
                        t_lab = v_label_T[v]
                        Tk_lab = t_lab + 273.16
                        W_lab = (v * 0.62198 * P_atm) / (Rda * Tk_lab) - 0.62198
                        w_lab_g = W_lab * 1000.0

                        if (
                            np.isfinite(w_lab_g)
                            and W_min_g <= w_lab_g <= W_max_g
                            and T_min <= t_lab <= T_max
                        ):
                            dt = 0.5

                            def W_from_v_T(vv, TT):
                                Tk = TT + 273.16
                                return ((vv * 0.62198 * P_atm) / (Rda * Tk) - 0.62198) * 1000.0

                            w1_g = W_from_v_T(v, t_lab - dt)
                            w2_g = W_from_v_T(v, t_lab + dt)

                            angle = (
                                _angle_on_screen(ax, t_lab - dt, w1_g, t_lab + dt, w2_g)
                                if np.isfinite(w1_g) and np.isfinite(w2_g)
                                else 0.0
                            )

                            ax.annotate(
                                f"v={v:.2f}",
                                xy=(t_lab, w_lab_g),
                                xytext=label_v_offset_px,
                                textcoords="offset pixels",
                                rotation=angle,
                                rotation_mode="anchor",
                                va="center",
                                ha="left",
                                bbox=dict(
                                    boxstyle="round,pad=0.16",
                                    fc="white",
                                    ec="none",
                                    alpha=0.65,
                                ),
                            )

                
                
    # ---- Wet-bulb lines (right-side labels + aesthetic extension) ----

    
    # ==========================================================
    # Wet-bulb lines (from precomputed CSV)
    # ==========================================================
    if Twb_lines:
        try:
            df_twb = load_twb_library()  # expects columns: Twb_C, Tdb_C, W
        except Exception:
            df_twb = None

        if df_twb is not None:
            for Twb_target in Twb_lines:
                seg = df_twb[df_twb["Twb_C"] == float(Twb_target)]
                if seg.empty:
                    continue

                Tseg = seg["Tdb_C"].to_numpy()
                Wseg_g = seg["W"].to_numpy() * 1000.0  # kg/kg → g/kg

                mask = (
                    np.isfinite(Wseg_g)
                    & (Wseg_g >= W_min_g) & (Wseg_g <= W_max_g)
                    & (Tseg >= T_min) & (Tseg <= T_max)
                )
                if not np.any(mask):
                    continue

                ax.plot(
                    Tseg[mask],
                    Wseg_g[mask],
                    linestyle=":",
                    linewidth=1.2
                )

                # ---- right-side angled label ----
                if label_Twb:
                    idx = np.where(mask)[0]
                    jlab = idx[int(0.40 * (len(idx) - 1))]
                    t_lab = Tseg[jlab]
                    w_lab_g = Wseg_g[jlab]

                    j1 = max(jlab - 1, idx[0])
                    j2 = min(jlab + 1, idx[-1])

                    angle = _angle_on_screen(
                        ax,
                        Tseg[j1], Wseg_g[j1],
                        Tseg[j2], Wseg_g[j2]
                    )

                    ax.annotate(
                        f"Twb={int(Twb_target)}°C",
                        xy=(t_lab, w_lab_g),
                        xytext=(10, -10),
                        textcoords="offset pixels",
                        rotation=angle,
                        rotation_mode="anchor",
                        va="center",
                        ha="left",
                        bbox=dict(
                            boxstyle="round,pad=0.16",
                            fc="white",
                            ec="none",
                            alpha=0.65
                        ),
                    )

    # ---- plot state points ----
    if states is not None:
        if isinstance(states, dict):
            states = [states]
        for k, s in enumerate(states, start=1):
            ax.scatter(s["T_C"], s["W"] * 1000.0, s=70)
            ax.annotate(
                f"P{k}",
                (s["T_C"], s["W"] * 1000.0),
                textcoords="offset pixels",
                xytext=(8, 8),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            )

    return fig
