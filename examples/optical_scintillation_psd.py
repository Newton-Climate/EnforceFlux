"""PSD of received OPTICAL POWER for a retro-reflected, 2f-WMS open-path CH4 beam.

Instrument modelled: monostatic open path at lambda = 1654 nm (the CH4 line),
beam to a corner-cube RETRO and back, return demodulated by 2f wavelength-
modulation spectroscopy (WMS) with a lock-in. This is the GasFinder-class
sensor geometry. Three things distinguish it from a one-way plane wave, and the
detection chain is what really governs how much turbulence-induced scintillation
reaches the methane readout.

  (1) AMPLITUDE is still C_n^2, read off the LES temperature field:
        C_n^2 = (79e-6 P/T^2)^2 C_T^2 ,  C_T^2: D_T(r)=C_T^2 r^(2/3).
      The coarse 20 m demo gives a LOWER BOUND; the fine grid raises it.

  (2) GEOMETRY is double-pass, spherical, correlated. One-way spherical Rytov
      variance beta0^2 = 0.4 * sigma_R,plane^2. The retro return retraces the
      same eddies, so forward/return log-amplitudes are correlated (rho): the
      double-pass irradiance variance is
        sigma_I,retro^2 = 2 (1 + rho) beta0^2 ,  rho in [0,1],
      rho -> 1 for a small retro with a co-located detector (enhanced back-
      scatter, 4x one-way); rho -> 0 if the legs decorrelate (2x). A finite
      receiver aperture further averages the pattern down by A(D_r).

  (3) DETECTION: 2f WMS + lock-in is a narrow band-pass FAR below the
      scintillation band, so it rejects most scintillation two ways:
        - the lock-in low-pass (time constant tau) keeps only ~1/(4 tau) Hz,
          while scintillation power is spread flat out to f_F ~ 135 Hz;
        - 2f/1f (or 2f/DC) normalisation cancels multiplicative power
          fluctuations to first order, since 2f and 1f both scale with the
          received power. Residual set by --norm-rejection-db.
      The scintillation-induced RELATIVE noise that survives into the
      concentration channel is sqrt( int S_P(f) |H_lockin(f)|^2 df ), before
      and after normalisation.

Usage:
    python examples/optical_scintillation_psd.py
    python examples/optical_scintillation_psd.py --th-field PATH --itot 480 --jtot 240 --spacing-m 4
    python examples/optical_scintillation_psd.py --lockin-tau-ms 20 --norm-rejection-db 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "runs" / "microhh_sacramento_demo"


# ── C_n^2 from the LES temperature field ─────────────────────────────────────
def ct2_from_field(T_beam: np.ndarray, spacing_m: float,
                   seps_cells=(2, 3, 4, 5, 6)) -> float:
    """C_T^2 from a horizontal T slice: fit D_T(r)=C_T^2 r^(2/3) (inertial range)."""
    est = []
    for r in seps_cells:
        diff = T_beam[r:, :] - T_beam[:-r, :]
        est.append(float(np.mean(diff**2)) / (r * spacing_m) ** (2 / 3))
    return float(np.mean(est))


def cn2_optical(ct2: float, T_mean_K: float, pressure_hpa: float = 1013.0) -> float:
    """Optical/near-IR C_n^2 from C_T^2 (humidity term negligible at 1654 nm)."""
    A = 79e-6 * pressure_hpa / T_mean_K**2
    return A**2 * ct2


def read_th_beam(th_path: Path, itot: int, jtot: int, ktot: int, k_beam: int):
    """Return the beam-height T slice + mean; accepts a 3-D field or a 2-D xy cross."""
    arr = np.fromfile(th_path, dtype=np.float64)
    if arr.size == ktot * jtot * itot:
        T = arr.reshape(ktot, jtot, itot)[k_beam]
    elif arr.size == jtot * itot:
        T = arr.reshape(jtot, itot)
    else:
        raise ValueError(f"{th_path.name}: {arr.size} values match neither 3-D "
                         f"({ktot*jtot*itot}) nor 2-D xy ({jtot*itot})")
    return T, float(T.mean())


# ── optics: double-pass retro geometry ───────────────────────────────────────
def sigma_I2_retro(cn2, lambda_m, L_m, rho, aperture_m):
    """Double-pass irradiance variance for a monostatic retro path (weak turb.).

    beta0^2 = 0.4 * 1.23 C_n^2 k^(7/6) L^(11/6)  (one-way spherical Rytov)
    sigma_I,retro^2 = 2(1+rho) beta0^2 * A(aperture)   (correlated double pass).
    """
    k = 2 * np.pi / lambda_m
    beta0_2 = 0.4 * 1.23 * cn2 * k ** (7 / 6) * L_m ** (11 / 6)
    A = aperture_averaging(aperture_m, lambda_m, L_m)
    return 2.0 * (1.0 + rho) * beta0_2 * A, beta0_2, A


def aperture_averaging(D_r, lambda_m, L_m):
    """Aperture-averaging factor A<=1 (Andrews plane-wave approx). D_r=0 -> A=1."""
    if D_r <= 0:
        return 1.0
    return (1.0 + 1.06 * (D_r**2 / (4.0 * lambda_m * L_m))) ** (-7 / 6)


def power_psd(freqs, sigma_I2, f_fresnel):
    """Temporal PSD of relative received power dP/P; integrates (two-sided) to sigma_I^2.

    Weak-fluctuation transverse-wind model: plateau then f^(-8/3), knee at the
    Fresnel frequency. Shape W(f)=W0/(1+(f/f_c)^2)^(4/3), f_c=0.5 f_fresnel.
    """
    f_c = 0.5 * f_fresnel
    shape = 1.0 / (1.0 + (freqs / f_c) ** 2) ** (4 / 3)
    W0 = sigma_I2 / (2.0 * np.trapezoid(shape, freqs))
    return W0 * shape


# ── detection: 2f WMS lock-in low-pass ───────────────────────────────────────
def lockin_lowpass(freqs, tau_s):
    """|H(f)|^2 of a first-order lock-in output filter, -3 dB at 1/(2 pi tau)."""
    return 1.0 / (1.0 + (2 * np.pi * freqs * tau_s) ** 2)


def invert_flicker(sigma_I2_measured, f_fresnel_measured, lambda_m, L_m,
                   rho=1.0, aperture_m=0.0):
    """Retrieve atmospheric turbulence quantities from measured flicker statistics.

    Inverts the forward chain of Steps 2-3:
       Cn^2 = sigma_I,retro^2 / [ 2(1+rho) * 0.4 * 1.23 k^(7/6) L^(11/6) * A ]
       U    = f_fresnel * sqrt(lambda L)

    Also derives the temperature structure parameter C_T^2 (assumes P=1013 hPa,
    T = 288 K) and reports intermediate scales useful for atmospheric-boundary-
    layer inference (Obukhov, mixing length, TKE) — see the walkthrough for the
    reasoning; those onward inversions need auxiliary data (a wind sensor for
    u*, a temperature reading for T), so we just report the pathway.
    """
    k = 2 * np.pi / lambda_m
    A = aperture_averaging(aperture_m, lambda_m, L_m)
    denom = 2.0 * (1.0 + rho) * 0.4 * 1.23 * k ** (7 / 6) * L_m ** (11 / 6) * A
    Cn2 = sigma_I2_measured / denom
    U = f_fresnel_measured * np.sqrt(lambda_m * L_m)

    # C_T^2 back from Cn^2 (assumed near-surface P, T)
    P_hpa, T_K = 1013.0, 288.0
    C_T2 = Cn2 / (79e-6 * P_hpa / T_K**2) ** 2
    return {"Cn2": Cn2, "U_crosswind": U, "C_T2": C_T2, "A": A}


def synthesize_dc_signal(sigma_I2, f_fresnel, fs, duration_s, seed=0):
    """A synthetic DIRECT-DETECTION (DC) received-power series with the given PSD.

    No lock-in, no 2f/1f: the photodetector sees P_DC(t)=<P>(1+chi(t)), where
    chi is the full-band scintillation. Returns (t, s=P/<P>). The exact target
    variance is enforced by rescaling, so FFT normalisation drops out.
    """
    rng = np.random.default_rng(seed)
    N = int(fs * duration_s)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    S1 = 2.0 * power_psd(np.maximum(freqs, freqs[1] / 10), sigma_I2, f_fresnel)  # one-sided
    df = freqs[1] - freqs[0]
    mag = np.sqrt(S1 * df) * N / np.sqrt(2.0)
    spec = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, freqs.size))
    spec[0] = 0.0                                   # zero-mean fluctuation
    chi = np.fft.irfft(spec, n=N)
    chi *= np.sqrt(sigma_I2 / np.var(chi))          # enforce exact rms
    return np.arange(N) / fs, 1.0 + chi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--th-field", type=Path, default=DEMO / "th.0003600")
    ap.add_argument("--itot", type=int, default=192)
    ap.add_argument("--jtot", type=int, default=96)
    ap.add_argument("--ktot", type=int, default=64)
    ap.add_argument("--spacing-m", type=float, default=20.0)
    ap.add_argument("--k-beam", type=int, default=0)
    ap.add_argument("--lambda-nm", type=float, default=1654.0, help="CH4 WMS line")
    ap.add_argument("--path-m", type=float, default=300.0, help="one-way beam length L")
    ap.add_argument("--wind-ms", type=float, default=3.0, help="transverse wind U")
    ap.add_argument("--rho", type=float, default=1.0,
                    help="forward/return correlation (1=enhanced backscatter, small retro)")
    ap.add_argument("--aperture-cm", type=float, default=0.0,
                    help="receiver aperture diameter (0=point, no aperture averaging)")
    ap.add_argument("--fm-khz", type=float, default=9.0,
                    help="WMS modulation frequency (2f detection = 2*fm)")
    ap.add_argument("--lockin-tau-ms", type=float, default=100.0, help="lock-in time constant")
    ap.add_argument("--norm-rejection-db", type=float, default=30.0,
                    help="2f/1f normalisation rejection of power fluctuations")
    ap.add_argument("--dx-fine-m", type=float, default=4.0)
    ap.add_argument("--dc-signal", action="store_true",
                    help="no lock-in: synthesise + plot the direct-detection (DC) power signal")
    ap.add_argument("--fs-hz", type=float, default=1000.0, help="DC synthesis sample rate")
    ap.add_argument("--duration-s", type=float, default=30.0, help="DC synthesis length")
    ap.add_argument("--out", type=Path, default=REPO / "analysis" / "optical_scintillation_psd.png")
    args = ap.parse_args()

    T_beam, T_mean = read_th_beam(args.th_field, args.itot, args.jtot, args.ktot, args.k_beam)
    ct2 = ct2_from_field(T_beam, args.spacing_m)
    cn2 = cn2_optical(ct2, T_mean)

    lam, L, U = args.lambda_nm * 1e-9, args.path_m, args.wind_ms
    ap_m = args.aperture_cm * 1e-2
    sigma_I2, beta0_2, A = sigma_I2_retro(cn2, lam, L, args.rho, ap_m)
    f_fresnel = U / np.sqrt(lam * L)
    regime = "weak (Rytov valid)" if sigma_I2 < 1 else "moderate/strong"

    # extend to 100 kHz so the 18 kHz WMS carrier is on-plot
    f = np.logspace(-2, 5, 4000)
    S_raw = power_psd(f, sigma_I2, f_fresnel)                 # received-power PSD
    tau = args.lockin_tau_ms * 1e-3
    H2 = lockin_lowpass(f, tau)
    S_lockin = S_raw * H2                                     # survives lock-in, un-normalised
    rej = 10 ** (-args.norm_rejection_db / 10)               # 2f/1f power-rejection
    S_norm = S_lockin * rej                                  # after 2f/1f normalisation

    var_in = 2.0 * np.trapezoid(S_lockin, f)                 # two-sided in-band variance
    var_norm = 2.0 * np.trapezoid(S_norm, f)
    enbw = 1.0 / (4.0 * tau)                                 # first-order lock-in noise BW

    # WMS carrier: 2f detection at 2 * fm. Scintillation power at that
    # frequency is what aliases into the demodulated output.
    f_2f = 2.0 * args.fm_khz * 1e3
    S_at_carrier = float(np.interp(f_2f, f, S_raw))
    plateau = float(S_raw[0])
    carrier_below_dc_db = 10.0 * np.log10(plateau / S_at_carrier)

    print(f"beam-height mean T   : {T_mean:.2f} K")
    print(f"C_T^2 / C_n^2        : {ct2:.3e} K^2 m^-2/3  (LOWER BOUND at {args.spacing_m:g} m)"
          f"  ->  C_n^2={cn2:.3e} m^-2/3")
    print(f"one-way spherical    : beta0^2 = {beta0_2:.3e}")
    print(f"double-pass retro    : sigma_I^2 = {sigma_I2:.3e}  (rho={args.rho}, A={A:.3f})  [{regime}]")
    print(f"Fresnel frequency    : {f_fresnel:.1f} Hz    lock-in ENBW: {enbw:.2f} Hz (tau={args.lockin_tau_ms:g} ms)")
    print(f"WMS: fm={args.fm_khz:g} kHz  ->  2f detection at {f_2f/1e3:g} kHz")
    print(f"   scint. PSD at 2f  : {S_at_carrier:.3e} Hz^-1   "
          f"({carrier_below_dc_db:.1f} dB below the DC plateau {plateau:.3e})")
    print(f"relative power noise reaching readout:")
    print(f"   un-normalised 2f  : {np.sqrt(var_in):.3e}  ({np.sqrt(var_in)*100:.3f} %)")
    print(f"   2f/1f normalised  : {np.sqrt(var_norm):.3e}  ({np.sqrt(var_norm)*100:.4f} %)  "
          f"[{args.norm_rejection_db:g} dB rejection]")

    # ── closed-loop retrieval: pretend sigma_I^2 and f_F were MEASURED ────────
    ret = invert_flicker(sigma_I2, f_fresnel, lam, L, args.rho, ap_m)
    print("\n── TURBULENCE RETRIEVAL FROM FLICKER (scintillometer mode) ──")
    print(f"given measured sigma_I^2={sigma_I2:.3e} and f_F={f_fresnel:.1f} Hz:")
    print(f"   Cn^2 (retrieved)     : {ret['Cn2']:.3e} m^-2/3   "
          f"(input Cn^2={cn2:.3e}; matches by construction)")
    print(f"   C_T^2 (retrieved)    : {ret['C_T2']:.3e} K^2 m^-2/3")
    print(f"   U crosswind (Fresnel): {ret['U_crosswind']:.2f} m/s   "
          f"(input U={U:g} m/s)")
    print("Onward inferences (need auxiliary data — see walkthrough):")
    print("   * Sensible heat flux H via MOST — Cn^2 + surface layer scaling")
    print("   * Momentum flux / u* — needs a sonic anemometer or wind profile")
    print("   * Mixing / integral length scale l_m ~ U/f_peak from DC-signal PSD")
    print("   * TKE via convective scaling w* = (g/T * H * z_i)^(1/3)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    # ── no-lock-in DC (direct-detection) signal ──────────────────────────────
    if args.dc_signal:
        W0 = float(S_raw[0])                         # S_P(f->0): plateau value
        t, s = synthesize_dc_signal(sigma_I2, f_fresnel, args.fs_hz, args.duration_s)
        sigma = float(np.std(s))
        print("\n── DIRECT DETECTION, NO LOCK-IN (DC channel) ──")
        print(f"normalised DC level <P/<P>>  : 1.000")
        print(f"full-band rms fluctuation    : {sigma*100:.2f} %   (sigma_I = sqrt(sigma_I^2))")
        print(f"peak-to-peak (~+/-3 sigma)   : {6*sigma*100:.1f} %")
        print(f"PSD at DC  S_P(f->0)=W0       : {W0:.3e} Hz^-1   (white plateau to ~{f_fresnel:.0f} Hz)")

        fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.6, 6.6),
                                     gridspec_kw={"height_ratios": [1.1, 1]})
        show = t <= min(5.0, args.duration_s)                # first 5 s for legibility
        a0.plot(t[show], s[show], color="#1f4e79", lw=0.8)
        a0.axhline(1.0, color="#c0392b", lw=1.0, ls="--", label=r"mean (DC) = $\langle P\rangle$")
        a0.axhspan(1 - sigma, 1 + sigma, color="#e08a00", alpha=0.18, label=r"$\pm\sigma_I$")
        a0.set_xlabel("time  [s]"); a0.set_ylabel(r"$P_{DC}(t)/\langle P\rangle$")
        a0.set_title(f"Direct-detection DC signal (no lock-in) — $\\sigma_I$={sigma*100:.1f}% rms, "
                     f"$C_n^2$={cn2:.2e} m$^{{-2/3}}$", fontsize=10)
        a0.legend(loc="upper right", fontsize=8); a0.grid(True, alpha=0.25)

        fr = np.fft.rfftfreq(len(t), 1 / args.fs_hz)[1:]
        # two-sided density, to match S_raw's convention (integrates 2x to sigma_I^2)
        Pxx = (np.abs(np.fft.rfft(s - s.mean())) ** 2)[1:] / (args.fs_hz * len(t))
        a1.loglog(fr, Pxx, color="#8aa4c8", lw=0.6, alpha=0.7, label="synthesised periodogram")
        a1.loglog(f, S_raw, color="#1f4e79", lw=2.0, label="target DC PSD (no filter)")
        a1.axvline(f_fresnel, color="#c0392b", lw=1.0, ls=":")
        a1.text(f_fresnel * 1.05, S_raw.max() * 0.3, f"Fresnel ≈ {f_fresnel:.0f} Hz",
                color="#c0392b", fontsize=8, rotation=90, va="top")
        a1.set_xlabel("frequency  [Hz]")
        a1.set_ylabel(r"$S_{P/\langle P\rangle}(f)$  [Hz$^{-1}$]")
        a1.set_ylim(S_raw.max() * 1e-6, S_raw.max() * 5)
        a1.legend(loc="lower left", fontsize=8); a1.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        out = args.out.with_name("dc_received_power_signal.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        print(f"\nwrote {out}")
        return

    f_les = U / (2 * args.dx_fine_m)
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.loglog(f, S_raw, color="#1f4e79", lw=2.3, label="received-power PSD (double-pass retro)")
    ax.loglog(f, S_lockin, color="#e08a00", lw=2.0, label=f"after lock-in ($\\tau$={args.lockin_tau_ms:g} ms)")
    ax.loglog(f, S_norm, color="#2e7d32", lw=2.0, ls="-",
              label=f"after 2f/1f norm ({args.norm_rejection_db:g} dB)")

    # f^-8/3 guide, extended to plot edge
    W_at = float(np.interp(f_fresnel, f, S_raw))
    fg = np.array([f_fresnel, f.max()])
    ax.loglog(fg, W_at * (fg / f_fresnel) ** (-8 / 3), "--", color="#aaaaaa", lw=1.2,
              label=r"$f^{-8/3}$ inertial roll-off")

    # WMS carrier marker (2f at 2*fm)
    ax.plot([f_2f], [S_at_carrier], marker="o", ms=9, mfc="#ffde59",
            mec="#7a5b00", mew=1.5, zorder=5, label=f"WMS 2f carrier @ {f_2f/1e3:g} kHz")

    ax.axvspan(f.min(), enbw, color="#2e7d32", alpha=0.08)
    for x, c, lab in [(enbw, "#2e7d32", f"lock-in ENBW ≈ {enbw:.1f} Hz"),
                      (f_les, "#8e24aa", f"fine-LES cutoff ≈ {f_les:.1f} Hz"),
                      (f_fresnel, "#c0392b", f"Fresnel ≈ {f_fresnel:.0f} Hz"),
                      (f_2f, "#7a5b00", f"2f detection @ {f_2f/1e3:g} kHz")]:
        ax.axvline(x, color=c, lw=1.1, ls=":")
        ax.text(x * 1.06, S_raw.max() * (0.5 if x == f_fresnel else 2e-3),
                lab, color=c, fontsize=8, rotation=90, va="top")

    ax.set_xlabel("frequency  [Hz]")
    ax.set_ylabel(r"PSD of relative received power  $S_{P/\langle P\rangle}(f)$  [Hz$^{-1}$]")
    ax.set_title(
        f"Retro / 2f-WMS scintillation, $\\lambda$=1654 nm, fm={args.fm_khz:g} kHz — "
        f"LES $C_n^2$={cn2:.2e} m$^{{-2/3}}$\n"
        f"$\\sigma_I^2$(retro)={sigma_I2:.2e}; readout noise "
        f"{np.sqrt(var_in)*100:.2f}%→{np.sqrt(var_norm)*100:.3f}% (raw→2f/1f); "
        f"scint. at 2f is {carrier_below_dc_db:.0f} dB below DC",
        fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(S_raw.max() * 1e-11, S_raw.max() * 3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
