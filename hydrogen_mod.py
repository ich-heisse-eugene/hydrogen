#!/usr/local/bin/python3.11

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from matplotlib.widgets import Cursor, Slider, Button
from scipy import interpolate
from sys import argv,exit

fontsize = 10
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.serif'] = ['Palatino, Latin Modern Roman']
mpl.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']

maxdist = 25.
maxwidth = 25.

# ---------------------------------------------------------------------------
# Noise model settings
# ---------------------------------------------------------------------------
# What the S/N value (FITS keyword SNR) refers to:
#   'beam'   - continuum S/N per original pixel of EACH polarised spectrum
#              (files _1 and _2 separately)
#   'stokesI'- continuum S/N per original pixel of the combined Stokes I
#              spectrum (then each beam has S/N lower by sqrt(2))
SNR_REFERS_TO = 'beam'
# Photon-noise dominated spectra: sigma(r) = sqrt(r)/SNR, i.e. S/N drops
# inside the line wings. Set to False for a constant noise level
# (read-out or background dominated spectra).
PHOTON_NOISE = True
# Use the S/N estimated from the scatter of the data themselves
# instead of the header value (the estimate is always printed).
USE_EMPIRICAL_SNR = True
# ---------------------------------------------------------------------------

def interpol_spec(w, r, wi, d):
  tck = interpolate.splrep(w, r, s=0)
  ri = interpolate.splev(wi, tck, der=d)
  return ri

def boxcar(r, npt):
  rs = r.copy()
  for i in np.arange((npt-1)/2, len(rs)-(npt-1)/2):
    rs[i] = r[i-(npt/2):i+(npt/2)-1].sum()
  return rs

def noise_arrays(r1, r2, snr_beam, photon=True):
    """Per-pixel 1-sigma errors of V/I and of I from the per-beam S/N.

    sigma(r_k) = sqrt(r_k)/SNR (photon noise) or 1/SNR (constant noise),
    propagated through V/I = (r1-r2)/(r1+r2) and I = (r1+r2)/2.
    These are errors per ORIGINAL pixel.
    """
    if photon:
        s1 = np.sqrt(np.clip(r1, 1e-3, None)) / snr_beam
        s2 = np.sqrt(np.clip(r2, 1e-3, None)) / snr_beam
    else:
        s1 = np.full_like(r1, 1. / snr_beam)
        s2 = np.full_like(r2, 1. / snr_beam)
    rsum = r1 + r2
    sig_v = 2. * np.sqrt((r2 * s1)**2 + (r1 * s2)**2) / rsum**2
    sig_i = 0.5 * np.sqrt(s1**2 + s2**2)
    return sig_v, sig_i

def empirical_snr(wl1, r1, wl2, r2):
    """Per-beam continuum S/N per pixel estimated from the data.

    Uses the point-to-point scatter of z = (V/I)*sqrt(I) on the original
    pixel grid; for photon noise z has a constant sigma = 1/(sqrt(2)*SNR_beam).
    Differencing adjacent pixels removes the (smooth) Zeeman signature.
    A robust (MAD) scatter is used. Underestimates the noise if the
    original pixels are already correlated (rebinned spectra).
    """
    r2o = np.interp(wl1, wl2, r2)
    m = (wl1 >= max(wl1[0], wl2[0])) & (wl1 <= min(wl1[-1], wl2[-1]))
    a, b = r1[m], r2o[m]
    v = (a - b) / (a + b)
    i = 0.5 * (a + b)
    z = v * np.sqrt(np.clip(i, 1e-3, None))
    dz = np.diff(z)
    sig_z = 1.4826 * np.median(np.abs(dz - np.median(dz))) / np.sqrt(2.)
    return 1. / (np.sqrt(2.) * sig_z)

def band_stats(lo, hi, wli, vstockes, istockes, i_der, sig_v, sig_i, oversample):
    """Mean V, I, dI/dlambda in a passband and their 1-sigma errors.

    V = (V/I)*I is averaged directly (not V/I and I separately), because the
    weak-field relation V = -F g lambda^2 B dI/dlambda is linear in V, so
    <V>/<dI/dlambda> is unbiased for any band width.
    Since V = (r1-r2)/2 and I = (r1+r2)/2, the per-pixel error of V equals
    sig_i.
    The interpolated grid is oversampled by `oversample` with respect to the
    original pixels, so the interpolated points are not independent: the
    number of independent measurements is n_interp / oversample.
    The band mean of the spline derivative equals (I(hi)-I(lo))/(hi-lo),
    so its error comes from the noise of I at the two band edges.
    """
    m = (wli >= lo) & (wli <= hi)
    n = m.sum()
    if n < 2:
        return None
    n_eff = max(n / oversample, 1.)
    idx = np.where(m)[0]
    ia, ib = idx[0], idx[-1]
    res = dict(
        V=np.average(vstockes[m] * istockes[m]),
        i=np.average(istockes[m]),
        d=np.average(i_der[m]),
        sV=np.sqrt(np.mean(sig_i[m]**2) / n_eff),
        sd=np.sqrt(sig_i[ia]**2 + sig_i[ib]**2) / (wli[ib] - wli[ia]),
        n_eff=n_eff)
    return res

def measure_field(hlam, cent, width, wli, vstockes, istockes, i_der,
                  sig_v, sig_i, oversample, zlande=1.0):
    """Field from the two passbands and its 1-sigma error (G).

    Returns B (sign convention as in the original code, i.e. -Be_aver),
    sigma_B and the contributions from the V noise and the slope noise,
    plus the single-wing fields with their errors.
    """
    F = 4.67*10**(-13)
    L = band_stats(hlam-cent-width/2., hlam-cent+width/2., wli, vstockes,
                   istockes, i_der, sig_v, sig_i, oversample)
    R = band_stats(hlam+cent-width/2., hlam+cent+width/2., wli, vstockes,
                   istockes, i_der, sig_v, sig_i, oversample)
    if L is None or R is None:
        return None
    K = F * zlande

    # single wings
    def wing(b, wl):
        c = 1. / (K * wl * wl)
        B = c * b['V'] / b['d']
        sB = np.sqrt((c * b['sV'] / b['d'])**2 + (B * b['sd'] / b['d'])**2)
        return -B, sB
    BL, sBL = wing(L, hlam-cent)
    BR, sBR = wing(R, hlam+cent)

    # combined (difference of the two wings)
    V_av = 0.5 * (R['V'] - L['V'])
    d_av = 0.5 * (R['d'] - L['d'])
    sV_av = 0.5 * np.sqrt(L['sV']**2 + R['sV']**2)
    sd_av = 0.5 * np.sqrt(L['sd']**2 + R['sd']**2)
    c = 1. / (K * hlam * hlam)
    B = c * V_av / d_av
    sB_v = abs(c * sV_av / d_av)          # from noise in V
    sB_d = abs(B * sd_av / d_av)          # from noise in the slope dI/dlambda
    sB = np.sqrt(sB_v**2 + sB_d**2)
    return dict(B=-B, sB=sB, sB_v=sB_v, sB_d=sB_d, BL=BL, sBL=sBL,
                BR=BR, sBR=sBR, n_eff=L['n_eff'] + R['n_eff'])

def onpress(event):
    global hlam, cid1, cursor, l1, l2
    if event.button == 1:
        hlam = event.xdata
        print(f"Selected line is centered on {hlam:.4f}")
        fig1.canvas.mpl_disconnect(cid1)
        width = swid.val
        cent = sdist.val
        l1, = ax1.bar(hlam-cent-width/2., height=1.4, width=width, color='#b1d1fc', linewidth=0, alpha=0.3)
        l2, = ax1.bar(hlam+cent-width/2., height=1.4, width=width, color='#b1d1fc', linewidth=0, alpha=0.3)
        plt.draw()
        compfield(event)
    return None


def compfield(val):
    global hlam
    global l1, l2
    global istockes, vstockes, i_der
    global wli
    global sig_v, sig_i, oversample
    width = swid.val
    cent = sdist.val
    l1.set_x(hlam-cent-width/2.)
    l1.set_width(width)
    l2.set_x(hlam+cent-width/2.)
    l2.set_width(width)
    r = measure_field(hlam, cent, width, wli, vstockes, istockes, i_der,
                      sig_v, sig_i, oversample)
    if r is None:
        print("Passband outside the spectrum")
        return None
    print(f"Mean longitudinal magnetic field: average = {r['B']:5.0f} +/- {r['sB']:.0f} G"
          f"  (V noise {r['sB_v']:.0f} G, slope noise {r['sB_d']:.0f} G)"
          f"\tblue wing {r['BL']:5.0f} +/- {r['sBL']:.0f} G"
          f"\tred wing {r['BR']:5.0f} +/- {r['sBR']:.0f} G")
    plt.draw()
    return None

def reset(event):
    swid.reset()
    sdist.reset()
    return None

def setline(event):
    global cid1, fig1
    cursor = Cursor(ax1, useblit=True, color='red', linewidth=1)
    print("Please, mark center of the hydrogen line.")
    cid1 = fig1.canvas.mpl_connect('button_press_event', onpress)
    return None

if __name__ == "__main__":
    if len(argv) < 2:
        exit("Usage: %s <input FITS file>" %argv[0])

    ifile1 = argv[1]
    if ifile1.find('_1.f') != -1:
        ifile2 = ifile1.replace('_1.', '_2.')
    elif ifile1.find('_2.f') != -1:
        ifile2 = ifile1.replace('_2.', '_1.')

    print(f"File 1: {ifile1}, file 2: {ifile2}")

    hdr1 = fits.getheader(ifile1)
    hdr2 = fits.getheader(ifile2)

    npix1 = hdr1['NAXIS1']; npix2 = hdr2['NAXIS1']
    crval1 = hdr1['CRVAL1']; crval2 = hdr2['CRVAL1']
    crpix1 = hdr1['CRPIX1']; crpix2 = hdr2['CRPIX1']
    cdelt1 = hdr1['CDELT1']; cdelt2 = hdr2['CDELT1']

    if 'SNR' in hdr1:
       snr = hdr1['SNR']
    else:
       print("No valid SNR value in the header")
       snr = 200.

    wl1 = crval1 + (np.arange(npix1) - crpix1 + 1) * cdelt1
    wl2 = crval2 + (np.arange(npix2) - crpix2 + 1) * cdelt2
    r1 = fits.getdata(ifile1); r2 = fits.getdata(ifile2)
    # determine
    wli = np.arange(max(wl1[0],wl2[0]), min(wl1[-1],wl2[-1]), cdelt1/3.)
    oversample = max(cdelt1, cdelt2) / (wli[1] - wli[0])
    ri1 = interpol_spec(wl1, r1, wli, 0)
    ri2 = interpol_spec(wl2, r2, wli, 0)
    istockes = (ri1 + ri2) / 2.0
    vstockes = (ri1 - ri2) / (ri1 + ri2)
    i_der = interpol_spec(wli, istockes, wli, 1)
    i_2der = interpol_spec(wli, istockes, wli, 2)

    # noise model
    snr_beam = snr if SNR_REFERS_TO == 'beam' else snr / np.sqrt(2.)
    snr_emp = empirical_snr(wl1, r1, wl2, r2)
    print(f"S/N per pixel per beam: header {snr_beam:.0f}, "
          f"estimated from data {snr_emp:.0f}")
    if USE_EMPIRICAL_SNR:
        snr_beam = snr_emp
    sig_v, sig_i = noise_arrays(ri1, ri2, snr_beam, PHOTON_NOISE)

    fig1 = plt.figure(figsize=(15,5))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.tick_params(axis='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    plt.subplots_adjust(bottom=0.35)
    # ax1.set_title('Measure magnetic field of hydrogen lines')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Residual intensity')
    ax1.plot(wli,ri1, linestyle='-', lw=0.9, marker='.', ms=1.2, color='#fc5a50')
    ax1.plot(wli,ri2, linestyle='-', lw=0.8, marker='.', ms=1.2, color='#069af3')
    ax1.plot(wli, vstockes*5+1.05, linestyle='-', lw=0.9, marker='.', ms=1.2, color='#5cac2d')

    # widgets
    axwid = plt.axes([0.25, 0.1, 0.65, 0.03])
    axdist  = plt.axes([0.25, 0.20, 0.65, 0.03])
    swid = Slider(axwid, 'Width', 0.5, maxwidth, valinit=5)
    sdist = Slider(axdist, 'Distance', 0.5, maxdist, valinit=5) # valinit=5 in Landstreet's setup
    swid.on_changed(compfield)
    sdist.on_changed(compfield)
    setlax = plt.axes([0.25, 0.025, 0.1, 0.04])
    butsel = Button(setlax, 'Set line', color='grey', hovercolor='0.975')
    smeas = plt.axes([0.36, 0.025, 0.1, 0.04])
    butmeas = Button(smeas, 'Measure field', color='grey', hovercolor='0.975')
    butmeas.on_clicked(compfield)
    resetax = plt.axes([0.47, 0.025, 0.1, 0.04])
    butres = Button(resetax, 'Reset', color='grey', hovercolor='0.975')
    butres.on_clicked(reset)
    butsel.on_clicked(setline)

    quitax = plt.axes([0.58, 0.025, 0.1, 0.04])
    butquit = Button(quitax, 'Quit', color='grey', hovercolor='0.975')
    butquit.on_clicked(exit)
    plt.show()

    exit(0)
