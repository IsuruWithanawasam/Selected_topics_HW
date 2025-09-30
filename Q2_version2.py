# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 21:33:52 2025

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

# Given parameters
tau_input = 50e-15  # 50 fs FWHM
lmd = 800e-9  # 800 nm central wavelength
omega_0 = 2 * pi * c / lmd  # central angular frequency
I_0 = 1e12 * 1e4  # 1e12 W/cm^2 = 1e16 W/m^2
L = 1  # 1 m propagation length
n2 = 0.85e-19 * 1e-4  # 0.85e-19 cm^2/W = 0.85e-23 m^2/W

# Time array
t = np.linspace(-100000e-15, 100000e-15, 2000000)

# Input Gaussian pulse intensity
I = I_0 * np.exp(-4 * np.log(2) * (t / tau_input)**2)

# Plot input pulse
plt.figure(figsize=(10, 6))
plt.plot(t * 1e15, I / 1e16, 'b-', linewidth=2)
plt.xlabel('Time (fs)', fontsize=12)
plt.ylabel('Intensity (10$^{12}$ W/cm$^2$)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-100, 100)
plt.tight_layout()
plt.show()

# SPM phase and instantaneous frequency
phi_NL = (omega_0 * n2 * L / c) * I
omega_inst = omega_0 - np.gradient(phi_NL, t)

# Convert to THz
freq_0 = omega_0 / (2 * pi) * 1e-12  # THz
freq_inst = omega_inst / (2 * pi) * 1e-12  # THz

print("="*60)
print("INSTANTANEOUS FREQUENCY ANALYSIS")
print("="*60)
print(f"Central frequency: {freq_0:.4f} THz")
print(f"Instantaneous frequency range: {freq_inst.min():.4f} - {freq_inst.max():.4f} THz")
print(f"Frequency shift: Δf = {freq_inst.max() - freq_inst.min():.4f} THz")


phi_max = np.max(np.abs(phi_NL))
N_peaks = int(2 * phi_max / pi) + 1

print('Number of Oscillation peaks', N_peaks)


# Plot instantaneous frequency
plt.plot(t * 1e15, freq_inst, 'b-', linewidth=2, label='Instantaneous frequency')
plt.axhline(freq_0, color='r', linestyle='--', linewidth=2, label='Central frequency')
plt.xlabel('Time (fs)', fontsize=12)
plt.ylabel('Frequency (THz)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(-100, 100)
plt.show()

E_broadened = np.sqrt(I / I_0) * np.exp(-1j * phi_NL)
I_broadened_envelope = np.abs(E_broadened)**2 * I_0

I_broadened_norm = I_broadened_envelope / np.max(I_broadened_envelope)
indices_half_max = np.where(I_broadened_norm >= 0.5)[0]
if len(indices_half_max) > 0:
    tau_broadened = (t[indices_half_max[-1]] - t[indices_half_max[0]])
else:
    tau_broadened = tau_input


plt.plot(t * 1e15, I_broadened_envelope / 1e16, 'b-', linewidth=2,label='Broadened pulse')
plt.plot(t * 1e15, I / 1e16, 'r--', linewidth=2,label='Input Pulse pulse')
plt.xlabel('Time (fs)', fontsize=12)
plt.ylabel('Intensity (10$^{12}$ W/cm$^2$)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-100, 100)
plt.legend()
plt.show()   

print('FWHM of Broadened pulse ', tau_broadened*1e15) 



def fupulse(t,E_val):
    dt = t[1] - t[0]
    # FFT of the envelope
    freq_fft = np.fft.fftfreq(len(t), dt)

    E_spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_val)))
    freq_fft_shifted = np.fft.fftshift(freq_fft)
    spectrum_power = np.abs(E_spectrum)**2

    # Convert to actual frequency (add back carrier frequency)
    freq_plot = freq_0 + freq_fft_shifted * 1e-12  # Convert to THz

    # Normalize for plotting
    spectrum_sorted = spectrum_power
    spectrum_sorted = spectrum_sorted / np.max(spectrum_sorted)

    half_max = np.max(spectrum_sorted) / 2.0
    indices = np.where(spectrum_sorted >= half_max)[0]
    fwhm_val = freq_plot[indices[-1]] - freq_plot[indices[0]]

    return freq_plot,spectrum_sorted,fwhm_val


freq_plot_br,spectrum_sorted_br,fwhm_val_br=fupulse(t, E_broadened)
freq_plot_or,spectrum_sorted_or,fwhm_val_or=fupulse(t, np.sqrt(I / I_0))

print('spectrum width of broadened pulse ',fwhm_val_br )
print('spectrum width of original pulse ',fwhm_val_or )

plt.plot(freq_plot_br, (fwhm_val_or/fwhm_val_br)*spectrum_sorted_br, 'b-', linewidth=1.5,label='Broadened pulse')
plt.plot(freq_plot_or, spectrum_sorted_or, 'r-', linewidth=1.5,label='Original pulse')
plt.xlabel('Frequency (THz)', fontsize=12)
plt.ylabel('Spectral Power (normalized)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(355,395)
plt.legend()
plt.show()


print('pulse duration after compression:',1e15*0.441/(fwhm_val_br*1e12))



