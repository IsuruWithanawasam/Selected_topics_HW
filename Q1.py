# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 08:16:11 2025

@author: Isuru Withanawasam
"""

import numpy as np
import matplotlib.pyplot as plt

def propagate_gaussian_pulse(tau_input, thickness, k2_prime):
    
    GDD = k2_prime * thickness  # fs²
    tau_output = tau_input * np.sqrt(1 + (4 * np.log(2) * GDD / tau_input**2)**2)
    
    Area=tau_input*np.sqrt(4* np.log(2)/np.pi)
    I_peak=(Area/tau_output)*np.sqrt(4*np.log(2)/np.pi)
    
    return tau_output, GDD,I_peak

def gaussian_pulse(t, tau_fwhm, I_peak, t0=0):
    return I_peak*(np.exp(-4 * np.log(2) * (t - t0)**2 / tau_fwhm**2))

#parameters
tau_input = 50  # fs
thicknesses = [20, 30, 40]  # mm
k2_prime = 36.1 # fs²/mm fused crystal


I_peak=1

t = np.linspace(-200, 200, 1000)  # fs

input_intensity = gaussian_pulse(t, tau_input,I_peak)
plt.plot(t, input_intensity, 'b-', linewidth=2, label=f'Input: {tau_input} fs FWHM')
plt.xlabel('Time (fs)')
plt.ylabel('Intensity (a.u.)')
plt.title('Input Pulse')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-100, 100)
plt.show()

# Calculate and plot output pulses for different thicknesses
colors = ['red', 'green', 'orange']
results = []

for i, thickness in enumerate(thicknesses):
    tau_output, GDD, I_peak = propagate_gaussian_pulse(tau_input, thickness, k2_prime)
    output_intensity = gaussian_pulse(t, tau_output, I_peak)
    
    plt.plot(t, output_intensity, color=colors[i], linewidth=2, 
             label=f'{thickness} mm: {tau_output:.1f} fs FWHM')
    
    results.append((thickness, tau_output, GDD))

plt.plot(t, input_intensity, 'b--', linewidth=2, alpha=0.7, label='Input (50 fs)')
plt.xlabel('Time (fs)')
plt.ylabel('Intensity (a.u.)')
plt.title('Output Pulses for Different Thicknesses')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-200, 200)
plt.show()

# Find thickness for doubling pulse duration
target_duration = 2 * tau_input  # 100 fs
thicknesses_fine = np.linspace(1, 100, 1000)  # mm
durations = []

for thickness in thicknesses_fine:
    tau_out, _ ,_= propagate_gaussian_pulse(tau_input, thickness, k2_prime)
    durations.append(tau_out)

durations = np.array(durations)

# Find where duration doubles
idx_double = np.argmin(np.abs(durations - target_duration))
thickness_double = thicknesses_fine[idx_double]
duration_double = durations[idx_double]

# Plot pulse duration vs thickness
plt.plot(thicknesses_fine, durations, 'blue', linewidth=2)
plt.axhline(y=target_duration, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_duration} fs')
plt.axvline(x=thickness_double, color='red', linestyle='--', alpha=0.7)
plt.plot(thickness_double, duration_double, 'ro', markersize=8, 
         label=f'Double at {thickness_double:.1f} mm')
plt.xlabel('Thickness (mm)')
plt.ylabel('Output Pulse Duration (fs)')
plt.title('Pulse Duration vs Material Thickness')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 80)
plt.show()

# Compare input and doubled pulse
Area=tau_input*np.sqrt(4* np.log(2)/np.pi)
I_peak=(Area/duration_double)*np.sqrt(4*np.log(2)/np.pi)
print(I_peak)
doubled_intensity = gaussian_pulse(t, target_duration,I_peak)
print(max(doubled_intensity))
plt.plot(t, input_intensity, 'b-', linewidth=2, label=f'Input: {tau_input} fs')
plt.plot(t, doubled_intensity, 'r-', linewidth=2, label=f'Doubled: {target_duration} fs')
plt.xlabel('Time (fs)')
plt.ylabel('Intensity (a.u.)')
plt.title('Input vs Doubled Pulse Duration')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-200, 200)

plt.tight_layout()
plt.show()

# Print results
print("=== HOMEWORK PROBLEM 1 RESULTS ===")
print(f"Input pulse duration (FWHM): {tau_input} fs")
print(f"Material: Fused Quartz (k'' ≈ {k2_prime} fs²/mm)")
print()

print("Pulse durations after propagation:")
for thickness, tau_out, GDD in results:
    broadening_factor = tau_out / tau_input
    print(f"  {thickness:2d} mm: {tau_out:5.1f} fs (×{broadening_factor:.2f}, GDD = {GDD:.1f} fs²)")

print()
print(f"Thickness needed to double pulse duration:")
print(f"  {thickness_double:.1f} mm of fused quartz")
print(f"  Results in {duration_double:.1f} fs pulse duration")

# Verify the calculation using the dispersion length formula
# L_d = τ²/(4*ln(2)*|k''|) for doubling
L_d_theoretical = tau_input**2 / (4 * np.log(2) * k2_prime * 1000)  # Convert to mm
print(f"  Theoretical dispersion length L_d = {L_d_theoretical:.1f} mm")
print(f"  (Pulse doubles at L = 1.73 * L_d = {1.73 * L_d_theoretical:.1f} mm)")

# Additional analysis
print(f"\n=== DISPERSION ANALYSIS ===")
print(f"For a {tau_input} fs pulse in fused quartz:")
print(f"  GDD per mm: {k2_prime * 1000:.0f} fs²")
print(f"  Dispersion length: {L_d_theoretical:.1f} mm")
print(f"  At L = L_d: pulse broadens by factor of √2 = {np.sqrt(2):.3f}")
print(f"  At L = 1.73*L_d: pulse broadens by factor of 2.0")