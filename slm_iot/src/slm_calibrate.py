import os
import logging
import json
import numpy as np
import time
from scipy.signal import get_window
import queue

import sounddevice as sd
from src.slm_cal_SPL_timedomain import A_weighting, process_block
from src.slm_constant import SAMPLE_RATE, REF_PRESSURE, optionCAL
from src.slm_meter import initilize_serialport
from src.slm_auxiliary_function import wifi_connected

logger = logging.getLogger(__name__)

# Global calibration gain (initially 1.0)
CALIBRATION_GAIN_1KHZ = None
CALIBRATION_GAIN_SENS = None

def save_calibration_result(result_dict, filename=None):
    """Save calibration result to a JSON file in the config folder. Create folder/file if missing."""
    if filename is None:
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        filename = os.path.join(config_dir, 'mic_calibration.json')
        filename = os.path.abspath(filename)
    # Ensure config directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(result_dict)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def calibrate_with_1khz_tone(display_queue=None):
    ser_new = initilize_serialport()
    global CALIBRATION_GAIN_1KHZ, ACTIVE_CALIBRATION_GAIN
    logger.info("Calibrating with 1 kHz tone at 94 dB...")

    msg = (':1CAL\r').encode()  # Ensure message ends with CRLF
    ser_new.write(msg)
    
    """Display countdown 3 → 1 before recording."""
    wifi_status = wifi_connected()  # True/False
    
    for i in range(3, 0, -1):
        if display_queue:
            try:
                display_queue.put_nowait({"countdown": i, "wifi": wifi_status})
            except queue.Full:
                display_queue.get_nowait()
                display_queue.put_nowait({"countdown": i, "wifi": wifi_status})
        time.sleep(1)
    
    # Start recording
    if display_queue:
        try:
            display_queue.put_nowait({"countdown": -1, "wifi": wifi_status})
        except queue.Full:
            display_queue.get_nowait()
            display_queue.put_nowait({"countdown": -1, "wifi": wifi_status})
    
    duration = 3
    recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype='float64')
    
    # Continuous OLED update during recording
    start_time = time.time()
    while time.time() - start_time < duration:
        if display_queue:
            elapsed = time.time() - start_time
            try:
                display_queue.put_nowait({"countdown": -1, "wifi": wifi_status, "recording_progress": elapsed / duration})
            except queue.Full:
                display_queue.get_nowait()
                display_queue.put_nowait({"countdown": -1, "wifi": wifi_status, "recording_progress": elapsed / duration})
        time.sleep(0.1)  # update every 0.1 s
    
    sd.wait()
    
    x = recording[:, 0]
    if not np.isfinite(x).all() or np.max(np.abs(x)) < 1e-6:
        logger.error("Invalid recording (NaNs/Inf or near-zero).")
        return False
    if np.max(np.abs(x)) > 0.98:
        logger.warning("Potential clipping detected in calibration recording.")

    # Measure RMS level
    if optionCAL == 0:
        logger.info("Using FFT-based A-weighted RMS measurement for calibration.")
        measured_rms = rms_from_fft_a_weighted(x, SAMPLE_RATE, band_center=1000.0, band_width=100.0)
    else:
        logger.info("Using time-domain A-weighted RMS measurement for calibration.")
        b_coeffs, a_coeffs = A_weighting(SAMPLE_RATE)
        measured_rms = process_block(recording[:, 0], b_coeffs, a_coeffs)
    
    logger.info(f"Measured RMS: {measured_rms:.6f} Pa")
    target_rms = REF_PRESSURE * 10 ** (93.89 / 20)
    
    raw_rms = np.sqrt(np.mean(x**2))
    spl_Z = 20*np.log10(raw_rms / 2e-5)
    spl_A = 20*np.log10(measured_rms / 2e-5)  # from your A-weighted path
    logger.info(f"SPL(Z)={spl_Z:.2f} dB, SPL(A)={spl_A:.2f} dB")
    
    # Compute gain
    CALIBRATION_GAIN_1KHZ = target_rms / measured_rms
    ACTIVE_CALIBRATION_GAIN = CALIBRATION_GAIN_1KHZ
    
    logger.info(f"Calibration complete. Gain set to {CALIBRATION_GAIN_1KHZ}")
    
    # Save result
    save_calibration_result({"CALIBRATION_GAIN_1KHZ": CALIBRATION_GAIN_1KHZ})
    
    # Final display update
    if display_queue:
        try:
            display_queue.put_nowait({"countdown": -2, "wifi": wifi_status})
        except queue.Full:
            display_queue.get_nowait()
            display_queue.put_nowait({"countdown": -2, "wifi": wifi_status})
    
    
    if display_queue:
        try:
            display_queue.put_nowait({"reboot": True, "wifi": wifi_status})
        except queue.Full:
            display_queue.get_nowait()
            display_queue.put_nowait({"reboot": True, "wifi": wifi_status})


def calibrate_with_sensitivity(mv_per_pa):
    global CALIBRATION_GAIN_SENS, ACTIVE_CALIBRATION_GAIN
    volts_per_pa = mv_per_pa / 1000.0
    CALIBRATION_GAIN_SENS = 1.0 / volts_per_pa
    ACTIVE_CALIBRATION_GAIN = CALIBRATION_GAIN_SENS
    print(f"Calibration using {mv_per_pa} mV/Pa sensitivity complete. Gain set to {CALIBRATION_GAIN_SENS:.4f}")
    save_calibration_result({"CALIBRATION_GAIN_SENS": CALIBRATION_GAIN_SENS, "mv_per_pa": mv_per_pa})
    

def a_weighting_db(freqs):
    """
    Standard A-weighting in dB re: 1 kHz = 0 dB.
    Returns array same shape as freqs.
    """
    f = np.array(freqs, dtype=float)
    f2 = f * f
    ra_num = (12200.0**2) * (f2**2)
    ra_den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12200.0**2)
    ra = ra_num / np.maximum(ra_den, 1e-30)
    A = 20.0 * np.log10(ra) + 2.0  # yields ~0 dB at 1 kHz
    # Handle f=0 exactly (A-weighting -> -inf dB). Force very negative.
    A = np.where(f == 0, -1000.0, A)
    return A

def a_weighting_linear(freqs):
    return 10.0 ** (a_weighting_db(freqs) / 20.0)

def rms_from_fft_a_weighted(signal, fs, band_center=1000.0, band_width=100.0, window_type="hann"):
    """
    Compute A-weighted RMS via one-sided FFT with correct window power scaling
    and narrowband integration around 1 kHz.

    - Applies window to reduce leakage.
    - Uses Parseval-consistent power scaling.
    - Applies A-weighting as power weights (|A(f)|^2).
    - Integrates only bins within [center - width/2, center + width/2].

    Returns RMS in the same units as 'signal'.
    """
    x = np.asarray(signal, dtype=np.float64)
    N = len(x)
    if N < 1024:
        raise ValueError("Signal too short for reliable FFT-based RMS; need >= 1024 samples.")

    # Window and its power normalization (U = mean(w^2))
    w = get_window(window_type, N, fftbins=True)
    U = np.mean(w**2)  # window power normalization
    xw = x * w

    # One-sided FFT
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    # One-sided power spectrum with Parseval-consistent scaling:
    # For real x: sum(xw^2) = (1/N) * sum_k |X|^2
    # Original (unwindowed) power ≈ sum(x^2) = (1/U) * sum(xw^2)
    # So RMS^2 = (1/N) * sum(x^2) ≈ (1/(N*U)) * (1/N) * sum_k |X|^2
    # For one-sided spectrum, double non-DC/Nyquist bins.
    ps_two_sided = (np.abs(X)**2) / (N**2)  # power per bin (two-sided, normalized by N^2)
    ps_one_sided = ps_two_sided.copy()
    if N % 2 == 0:
        # Even N has Nyquist bin
        ps_one_sided[1:-1] *= 2.0
    else:
        ps_one_sided[1:] *= 2.0

    # A-weighting as power weights
    A_lin = a_weighting_linear(freqs)
    A_pow = A_lin**2

    # Narrowband mask
    half = band_width / 2.0
    mask = (freqs >= (band_center - half)) & (freqs <= (band_center + half))

    # Weighted power in band
    weighted_power_sum = np.sum(ps_one_sided[mask] * A_pow[mask])

    # Convert FFT bin-sum to time-domain RMS^2 with window power correction
    rms_sq = weighted_power_sum / U
    rms = np.sqrt(max(rms_sq, 0.0))
    return rms

