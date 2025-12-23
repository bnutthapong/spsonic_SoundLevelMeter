import os
import logging
import json
import numpy as np
import time
from scipy.signal import get_window
from scipy.interpolate import interp1d
import queue

import sounddevice as sd
from src.slm_cal_SPL_timedomain import A_weighting, process_block
from src.slm_constant import SAMPLE_RATE, REF_PRESSURE, optionCAL, REF_DB
from src.slm_meter import initilize_serialport
from src.slm_auxiliary_function import wifi_connected
from src.slm_cal_SPL_FFT import CENTERS, C_CURVE_DB, A_CURVE_DB

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
        logger.info("Using FFT-based C-weighted RMS measurement for calibration.")
        measured_rms = rms_from_fft_c_weighted(x, SAMPLE_RATE, band_center=1000.0, band_width=100.0)
        measured_A_rms = rms_from_fft_a_weighted_table(x, SAMPLE_RATE, band_center=1000.0, band_width=100.0)
    else:
        logger.info("Using time-domain A-weighted RMS measurement for calibration.")
        b_coeffs, a_coeffs = A_weighting(SAMPLE_RATE)
        measured_rms = process_block(recording[:, 0], b_coeffs, a_coeffs)
        measured_A_rms = measured_rms
        
    
    logger.info(f"Measured RMS: {measured_rms:.6f} Pa")
    target_rms = REF_PRESSURE * 10 ** (REF_DB / 20)
    
    raw_rms = np.sqrt(np.mean(x**2))
    spl_Z = 20*np.log10(raw_rms / 2e-5)
    spl_A = 20*np.log10(measured_A_rms / 2e-5)  # from your A-weighted path
    
    # Compute gain
    CALIBRATION_GAIN_1KHZ = target_rms / measured_rms
    ACTIVE_CALIBRATION_GAIN = CALIBRATION_GAIN_1KHZ
    DELTA_DB = REF_DB - spl_Z
    logger.info(f"SPL(Z)={spl_Z:.2f} dB, SPL(A)={spl_A:.2f} dB, DELTA_DB(REF-spl_Z) ={DELTA_DB:.2f} dB")
    logger.info(f"Calibration complete. Gain set to {CALIBRATION_GAIN_1KHZ}")
    
    # Save result
    save_calibration_result({"CALIBRATION_GAIN_1KHZ": CALIBRATION_GAIN_1KHZ, "DELTA_DB": DELTA_DB})
    
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
    

def c_weighting_from_table(freqs):
    """
    Interpolate C-weighting table and convert dB -> linear amplitude.
    Protect against divide-by-zero for f=0.
    """
    f = np.asarray(freqs, dtype=float)
    
    # Avoid log10(0)
    f_safe = np.maximum(f, 1e-6)

    # Interpolate table in log-frequency domain
    interp_func = interp1d(np.log10(CENTERS), C_CURVE_DB, kind='linear',
                           bounds_error=False, fill_value="extrapolate")
    c_db = interp_func(np.log10(f_safe))

    # Convert dB -> linear amplitude
    W_lin = 10.0 ** (c_db / 20.0)
    return W_lin


# --- FFT-based RMS with table-based C-weighting ---
def rms_from_fft_c_weighted(signal, fs, band_center=1000.0, band_width=100.0,
                            window_type="hann"):
    """
    Compute RMS of signal using FFT and table-based C-weighting.
    Narrowband integration around 'band_center' +/- band_width/2.
    
    Parameters
    ----------
    signal : array_like
        Input time-domain signal.
    fs : float
        Sampling rate in Hz.
    band_center : float
        Center frequency for narrowband integration (Hz).
    band_width : float
        Width of narrowband (Hz).
    window_type : str
        Window function for FFT.
        
    Returns
    -------
    rms : float
        RMS in the same unit as input (e.g., Pascal)
    """
    x = np.asarray(signal, dtype=np.float64)
    N = len(x)
    if N < 1024:
        raise ValueError("Signal too short for reliable FFT-based RMS; need >= 1024 samples.")

    # Apply window
    w = get_window(window_type, N, fftbins=True)
    U = np.mean(w**2)  # window power normalization
    xw = x * w

    # FFT
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    
    # One-sided power spectrum
    ps = (np.abs(X)**2) / (N**2)
    if N % 2 == 0:
        ps[1:-1] *= 2.0
    else:
        ps[1:] *= 2.0

    # C-weighting linear amplitude
    W_lin = c_weighting_from_table(freqs)
    W_pow = W_lin**2

    # Narrowband mask
    half = band_width / 2.0
    mask = (freqs >= (band_center - half)) & (freqs <= (band_center + half))

    # Weighted power in narrowband
    weighted_power_sum = np.sum(ps[mask] * W_pow[mask])

    # Correct for window power
    rms_sq = weighted_power_sum / U
    rms = np.sqrt(max(rms_sq, 0.0))
    return rms


def a_weighting_from_table(freqs):
    """
    Interpolate A-weighting table and convert dB -> linear amplitude.
    Protect against divide-by-zero for f=0.
    """
    f = np.asarray(freqs, dtype=float)
    
    # Avoid log10(0)
    f_safe = np.maximum(f, 1e-6)

    # Interpolate table in log-frequency domain
    interp_func = interp1d(np.log10(CENTERS), A_CURVE_DB, kind='linear',
                           bounds_error=False, fill_value="extrapolate")
    a_db = interp_func(np.log10(f_safe))

    # Convert dB -> linear amplitude
    W_lin = 10.0 ** (a_db / 20.0)
    return W_lin


def rms_from_fft_a_weighted_table(signal, fs, band_center=1000.0, band_width=100.0,
                                  window_type="hann"):
    """
    Compute RMS of signal using FFT and table-based A-weighting.
    Narrowband integration around 'band_center' +/- band_width/2.
    """
    x = np.asarray(signal, dtype=np.float64)
    N = len(x)
    if N < 1024:
        raise ValueError("Signal too short for reliable FFT-based RMS; need >= 1024 samples.")

    # Window
    w = get_window(window_type, N, fftbins=True)
    U = np.mean(w**2)
    xw = x * w

    # FFT
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    ps = (np.abs(X)**2) / (N**2)
    if N % 2 == 0:
        ps[1:-1] *= 2.0
    else:
        ps[1:] *= 2.0

    # Table-based A-weighting linear amplitude
    W_lin = a_weighting_from_table(freqs)
    W_pow = W_lin**2

    # Narrowband mask
    half = band_width / 2.0
    mask = (freqs >= (band_center - half)) & (freqs <= (band_center + half))

    weighted_power_sum = np.sum(ps[mask] * W_pow[mask])

    rms_sq = weighted_power_sum / U
    rms = np.sqrt(max(rms_sq, 0.0))
    return rms
