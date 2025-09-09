import os
import numpy as np
import sounddevice as sd
import time, queue, logging, json

from src.slm_constant import (
    REF_PRESSURE, SAMPLE_RATE, CHUNK_SIZE, LEQ_INTERVAL, TIME_WEIGHTING,
    error_counter, silent_frame_counter, error_threshold
)
from src.slm_auxiliary_function import get_alpha, get_l90, wifi_connected
from src.slm_cal_SPL_timedomain import load_active_calibration_gain
from src.slm_oled_display import display_slm

logger = logging.getLogger(__name__)

# ======== 1/3-octave center frequencies ========
CENTERS = [
    12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
]

# ======== Pre-loaded weighting tables (dB) ========
# IEC Class-1 reference weightings (dB)
A_CURVE_DB = [
    -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5,
    -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9,
    -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5,
    -0.1, -1.1, -2.5, -4.3, -6.6, -9.3
]


C_CURVE_DB = [
    -11.2, -8.5, -6.2, -4.4, -3.0, -2, -1.3, -0.8, -0.5, -0.3,
    -0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0, -4.4,
    -6.2, -8.5, -11.2
]


Z_CURVE_DB = [0.0] * len(CENTERS)


# ======== Band mapping ========
def third_octave_bands(sample_rate, fft_size):
    freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    bands = []
    for fc in CENTERS:
        f1 = fc / (2 ** (1 / 6))
        f2 = fc * (2 ** (1 / 6))
        idx = np.where((freqs >= f1) & (freqs <= f2))[0]

        # Always append, even if empty
        bands.append((fc, idx))
        # print(f"{fc} Hz: bins={len(idx)}")
    return bands


def process_block(band_energies, weighting_curve_db, alpha,
                  smoothed_energy_prev, total_energy_accum,
                  total_time_accum, block_duration):
    
    # print("band_energies:", len(band_energies))
    # print("weighting_curve_db:", len(weighting_curve_db))
    
    """Process one FFT block for a given weighting (A, C, or Z)."""
    gains = 10 ** (np.array(weighting_curve_db) / 20.0)
    total_energy = np.sum(band_energies * gains**2)

    # Time-weighted SPL
    smoothed_energy = alpha * total_energy + (1 - alpha) * smoothed_energy_prev
    spl_value = 10 * np.log10(smoothed_energy / (REF_PRESSURE**2) + 1e-12)

    # Leq accumulation
    total_energy_accum += total_energy * block_duration
    total_time_accum += block_duration

    return spl_value, smoothed_energy, total_energy_accum, total_time_accum


def calc_leq(total_energy_accum, total_time_accum):
    """Calculate Leq from accumulated energy and time."""
    if total_time_accum <= 0:
        return None
    return 10 * np.log10(
        (total_energy_accum / total_time_accum) / (REF_PRESSURE**2) + 1e-12
    )

def load_band_offset():
    """Load per-band calibration values in dB, return as numpy array."""
    band_calibration_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'band_offset_in_dBA.json')
    band_calibration_path = os.path.abspath(band_calibration_path)
    with open(band_calibration_path, 'r', encoding='utf-8') as f:
        cal_data = json.load(f)
    return cal_data


def get_calibration_gain_for_bands(bands, cal_data):
    """Return calibration gains aligned to the bands used in processing."""
    band_freqs = [f for f, _ in bands]
    band_cal_dB = np.array([
        cal_data.get(str(fc), 0.0)  # fallback to 0 dB if missing
        for fc in band_freqs
    ])
    return 10 ** (band_cal_dB / 20.0)


# ======== Main function ========
def soundmeter_FFT(time_weighting_value=None, rs232_or_rs485=None):
    global ACTIVE_CALIBRATION_GAIN
    # Load calibration data once
    ACTIVE_CALIBRATION_GAIN = load_active_calibration_gain()
    cal_data = load_band_offset()
    spl_buf_A = []
    logger.info("Starting Sound Level Meter (FFT + 1/3 Octave + Preloaded Tables) ...")
    
    bands = third_octave_bands(SAMPLE_RATE, CHUNK_SIZE)
    interval_start = time.time()

    smoothed_A = smoothed_C = smoothed_Z = 0.0
    total_energy_A = total_energy_C = total_energy_Z = 0.0
    total_time_A = total_time_C = total_time_Z = 0.0

    current_weighting = (time_weighting_value.value if time_weighting_value else TIME_WEIGHTING)
    block_duration = CHUNK_SIZE / SAMPLE_RATE
    alpha = get_alpha(1.0 if current_weighting == "slow" else 0.125, block_duration)

    print_interval = 0.5
    last_print_time = time.time()
    q = queue.Queue()
    connected = wifi_connected()
    
    def callback(indata, frames, time_info, status):
        global error_counter, silent_frame_counter
        nonlocal spl_buf_A, interval_start, last_print_time
        nonlocal smoothed_A, smoothed_C, smoothed_Z
        nonlocal total_energy_A, total_energy_C, total_energy_Z
        nonlocal total_time_A, total_time_C, total_time_Z

        ## FFT + 1/3 Octave ##
        audio_data = indata[:, 0] * ACTIVE_CALIBRATION_GAIN
        windowed = audio_data * np.hanning(len(audio_data))
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum) / (len(windowed) / 2)
        pressure = mag / np.sqrt(2)  # RMS per bin

        # Convert calibration dB to linear gain for pressure
        calibration_gain = get_calibration_gain_for_bands(bands, cal_data)

        # Apply to band energies (Pa²) — gain² because energy ∝ pressure²
        band_energies = np.array([
            np.sum((pressure[idx] * calibration_gain[i])**2)
            for i, (_, idx) in enumerate(bands)
        ])

        # Process for A, C, Z
        LAF, smoothed_A, total_energy_A, total_time_A = process_block(
            band_energies, A_CURVE_DB, alpha,
            smoothed_A, total_energy_A, total_time_A, block_duration
        )
        LCF, smoothed_C, total_energy_C, total_time_C = process_block(
            band_energies, C_CURVE_DB, alpha,
            smoothed_C, total_energy_C, total_time_C, block_duration
        )
        LZF, smoothed_Z, total_energy_Z, total_time_Z = process_block(
            band_energies, Z_CURVE_DB, alpha,
            smoothed_Z, total_energy_Z, total_time_Z, block_duration
        )

        # Store SPL(A) for statistics (Lmax, Lmin, L90)
        spl_buf_A.append(LAF)
        
        # Lmax, Lmin, L90 calculation
        lmax_val = max(spl_buf_A)
        lmin_val = min(spl_buf_A)
        l90_val  = get_l90(np.array(spl_buf_A))

        ## Output to RS232/RS485 periodically ##
        now = time.time()
        if now - last_print_time >= print_interval:
            value = round(LAF, 1)
            if rs232_or_rs485.value == "rs232":
                if value < 10:
                    value = "S00" + str(value) + "\r\n"
                elif value > 100:
                    value = "S" + str(value) + "\r\n"
                else:
                    value = "S0" + str(value) + "\r\n"
                spl_dBA = value.encode()
            else:
                if value < 10.0:
                    value += 100.0
                elif value < 100.0:
                    value += 100.0
                elif value < 1000.0:
                    value += 10000.0
                spl_dBA = (':' + str(value) + '\r').encode()

            q.put(spl_dBA)
            last_print_time = now

        if now - interval_start >= LEQ_INTERVAL:
            # Calculate Leq for the interval
            leq_val = calc_leq(total_energy_A, total_time_A)
            lc_eq   = calc_leq(total_energy_C, total_time_C)
            lz_eq   = calc_leq(total_energy_Z, total_time_Z)

            msg_result = (
                f"Time: {time.strftime('%H:%M:%S')} | SPL(A): {LAF:.1f} dBA "
                f"| Leq: {leq_val:.1f} dBA | Lmax: {lmax_val:.1f} dBA "
                f"| Lmin: {lmin_val:.1f} dBA | L90: {l90_val:.1f} dBA "
                f"| SPL(C): {LCF:.1f} dBC | SPL(Z): {LZF:.1f} dBZ "
                f"| LCeq: {lc_eq:.1f} dBC | LZeq: {lz_eq:.1f} dBZ"
            )
            logger.info(msg_result)

            # Reset buffers and accumulators for the next interval
            spl_buf_A.clear()
            total_energy_A = total_energy_C = total_energy_Z = 0.0
            total_time_A   = total_time_C   = total_time_Z   = 0.0
            interval_start = now
        
        # Update weighting if changed
        display_slm(
            wifi=connected,
            mode=time_weighting_value.value,
            SPLA=LAF,
            Lmin=lmin_val,
            Lmax=lmax_val,
            Leq=leq_val if 'leq_val' in locals() else "-"
        )

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, callback=callback):
        print("Press Ctrl+C to stop.")
        try:
            while True:
                try:
                    spl_dBA = q.get(timeout=1)
                    yield spl_dBA
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            print("\nStopping meter...")