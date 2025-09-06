import os
import numpy as np
import sounddevice as sd
import time, queue, logging, json

from src.slm_meter import (
    load_active_calibration_gain, get_alpha, get_l90,
    REF_PRESSURE, SAMPLE_RATE, CHUNK_SIZE, LEQ_INTERVAL, TIME_WEIGHTING,
    error_counter, silent_frame_counter, error_threshold
)

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
    -11.2, -8.5, -6.2, -4.4, -3.0, -1.9, -1.0, -0.3, 0.0,
    0.2, 0.3, 0.3, 0.2, 0.0, -0.2, -0.5, -0.8, -1.2,
    0.0, -0.5, -1.2, -2.0, -3.0, -4.1, -5.3, -6.6,
    -8.0, -9.5, -11.1, -12.8, -14.6, -16.5
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
        if len(idx) > 0:
            bands.append((fc, idx))
    return bands


def process_block(band_energies, weighting_curve_db, alpha,
                  smoothed_energy_prev, total_energy_accum,
                  total_time_accum, block_duration):
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

def load_band_calibration():
    """Load per-band calibration values in dB, return as numpy array."""
    band_calibration_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'band_calibration.json')
    band_calibration_path = os.path.abspath(band_calibration_path)
    with open(band_calibration_path, 'r', encoding='utf-8') as f:
        cal_data = json.load(f)
    # Ensure order matches CENTERS
    return np.array([cal_data[str(fc)] for fc in CENTERS])


# ======== Main function ========
def soundmeter_FFT(time_weighting_value=None, rs232_or_rs485=None):
    global ACTIVE_CALIBRATION_GAIN
    ACTIVE_CALIBRATION_GAIN = load_active_calibration_gain()
    band_cal_dB = load_band_calibration()

    logger.info("Starting Sound Level Meter (FFT + 1/3 Octave + Preloaded Tables) ...")

    bands = third_octave_bands(SAMPLE_RATE, CHUNK_SIZE)

    spl_buf_A = []
    interval_start = time.time()

    smoothed_A = smoothed_C = smoothed_Z = 0.0
    total_energy_A = total_energy_C = total_energy_Z = 0.0
    total_time_A = total_time_C = total_time_Z = 0.0

    current_weighting = (
        time_weighting_value.value if time_weighting_value else TIME_WEIGHTING
    )
    block_duration = CHUNK_SIZE / SAMPLE_RATE
    alpha = get_alpha(1.0 if current_weighting == "slow" else 0.125, block_duration)

    print_interval = 0.5
    last_print_time = time.time()
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        global error_counter, silent_frame_counter
        nonlocal spl_buf_A, interval_start, last_print_time
        nonlocal smoothed_A, smoothed_C, smoothed_Z
        nonlocal total_energy_A, total_energy_C, total_energy_Z
        nonlocal total_time_A, total_time_C, total_time_Z

        # Error handling
        if status:
            logger.error(f"Stream error: {status}")
            error_counter += 1
            if error_counter >= error_threshold:
                logger.error("Persistent input overflow — assuming mic disconnected.")
                raise sd.CallbackAbort
            return

        if np.all(indata == 0):
            silent_frame_counter += 1
            logger.warning(f"Input buffer is silent ({silent_frame_counter}/{error_threshold})")
            if silent_frame_counter >= error_threshold:
                logger.error("Too many silent buffers — mic likely unplugged.")
                raise sd.CallbackAbort
            return
        else:
            silent_frame_counter = 0
            error_counter = 0

        ## FFT + 1/3 Octave ##
        audio_data = indata[:, 0] * ACTIVE_CALIBRATION_GAIN
        windowed = audio_data * np.hanning(len(audio_data))
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum) / (len(windowed) / 2)
        pressure = mag / np.sqrt(2)  # RMS per bin

        # Convert calibration dB to linear gain for pressure
        calibration_gain = 10 ** (band_cal_dB / 20.0)

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

            lmax_val = max(spl_buf_A)
            lmin_val = min(spl_buf_A)
            l90_val  = get_l90(np.array(spl_buf_A))

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