import os
import time
import numpy as np
import sounddevice as sd
import logging
import json
import serial
import queue

from scipy.signal import lfilter, bilinear

logger = logging.getLogger(__name__)

# Constants
REF_PRESSURE = 20e-6  # Reference pressure in pascals
LEQ_INTERVAL = 60     # Seconds (can be configured)
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
TIME_WEIGHTING = "slow"  # Options: "fast", "slow", "none"

ACTIVE_CALIBRATION_GAIN = 1.0

def load_active_calibration_gain():
    """Load ACTIVE_CALIBRATION_GAIN from calibration_results.json in config folder."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration_results.json')
    config_path = os.path.abspath(config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data.get('ACTIVE_CALIBRATION_GAIN') or data.get('CALIBRATION_GAIN_1KHZ') or 1.0
    return 1.0

def A_weighting(fs):
    """Return digital A-weighting filter coefficients (numerator, denominator) for sampling rate fs."""
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = np.convolve([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                       [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    DENs = np.convolve(np.convolve(DENs, [1, 2 * np.pi * f3]),
                       [1, 2 * np.pi * f2])

    b_coeffs, a_coeffs = bilinear(NUMs, DENs, fs)
    return b_coeffs, a_coeffs


def rms(samples):
    return np.sqrt(np.mean(samples ** 2))


def spl_db(pressure):
    return 20 * np.log10(pressure / REF_PRESSURE + 1e-10)


def get_l90(buffer):
    return np.percentile(buffer, 10)


def process_block(block, b_coeffs, a_coeffs):
    filtered = lfilter(b_coeffs, a_coeffs, block)
    return rms(filtered)


def get_alpha(time_constant, block_duration):
    return 1 - np.exp(-block_duration / time_constant)

def initilize_serialRS485():
    ser = serial.Serial(
        port='/dev/ttyS0',  # Change this to your USB-to-RS485 converter's port
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1  # Timeout in seconds
    )
    return ser

def soundmeter():
    global ACTIVE_CALIBRATION_GAIN
    ACTIVE_CALIBRATION_GAIN = load_active_calibration_gain()
    
    logger.info("Starting Sound Level Meter...")

    b_coeffs, a_coeffs = A_weighting(SAMPLE_RATE)
    leq_buf = []
    spl_buf = []
    interval_start = time.time()

    # Setup for time-weighting
    block_duration = CHUNK_SIZE / SAMPLE_RATE
    smoothed_energy = 0.0
    if TIME_WEIGHTING == "fast":
        alpha = get_alpha(0.125, block_duration)  # Fast = 125 ms
        print_interval = 0.125
    elif TIME_WEIGHTING == "slow":
        alpha = get_alpha(1.0, block_duration)    # Slow = 1 s
        print_interval = 1.0
    else:
        alpha = 1.0
        print_interval = 0.1

    last_print_time = time.time()
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        nonlocal leq_buf, spl_buf, interval_start, smoothed_energy, last_print_time

        if status:
            print("Stream error:", status)

        audio_data = indata[:, 0] * ACTIVE_CALIBRATION_GAIN
        block_rms = process_block(audio_data, b_coeffs, a_coeffs)

        # Apply time-weighting to squared RMS (energy)
        energy = block_rms ** 2
        if TIME_WEIGHTING in ["fast", "slow"]:
            energy = alpha * energy + (1 - alpha) * smoothed_energy
            smoothed_energy = energy

        block_spl = 10 * np.log10(energy / REF_PRESSURE ** 2 + 1e-10)

        leq_buf.append(energy)
        spl_buf.append(block_spl)

        now = time.time()
        if now - last_print_time >= print_interval:
            timestamp = time.strftime('%H:%M:%S')
            print(f"{timestamp} | Current SPL ({TIME_WEIGHTING.title()}): {block_spl:.1f} dBA")
            value = round(block_spl, 1)
            if value < 10.0:
                value += 100.0  # or whatever makes sense for smallest range
            elif value < 100.0:
                value += 100.0
            elif value < 1000.0:
                value += 10000.0
                
            spl_dBA = (':' + str(value) + '\r').encode()  # Ensure message ends with CRLF
            q.put(spl_dBA)  # Put into the queue
            last_print_time = now

        if now - interval_start >= LEQ_INTERVAL:
            timestamp = time.strftime('%H:%M:%S')
            leq_val = 10 * np.log10(np.mean(leq_buf) / REF_PRESSURE ** 2 + 1e-10)
            lmax_val = max(spl_buf)
            l90_val = get_l90(np.array(spl_buf))
            msg_result = f"Time: {timestamp} | SPL: {block_spl:.1f} dBA | Leq: {leq_val:.1f} dBA | Lmax: {lmax_val:.1f} dBA | L90: {l90_val:.1f} dBA"
            print(msg_result)
            logger.info(msg_result)
            leq_buf = []
            spl_buf = []
            interval_start = now

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, callback=callback):
        print("Press Ctrl+C to stop.")
        try:
            while True:
                try:
                    spl_dBA = q.get(timeout=1)
                    yield spl_dBA  # Yield value from callback
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            print("\nStopping meter...")

if __name__ == "__main__":
    soundmeter()
