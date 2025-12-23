import os
import numpy as np
import sounddevice as sd
import time
import logging
import json
import queue

from src.slm_constant import (
    REF_PRESSURE, SAMPLE_RATE, CHUNK_SIZE, LEQ_INTERVAL, TIME_WEIGHTING
)
from src.slm_auxiliary_function import get_alpha, get_l90, wifi_connected, write_daily_csv
from src.slm_cal_SPL_timedomain import load_active_calibration_gain
from src.slm_oled_display import display_slm, display_reboot, display_calibration, display_welcome, display_msg
from src.slm_mqtt_helper import publish_leq

logger = logging.getLogger(__name__)

# 1/3-octave center frequencies
CENTERS = [
    12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
]

# Weightings (dB)
A_CURVE_DB = [
    -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5,
    -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9,
    -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5,
    -4.3, -6.6, -9.3
]

C_CURVE_DB = [
    -11.2, -8.5, -6.2, -4.4, -3.0, -2, -1.3, -0.8, -0.5, -0.3,
    -0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0, -4.4,
    -6.2, -8.5, -11.2
]

Z_CURVE_DB = [0.0] * len(CENTERS)

_display_queue = None  # Global queue for display thread

def set_display_queue(shared_queue):
    global _display_queue
    _display_queue = shared_queue
    

def _display_thread():
    """Central OLED display thread for all modes."""
    last_data = None
    while True:
        if _display_queue is not None:
            try:
                # Drain queue to get the latest message
                while True:
                    last_data = _display_queue.get_nowait()
            except queue.Empty:
                pass

            if last_data is not None:
                if "welcome" in last_data and last_data["welcome"]:
                    display_welcome(wifi=last_data.get("wifi", False))
                elif "reboot" in last_data and last_data["reboot"]:
                    display_reboot(wifi=last_data.get("wifi", False))
                elif "countdown" in last_data:
                    display_calibration(
                        countdown=last_data["countdown"],
                        wifi=last_data.get("wifi", False)
                    )
                elif "error" in last_data and last_data["error"]:
                    display_msg(
                        message=last_data.get("message", "Error"),
                        wifi=last_data.get("wifi", False)
                    )
                elif "ap_mode" in last_data and last_data["ap_mode"]:
                    display_msg(
                        message=last_data.get("message", "AP Mode running"),
                        wifi=last_data.get("wifi", False))
                elif "initialise" in last_data and last_data["initialise"]:
                    display_msg(
                        message=last_data.get("message", "Initializing..."),
                        wifi=last_data.get("wifi", False))
                elif "ap_mode_prep" in last_data and last_data["ap_mode_prep"]:
                    display_msg(
                        message=last_data.get("message", "Preppare AP Mode"),
                        wifi=last_data.get("wifi", False))
                else:
                    display_slm(**last_data)

        time.sleep(0.5)


# ======== Helper functions ========
def third_octave_bands(sample_rate, fft_size):
    freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    bands = []
    for fc in CENTERS:
        f1 = fc / (2 ** (1 / 6))
        f2 = fc * (2 ** (1 / 6))
        idx = np.where((freqs >= f1) & (freqs <= f2))[0]
        bands.append((fc, idx))
    return bands


def process_block(band_energies, weighting_curve_db, alpha,
                  smoothed_energy_prev, total_energy_accum,
                  total_time_accum, block_duration, delta_db):
    gains = 10 ** (np.array(weighting_curve_db) / 20.0)
    total_energy = np.sum(band_energies * gains**2)
    smoothed_energy = alpha * total_energy + (1 - alpha) * smoothed_energy_prev
    spl_value = 10 * np.log10(smoothed_energy / (REF_PRESSURE**2) + 1e-12)
    total_energy_accum += total_energy * block_duration
    total_time_accum += block_duration
    return spl_value, smoothed_energy, total_energy_accum, total_time_accum


def calc_leq(total_energy_accum, total_time_accum):
    if total_time_accum <= 0:
        return None

    leq = 10 * np.log10((total_energy_accum / total_time_accum) / (REF_PRESSURE**2) + 1e-12)

    return round(leq, 1)


def load_band_offset():
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'band_offset_in_dBZ.json')
    with open(os.path.abspath(path), 'r', encoding='utf-8') as f:
        return json.load(f)

    
def load_delta_db():
    path = os.path.join(os.path.dirname(__file__), '..', 'config', 'mic_calibration.json')
    with open(os.path.abspath(path), 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data["DELTA_DB"]


def get_calibration_gain_for_bands(bands, cal_data):
    band_freqs = [f for f, _ in bands]
    band_cal_dB = np.array([cal_data.get(str(fc), 0.0) for fc in band_freqs])
    return 10 ** (band_cal_dB / 20.0)


# ======== Main function ========
def soundmeter_FFT(time_weighting_value=None, rs232_or_rs485=None, output_queue=None, display_queue=None, weighting_value=None, mqtt_client=None, mqtt_cfg=None, datalogs_dir=None):
    global ACTIVE_CALIBRATION_GAIN
    if display_queue is None:
        raise ValueError("display_queue must be provided")  # No fallback queue

    ACTIVE_CALIBRATION_GAIN = load_active_calibration_gain()
    band_offset_dbz = load_band_offset()
    delta_db = load_delta_db()
    
    spl_buf = []

    logger.info("Starting Sound Level Meter ...")
    bands = third_octave_bands(SAMPLE_RATE, CHUNK_SIZE)
    interval_start = time.time()

    smoothed_A = smoothed_C = smoothed_Z = 0.0
    total_energy_A = total_energy_C = total_energy_Z = 0.0
    total_time_A = total_time_C = total_time_Z = 0.0

    current_weighting = (time_weighting_value.value if time_weighting_value else TIME_WEIGHTING)
    block_duration = CHUNK_SIZE / SAMPLE_RATE
    alpha = get_alpha(1.0 if current_weighting == "slow" else 0.125, block_duration)
    
    #print_interval = 0.5  # RS232/RS485 interval
    last_print_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal spl_buf, interval_start, last_print_time
        nonlocal smoothed_A, smoothed_C, smoothed_Z
        nonlocal total_energy_A, total_energy_C, total_energy_Z
        nonlocal total_time_A, total_time_C, total_time_Z

        # Check wifi status
        network_connected = False
        network_connected = wifi_connected()
        
        audio_data = indata[:, 0] * ACTIVE_CALIBRATION_GAIN

        w = np.hanning(len(audio_data))
        U = np.mean(w**2)
        windowed = audio_data * w

        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum) / (len(windowed) * np.sqrt(U))

        # double non-DC/Nyquist bins
        if len(audio_data) % 2 == 0:
            mag[1:-1] *= 2
        else:
            mag[1:] *= 2

        pressure = mag / np.sqrt(2)  # convert to RMS

        calibration_gain = get_calibration_gain_for_bands(bands, band_offset_dbz)
        band_energies = np.array([
            np.sum((pressure[idx] * calibration_gain[i])**2)
            for i, (_, idx) in enumerate(bands)
        ])
        
        # Process for A, C, Z
        LAF, smoothed_A, total_energy_A, total_time_A = process_block(
            band_energies, A_CURVE_DB, alpha, smoothed_A, total_energy_A, total_time_A, block_duration, delta_db
        )
        LCF, smoothed_C, total_energy_C, total_time_C = process_block(
            band_energies, C_CURVE_DB, alpha, smoothed_C, total_energy_C, total_time_C, block_duration, delta_db
        )
        LZF, smoothed_Z, total_energy_Z, total_time_Z = process_block(
            band_energies, Z_CURVE_DB, alpha, smoothed_Z, total_energy_Z, total_time_Z, block_duration, delta_db
        )

        if weighting_value.value == "C":
            display_SPL = round(LCF,1)
        elif weighting_value.value == "Z":
            display_SPL = round(LZF,1)
        else:
            display_SPL = round(LAF,1)
        
        spl_buf.append(display_SPL)
        lmax_val = max(spl_buf)
        lmin_val = min(spl_buf)
        l90_val = get_l90(np.array(spl_buf))

        now = time.time()

        # ---- RS232/RS485 output every print_interval ----
        if rs232_or_rs485.value == "rs232":
            print_interval = 0.125
        else:
            print_interval = 0.5
        
        # logger.info(f"Print interval set to {print_interval} seconds based on mode {rs232_or_rs485.value}")
        
        if now - last_print_time >= print_interval :
            value = round(display_SPL, 1)
            if rs232_or_rs485.value == "rs232":
                if value < 10:
                    msg = f"S00{value}\r\n"
                elif value > 100:
                    msg = f"S{value}\r\n"
                else:
                    msg = f"S0{value}\r\n"
            else:
                if value < 10.0:
                    value += 100.0
                elif value < 100.0:
                    value += 100.0
                elif value < 1000.0:
                    value += 10000.0
                msg = f":{value}\r"
                
            # Non-blocking enqueue
            try:
                output_queue.put_nowait(msg.encode())
            except queue.Full:
                # logger.debug("RS232-RS485 queue Empty")
                output_queue.get_nowait()
                output_queue.put_nowait(msg.encode())
            
            last_print_time = now
            
        # ---- Leq calculation per LEQ_INTERVAL ----
        if now - interval_start >= LEQ_INTERVAL:
            leq_val = calc_leq(total_energy_A, total_time_A)
            lc_eq = calc_leq(total_energy_C, total_time_C)
            lz_eq = calc_leq(total_energy_Z, total_time_Z)

            # logger.info(
            #     f"Time: {time.strftime('%H:%M:%S')} | SPL(A): {LAF:.1f} dBA "
            #     f"| Leq: {leq_val:.1f} dBA | Lmax: {lmax_val:.1f} dBA "
            #     f"| Lmin: {lmin_val:.1f} dBA | L90: {l90_val:.1f} dBA "
            #     f"| SPL(C): {LCF:.1f} dBC | SPL(Z): {LZF:.1f} dBZ "
            #     f"| LCeq: {lc_eq:.1f} dBC | LZeq: {lz_eq:.1f} dBZ"
            # )
            
            # ---- Write daily CSV log ----
            if datalogs_dir is not None:
                write_daily_csv(
                    datalogs_dir,
                    node_id=mqtt_cfg.get("node_id", "slm_node"),
                    leq_val=leq_val,
                    lmax_val=lmax_val,
                    lmin_val=lmin_val,
                    l90_val=l90_val,
                    spl_current=display_SPL
                )
            
            # ---- MQTT publish ----
            if network_connected:
                # --- MQTT publishing ---
                if mqtt_client is not None and mqtt_cfg is not None:
                    try:
                        publish_leq(
                            timestamp=f"{time.strftime('%H:%M')}",
                            mqtt_client=mqtt_client,
                            mqtt_topic=mqtt_cfg['topic'],
                            node_id=mqtt_cfg.get('node_id', 'slm_node'),
                            leq_val=leq_val,
                            lmax_val=lmax_val,
                            lmin_val=lmin_val,
                            l90_val=l90_val,
                            spl_current=display_SPL
                        )
                    except Exception:
                        logger.exception("MQTT publish failed")
            
            spl_buf.clear()
            total_energy_A = total_energy_C = total_energy_Z = 0.0
            total_time_A = total_time_C = total_time_Z = 0.0
            interval_start = now
            
        # ---- Push OLED display data (non-blocking) ----
        display_data = {
            "wifi": network_connected,
            "mode": time_weighting_value.value,
            "SPL": display_SPL,
            "Lmin": lmin_val,
            "Lmax": lmax_val,
            "Leq": leq_val if 'leq_val' in locals() else "-",
            "weighting_show": weighting_value.value
        }
        
        try:
            display_queue.put_nowait(display_data)
        except queue.Full:
            # Drain the queue until empty
            try:
                while True:
                    display_queue.get_nowait()
            except queue.Empty:
                # logger.debug("display_data queue Empty")
                pass
            display_queue.put_nowait(display_data)


    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, callback=callback):
        print("Press Ctrl+C to stop.")
        try:
            while True:
                try:
                    spl_dBA = output_queue.get(timeout=1)
                    yield spl_dBA
                except queue.Empty:
                    # logger.debug("spl_dBA queue Empty")
                    continue
        except KeyboardInterrupt:
            print("\nStopping meter...")

    logger.info("Sound Level Meter stopped.")
