import os
import logging
import json
import sounddevice as sd
from src.slm_meter import A_weighting, process_block
from src.slm_meter import SAMPLE_RATE, REF_PRESSURE
from src.slm_meter import initilize_serialport

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

def calibrate_with_1khz_tone():
    ser_new = initilize_serialport()
    global CALIBRATION_GAIN_1KHZ, ACTIVE_CALIBRATION_GAIN
    logger.info("Calibrating with 1 kHz tone at 94 dB...")

    msg = (':1CAL\r').encode()  # Ensure message ends with CRLF
    ser_new.write(msg)

    duration = 3
    recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    b_coeffs, a_coeffs = A_weighting(SAMPLE_RATE)
    measured_rms = process_block(recording[:, 0], b_coeffs, a_coeffs)
    target_rms = REF_PRESSURE * 10 ** (93.89 / 20)
    CALIBRATION_GAIN_1KHZ = target_rms / measured_rms
    ACTIVE_CALIBRATION_GAIN = CALIBRATION_GAIN_1KHZ
    logger.info(f"Calibration complete. Gain set to {CALIBRATION_GAIN_1KHZ}")
    save_calibration_result({"CALIBRATION_GAIN_1KHZ": CALIBRATION_GAIN_1KHZ})


def calibrate_with_sensitivity(mv_per_pa):
    global CALIBRATION_GAIN_SENS, ACTIVE_CALIBRATION_GAIN
    volts_per_pa = mv_per_pa / 1000.0
    CALIBRATION_GAIN_SENS = 1.0 / volts_per_pa
    ACTIVE_CALIBRATION_GAIN = CALIBRATION_GAIN_SENS
    print(f"Calibration using {mv_per_pa} mV/Pa sensitivity complete. Gain set to {CALIBRATION_GAIN_SENS:.4f}")
    save_calibration_result({"CALIBRATION_GAIN_SENS": CALIBRATION_GAIN_SENS, "mv_per_pa": mv_per_pa})