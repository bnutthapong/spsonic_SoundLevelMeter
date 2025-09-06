import os
import time
import logging
import json
import serial
import numpy as np
import sounddevice as sd

from src.slm_cal_SPL_timedomain import soundmeter_timedomain
from src.slm_cal_SPL_FFT import soundmeter_FFT

logger = logging.getLogger(__name__)

# Constants
REF_PRESSURE = 20e-6  # Reference pressure in pascals
LEQ_INTERVAL = 60     # Seconds (can be configured)
SAMPLE_RATE = 48000
CHUNK_SIZE = 1024
TIME_WEIGHTING = "fast"  # Options: "fast", "slow", "none"

ACTIVE_CALIBRATION_GAIN = 25.0000  # Default calibration gain for microphone UMIK-2

error_counter = 0
silent_frame_counter = 0
error_threshold = 10  # Number of consecutive silent or bad frames before abort
optionCAL = 0  # 0: FFT, 1: Time Domain

if optionCAL == 1:
    soundmeter = soundmeter_timedomain
else:
    soundmeter = soundmeter_FFT

def load_active_calibration_gain():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'mic_calibration.json')
    config_path = os.path.abspath(config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('ACTIVE_CALIBRATION_GAIN') or data.get('CALIBRATION_GAIN_1KHZ') or 1.0
    return 1.0

def get_alpha(time_constant, block_duration):
    return 1 - np.exp(-block_duration / time_constant)

def get_l90(buffer):
    return np.percentile(buffer, 10)

def initilize_serialport():
    ser = serial.Serial(
        port='/dev/ttyS0', # ttyS0, ttyUSB0
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )
    return ser


def monitor_microphone(time_weighting_value=None, rs232_or_rs485=None):
    while True:
        try:
            logger.info("Attempting to initialize microphone...")
            sd._terminate()
            sd._initialize()
            time.sleep(2)
            input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
            if not input_devices:
                raise sd.PortAudioError("No input device available")

            sd.default.device = input_devices[0]['name']
            logger.info(f"Using input device: {sd.default.device}")

            # Now send SPL values as normal
            for spl in soundmeter(time_weighting_value, rs232_or_rs485):
                yield spl

        except (sd.PortAudioError, sd.CallbackAbort, OSError) as e:
            logger.warning("Microphone disconnected or unavailable. Retrying in 5 seconds...")
            logger.debug(f"Full error: {e}")
            yield (":1Err1\r").encode()
            time.sleep(5)


if __name__ == "__main__":
    for spl in monitor_microphone():
        print(spl)
