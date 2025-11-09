import os
import json
import logging
import numpy as np
import subprocess

logger = logging.getLogger(__name__)

def get_alpha(time_constant, block_duration):
    return 1 - np.exp(-block_duration / time_constant)


def get_l90(buffer):
    return np.percentile(buffer, 10)


def load_active_calibration_gain():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'mic_calibration.json')
    config_path = os.path.abspath(config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('ACTIVE_CALIBRATION_GAIN') or data.get('CALIBRATION_GAIN_1KHZ') or 1.0
    return 1.0


# Check Wi-Fi connection
def wifi_connected():
    try:
        ssid = subprocess.check_output("iwgetid -r", shell=True).decode().strip()
        return bool(ssid)
    except Exception:
        return False