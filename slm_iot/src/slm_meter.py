import time
import logging
import serial
import sounddevice as sd

from src.slm_cal_SPL_timedomain import soundmeter_timedomain
from src.slm_cal_SPL_FFT import soundmeter_FFT
from src.slm_constant import optionCAL

logger = logging.getLogger(__name__)

# 0: FFT, 1: Time Domain
if optionCAL == 0:
    soundmeter = soundmeter_FFT
else:
    soundmeter = soundmeter_timedomain


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
