import time
import logging
import serial
import sounddevice as sd
import queue

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


def monitor_microphone(time_weighting_value=None, rs232_or_rs485=None,
                       output_queue=None, display_queue=None, weighting_value=None):
    first_init = True # Track first-time initialization
    while True:
        try:
            # === Tell OLED what's happening ===
            if display_queue:
                msg = {"wifi": False}
                if first_init:
                    msg["initialise"] = True
                else:
                    msg["reboot"] = True

                try:
                    display_queue.put_nowait(msg)
                except queue.Full:
                    display_queue.get_nowait()
                    display_queue.put_nowait(msg)

            
            logger.info("Attempting to initialize microphone...")
            sd._terminate()
            sd._initialize()
            time.sleep(0.5)
            input_devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
            if not input_devices:
                raise sd.PortAudioError("No input device available")

            sd.default.device = input_devices[0]['name']
            logger.info(f"Using input device: {sd.default.device}")
            
            # Mark that we've done the first init
            first_init = False

            # Now send SPL values as normal
            for spl in soundmeter(time_weighting_value, rs232_or_rs485, output_queue, display_queue, weighting_value):
                yield spl

        except (sd.PortAudioError, sd.CallbackAbort, OSError) as e:
            logger.warning("Microphone disconnected or unavailable. Retrying in 5 seconds...")
            logger.debug(f"Full error: {e}")
            
            if display_queue:
                try:
                    display_queue.put_nowait({
                        "error": True,
                        "wifi": False,
                        "message": "Mic discon"  # you can customize
                    })
                except queue.Full:
                    display_queue.get_nowait()
                    display_queue.put_nowait({
                        "error": True,
                        "wifi": False,
                        "message": "Mic Error"
                    })
            
            yield (":1Err1\r").encode()
            time.sleep(5)


if __name__ == "__main__":
    for spl in monitor_microphone():
        print(spl)
