import time
import logging
import multiprocessing
from multiprocessing import Manager
from datetime import datetime
import RPi.GPIO as GPIO
import serial
import subprocess
import threading

from src.slm_meter import monitor_microphone as start_meter
from src.slm_meter import initilize_serialport
from src.slm_calibrate import calibrate_with_1khz_tone
from src.slm_logger import setup_logging
from src.slm_apmode import enable_ap_mode, disable_ap_mode
from src.slm_cal_SPL_FFT import _display_thread, set_display_queue

# Shared log filename
LOG_FILENAME = datetime.now().strftime("slm_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)

def run_slm(log_filename, output_queue, time_weighting_value, rs232_or_rs485, display_queue):
    setup_logging(log_filename)
    for spl_dBA in start_meter(time_weighting_value, rs232_or_rs485, output_queue, display_queue):
        output_queue.put(spl_dBA)  # Send value to parent process

# GPIO setup
SWITCH1_PIN = 17
SWITCH2_PIN = 27
SWITCH3_PIN = 22
SWITCH4_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SWITCH2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SWITCH3_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SWITCH4_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

prev_switch_state = GPIO.input(SWITCH3_PIN)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    setup_logging(LOG_FILENAME)
    logger = logging.getLogger(__name__)
    logger.info("Sound Level Meter (SLM) started.")

    # Manager for shared state and queues
    manager = Manager()
    output_queue = manager.Queue()
    display_queue = manager.Queue(maxsize=1)
    set_display_queue(display_queue)

    # Start OLED display thread in main process
    threading.Thread(target=_display_thread, daemon=True).start()

    # Shared values
    time_weighting_value = manager.Value('u', 'fast')
    rs232_or_rs485 = manager.Value('u', 'rs485')

    # Initialize serial port
    try:
        ser_port = initilize_serialport()
        if ser_port.is_open and rs232_or_rs485.value == "rs485":
            for val in ["1-", "1--", "1---", "1----"]:
                ser_port.write(f":{val}\r".encode())
                time.sleep(1)
        logger.info(f"Serial port {ser_port.port} opened successfully.")
    except Exception as e:
        logger.exception(f"Serial port error: {e}")

    # Start SLM process
    slm_process = multiprocessing.Process(
        target=run_slm,
        args=(LOG_FILENAME, output_queue, time_weighting_value, rs232_or_rs485, display_queue)
    )
    slm_process.start()

    pressed_time = None
    try:
        # Set initial mode based on RS232/RS485 switch
        if prev_switch_state:
            rs232_or_rs485.value = "rs232"
            ser_port.write((':1r232\r').encode())
        else:
            rs232_or_rs485.value = "rs485"

        while True:
            # Read SPL values from SLM process and optionally send to serial
            while not output_queue.empty():
                msg_spl_dBA = output_queue.get()
                if ser_port.is_open:
                    ser_port.write(msg_spl_dBA)

            # Switch1: Calibration
            if GPIO.input(SWITCH1_PIN):
                if pressed_time is None:
                    pressed_time = time.time()
                elif time.time() - pressed_time >= 3:
                    logger.info("Switch 1 held for 3 seconds - Starting calibration...")
                    slm_process.terminate()
                    slm_process.join()
                    if ser_port.is_open:
                        ser_port.close()

                    # Calibration in a separate thread (non-blocking display)
                    calib_thread = threading.Thread(
                        target=calibrate_with_1khz_tone,
                        kwargs={"display_queue": display_queue}
                    )
                    calib_thread.start()
                    calib_thread.join()

                    # Restart SLM process
                    ser_port = initilize_serialport()
                    slm_process = multiprocessing.Process(
                        target=run_slm,
                        args=(LOG_FILENAME, output_queue, time_weighting_value, rs232_or_rs485, display_queue)
                    )
                    slm_process.start()
                    pressed_time = None

            # Switch2: Hotspot mode
            elif GPIO.input(SWITCH2_PIN):
                if pressed_time is None:
                    pressed_time = time.time()
                elif time.time() - pressed_time >= 3:
                    logger.info("Switch 2 held for 3 seconds - Hotspot ON...")
                    slm_process.terminate()
                    slm_process.join()
                    if ser_port.is_open and rs232_or_rs485.value == "rs485":
                        ser_port.write((':1CONF\r').encode())
                        ser_port.close()

                    enable_ap_mode()
                    logger.info("Hotspot mode enabled")
                    hotspot_proc = subprocess.Popen([
                        "python3", "slm/src/slm_hotspot_server.py",
                        "--host", "192.168.4.1", "--port", "8080"
                    ])
                    while True:
                        time.sleep(0.5)
                        if GPIO.input(SWITCH2_PIN):
                            break
                    hotspot_proc.terminate()
                    hotspot_proc.wait()
                    disable_ap_mode()
                    pressed_time = None

            # Switch4: Toggle time weighting
            elif GPIO.input(SWITCH4_PIN):
                if time_weighting_value.value == "fast":
                    time_weighting_value.value = "slow"
                    mode_active = (':1-SL-\r').encode()
                else:
                    time_weighting_value.value = "fast"
                    mode_active = (':1-FA-\r').encode()
                if ser_port.is_open and rs232_or_rs485.value == "rs485":
                    ser_port.write(mode_active)
                    time.sleep(1)
            else:
                pressed_time = None

            # RS232/RS485 switch
            current_switch_state = GPIO.input(SWITCH3_PIN)
            if current_switch_state != prev_switch_state:
                if current_switch_state:
                    rs232_or_rs485.value = "rs232"
                    ser_port.write((':1r232\r').encode())
                else:
                    rs232_or_rs485.value = "rs485"
                    ser_port.write((':1r485\r').encode())
                    time.sleep(1)
                prev_switch_state = current_switch_state

    finally:
        GPIO.cleanup()
        if ser_port.is_open:
            ser_port.close()
        slm_process.terminate()
        slm_process.join()
