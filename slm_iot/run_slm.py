import time
import logging
import multiprocessing
from datetime import datetime
import RPi.GPIO as GPIO
import serial

from src.slm_meter import monitor_microphone as start_meter
from src.slm_meter import initilize_serialRS485
from src.slm_calibrate import calibrate_with_1khz_tone
from src.slm_logger import setup_logging

# Shared log filename (same across main and subprocesses)
LOG_FILENAME = datetime.now().strftime("logs/slm_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)

def run_slm(log_filename, output_queue):
    setup_logging(log_filename)
    for spl_dBA in start_meter():
        output_queue.put(spl_dBA)  # Send value to parent process
    logger.info("Sound level meter started.")

SWITCH1_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows compatibility
    setup_logging(LOG_FILENAME)
    logger = logging.getLogger(__name__)
    logger.info("Main process started.")
    output_queue = multiprocessing.Queue()  # ← create queue
    
    try:
        ser485 = initilize_serialRS485()
        logger.info(f"Serial port {ser485.port} opened successfully.")
    except serial.SerialException as e:
        logger.exception(f"Error opening or communicating with serial port: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

    # Start sound level meter
    slm_process = multiprocessing.Process(target=run_slm, args=(LOG_FILENAME, output_queue))
    slm_process.start()

    pressed_time = None
    try:
        while True:
             # Read SPL values if available
            while not output_queue.empty():
                msg_spl_dBA = output_queue.get()
                # Optional: Send to serial
                if ser485.is_open:
                    ser485.write(msg_spl_dBA)  # Assuming `spl_dBA` is a string like ":146.9\r"
                    
            if GPIO.input(SWITCH1_PIN):
                if pressed_time is None:
                    pressed_time = time.time()
                elif time.time() - pressed_time >= 3:
                    logger.info("Switch 1 held for 3 seconds - Starting calibration...")

                    slm_process.terminate()
                    slm_process.join()
                    
                    if ser485.is_open:
                        ser485.close()
                    
                    calibrate_with_1khz_tone()

                    # Restart meter
                    ser485 = initilize_serialRS485()
                    slm_process = multiprocessing.Process(target=run_slm, args=(LOG_FILENAME, output_queue))
                    slm_process.start()
                    pressed_time = None
            else:
                pressed_time = None

            time.sleep(0.1)
    finally:
        GPIO.cleanup()
        if ser485.is_open:
            ser485.close()
        slm_process.terminate()
        slm_process.join()
