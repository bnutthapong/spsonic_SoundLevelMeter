import time
import logging
import multiprocessing
from multiprocessing import Manager
from datetime import datetime
import RPi.GPIO as GPIO
import serial
import subprocess

from src.slm_meter import monitor_microphone as start_meter
from src.slm_meter import initilize_serialport
from src.slm_calibrate import calibrate_with_1khz_tone
from src.slm_logger import setup_logging
from src.slm_apmode import enable_ap_mode, disable_ap_mode

# Shared log filename (same across main and subprocesses)
LOG_FILENAME = datetime.now().strftime("slm/slm_logs/slm_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)

def run_slm(log_filename, output_queue, time_weighting_value, rs232_or_rs485):
    setup_logging(log_filename)
    for spl_dBA in start_meter(time_weighting_value, rs232_or_rs485):
        output_queue.put(spl_dBA)  # Send value to parent process

SWITCH1_PIN = 17
SWITCH2_PIN = 27
SWITCH3_PIN = 22
SWITCH4_PIN = 18
mode_active = None

GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SWITCH2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SWITCH3_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(SWITCH4_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

prev_switch_state = GPIO.input(SWITCH3_PIN)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows compatibility
    setup_logging(LOG_FILENAME)
    logger = logging.getLogger(__name__)
    logger.info("Sound Level Meter (SLM) started.")
    output_queue = multiprocessing.Queue()  # ← create queue
    
    # Create a shared value for time weighting
    # Using a Manager to share state between processes
    manager = Manager()
    time_weighting_value = manager.Value('u', 'fast')  # 'u' = Unicode string, initial value 'fast'
    rs232_or_rs485 = manager.Value('u', 'rs485')  # 'u' = Unicode string, initial value 'rs485'

    try:
        ser_port = initilize_serialport()
        if ser_port.is_open and rs232_or_rs485.value == "rs485":
            for val in ["1-", "1--", "1---", "1----"]:
                start_msg = (f':{val}\r').encode()  # Ensure message ends with CRLF
                ser_port.write(start_msg)
                time.sleep(1)  # Allow time for the message to be sent

        logger.info(f"Serial port {ser_port.port} opened successfully.")
    except serial.SerialException as e:
        logger.exception(f"Error opening or communicating with serial port: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

    # Start sound level meter
    slm_process = multiprocessing.Process(
        target=run_slm,
        args=(LOG_FILENAME, output_queue, time_weighting_value, rs232_or_rs485)
    )
    slm_process.start()

    pressed_time = None
    try:
        # Set initial mode based on actual switch position
        if prev_switch_state:
            rs232_or_rs485.value = "rs232"
            logger.info("RS232 mode selected at startup.")
            ser_port.write((':1r232\r').encode())
        else:
            rs232_or_rs485.value = "rs485"
            logger.info("RS485 mode selected at startup.")

        while True:
             # Read SPL values if available
            while not output_queue.empty():
                msg_spl_dBA = output_queue.get()
                # Optional: Send to serial
                if ser_port.is_open:
                    ser_port.write(msg_spl_dBA)  # Assuming `spl_dBA` is a string like ":146.9\r"
                    
            if GPIO.input(SWITCH1_PIN):
                if pressed_time is None:
                    pressed_time = time.time()
                elif time.time() - pressed_time >= 3:
                    logger.info("Switch 1 held for 3 seconds - Starting calibration...")
                    slm_process.terminate()
                    slm_process.join()
                    if ser_port.is_open:
                        ser_port.close()
                    calibrate_with_1khz_tone()
                    # Restart meter
                    ser_port = initilize_serialport()
                    slm_process = multiprocessing.Process(
                        target=run_slm,
                        args=(LOG_FILENAME, output_queue, time_weighting_value, rs232_or_rs485)
                    )
                    slm_process.start()
                    pressed_time = None
                    
            elif GPIO.input(SWITCH2_PIN):
                if pressed_time is None:
                    pressed_time = time.time()
                elif time.time() - pressed_time >= 3:
                    logger.info("Switch 2 held for 3 seconds - Hotspot ON...")
                    # Stop SLM process
                    slm_process.terminate()
                    slm_process.join()
                    
                    if ser_port.is_open and rs232_or_rs485.value == "rs485":
                        config_active = (':1CONF\r').encode()  # Ensure message ends with CRLF
                        ser_port.write(config_active)
                        ser_port.close()
                    
                    # Enable AP mode
                    logger.info("Enabling hotspot mode...")                    
                    enable_ap_mode()
                    logger.info("Hotspot mode enabled. Connect to Pi WiFi")
                    
                    # Start Flask server for config editing on fixed IP
                    hotspot_proc = subprocess.Popen([
                        'python3', 'slm/src/slm_hotspot_server.py', '--host', '192.168.4.1', '--port', '8080'
                    ])
                    logger.info("Hotspot server started. Connect to Pi WiFi and open http://192.168.4.1:8080")
                    # Wait until Switch 2 is pressed again to exit hotspot mode
                    while True:
                        time.sleep(0.5)
                        if GPIO.input(SWITCH2_PIN):
                            logger.info("Switch 2 pressed again - Exiting hotspot mode...")
                            break
                    # Stop Flask server
                    hotspot_proc.terminate()
                    hotspot_proc.wait()
                    
                    disable_ap_mode()
                    logger.info("Hotspot mode disabled.")
                    pressed_time = None
                    
            elif GPIO.input(SWITCH4_PIN):
                # Toggle the shared value
                if time_weighting_value.value == "fast":
                    time_weighting_value.value = "slow"
                    logger.info("Switched to slow time weighting.")
                    mode_active = (':1-SL-\r').encode()
                elif time_weighting_value.value == "slow":
                    time_weighting_value.value = "fast"
                    logger.info("Switched to fast time weighting.")
                    mode_active = (':1-FA-\r').encode()
                
                if ser_port.is_open and rs232_or_rs485.value == "rs485":
                    ser_port.write(mode_active)
                    time.sleep(1)  # Ensure message is sent
            else:
                pressed_time = None
            
            # Read GPIO state for RS232/RS485 switch
            current_switch_state = GPIO.input(SWITCH3_PIN)
            # Check RS232/RS485 switch state
            if current_switch_state != prev_switch_state:
                # Switch has changed
                if current_switch_state:
                    rs232_or_rs485.value = "rs232"
                    logger.info("RS232 mode selected.")
                    ser_port.write((':1r232\r').encode())
                else:
                    rs232_or_rs485.value = "rs485"
                    logger.info("RS485 mode selected.")
                    ser_port.write((':1r485\r').encode())
                    time.sleep(1)

                # Update previous state
                prev_switch_state = current_switch_state


    finally:
        GPIO.cleanup()
        if ser_port.is_open:
            ser_port.close()
        slm_process.terminate()
        slm_process.join()
