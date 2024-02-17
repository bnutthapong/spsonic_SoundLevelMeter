#!/usr/bin/env python
import os, errno
import numpy
import csv
import time
import serial
import RPi.GPIO as GPIO
import spl_measure
import sys

sys.path.append('/usr/lib/python3/dist-packages')
sys.path.append('./.local/lib/python3.9/site-packages')

sound = spl_measure.sound_measure(rate=44100, updatesPerSecond=10)
sound.stream_start()
os.system('sudo chmod 666 /dev/ttyS0')

from signal import signal, SIGTERM, SIGHUP, pause
#from rpi_lcd import LCD
#lcd = LCD()

## calibrate switch
slidePin_CAL = 17

## calibration reference
with open("/home/pi/SPsonic_SPL/cal_param.csv") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        #print ('line[{}] = {}'.format(i, line))  #Check 1st row in .csv
        if i == 0:
            diff = line[0]
            
def is_meaningful(old, new):
    return abs(old - new) > 0


# Define a setup function for some setup
def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(slidePin_CAL, GPIO.IN)

def destroy():
    # Release resource
    GPIO.cleanup()

def safe_exit(signum, frame):
    exit(1)


def stream_readchunk():
    try:
        ## read() returns string. You need to decode it into an array later.
        block = stream.read(CHUNK)
    except KeyboardInterrupt:
        #lcd.clear()
        stream.stop_stream()
        stream.close()
        pa.terminate()  


def listen(old=0, error_count=0, min_decibel=100, max_decibel=0):
    ser=serial.Serial("/dev/ttyS0",baudrate = 115200,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS,timeout=1)
    os.system('clear')
#    lcd.clear()
    print("--- Listen Starting ---")
    print("If you want to exit, please press Ctrl+c")
    CAL_On = False
    
    while True:

        ## read() returns string. You need to decode it into an array later.        
        block = sound.SPL_A
        #block = 100
        ## calculation factor for decibel A 
        new_decibel = block - float(diff)
        
        #print(block)
        
        dbA = "SPL: {:.1f}".format(new_decibel)
        send_dBA = "{:.1f}".format(new_decibel)
        #print(dbA +" dBA")

        signal(SIGTERM, safe_exit)
        signal(SIGHUP, safe_exit) 
        
        
        try:
          #lcd.text(str(dbA) +" dBA", 1)
          print(dbA +" dBA")
          
        except KeyboardInterrupt:
          print(" -- stop running : Keyboard Interrupt")
          #lcd.clear()
          #lcd.text("PROGRAM STOPPED", 1)
          #lcd.text("BY USER", 2)
          sound.close()
          break
            
        
        if GPIO.input(slidePin_CAL) == 1:
            #print("CALIBRATE OFF")
            CAL_On = False
            pass
        else:
            print("CAL at 94.0dB")
            #lcd.clear()
            #lcd.text("CAL at 94.0dB", 1)
            CAL_On = True
          
                      
        #Serial send to external
        if new_decibel < 10:
            x = "S00" + str(send_dBA) + "\r\n"
        elif new_decibel > 100 :
            x = "S" + str(send_dBA) + "\r\n"
        else:
            x = "S0" + str(send_dBA) + "\r\n"
            
        ser.write(str.encode(x))
        ser.flushOutput()
        time.sleep(0.09)
        
        
        #ser.write([ord(c) for c in x])
        #time.sleep(0.015)
        
        ## Check calibration switch is on
        if CAL_On:
           os.system('python3 /home/pi/SPsonic_SPL/spl_meter_cal.py') # get full path
           CAL_On = False
          
    #print("Run MIC cal") 
    

if __name__ == '__main__':
    setup()
    listen()

