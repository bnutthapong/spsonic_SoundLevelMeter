#!/usr/bin/env python
import os, errno
import pyaudio
import spl_A_filter as spl
from scipy.signal import lfilter
import numpy
import csv
import time

import RPi.GPIO as GPIO
slidePin_CAL = 17

from signal import signal, SIGTERM, SIGHUP, pause
from rpi_lcd import LCD
lcd = LCD()

''' The following is similar to a basic CD quality
   When CHUNK size is 4096 it routinely throws an IOError.
   When it is set to 8192 it doesn't.
   IOError happens due to the small CHUNK size

   What is CHUNK? Let's say CHUNK = 4096
   math.pow(2, 12) => RATE / CHUNK = 100ms = 0.1 sec
'''
CHUNKS = [4096, 9600]       # Use what you need
CHUNK = CHUNKS[1]
FORMAT = pyaudio.paInt16    # 16 bit
CHANNEL = 1    # 1 means mono. If stereo, put 2

'''
Different mics have different rates.
For example, Logitech HD 720p has rate 48000Hz
'''
RATES = [44300, 48000]
RATE = RATES[1]

NUMERATOR, DENOMINATOR = spl.A_weighting(RATE)


'''
Listen from microphone
'''
pa = pyaudio.PyAudio()

stream = pa.open(format = FORMAT,
                channels = CHANNEL,
                rate = RATE,
                input = True,
                frames_per_buffer = CHUNK)


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(slidePin_CAL, GPIO.IN)
                                    
def listen_calibrate(old=0, error_count=0, min_decibel=100, max_decibel=0, ref=94):
    #lcd.clear()
    while True:
        try:
            ## read() returns string. You need to decode it into an array later.
            block = stream.read(CHUNK)
        except IOError as e:
            error_count += 1
            print(" (%d) Error recording: %s" % (error_count, e))
        else:
            ## Int16 is a numpy data type which is Integer (-32768 to 32767)
            ## If you put Int8 or Int32, the result numbers will be ridiculous
            decoded_block = numpy.fromstring(block, 'Int16')
            
            ## This is where you apply A-weighted filter
            y = lfilter(NUMERATOR, DENOMINATOR, decoded_block)
            new_decibel = 20*numpy.log10(spl.rms_flat(y))
            if new_decibel != ref:
                os.system('clear')
                print("A")
                diff = new_decibel - int(ref)

                f = open('cal_param.csv', 'w')
                writer = csv.writer(f)
                writer.writerow([diff])
                f.close()
                break          

    os.system('clear')
    print('A-weighted: {:+.2f} dB'.format(new_decibel))


    stream.stop_stream()
    stream.close()
    pa.terminate()   


if __name__ == '__main__':
    os.system('clear')
    setup()
    lcd.clear()
    ref = 94
    '''
    if GPIO.input(slidePin_FREQ) == 1:
        ref = 94
        print("At " + str(ref) + " dB")
    else:
        ref = 114
        print("At " + str(ref) + " dB")
    '''
    #ref = Keptkey
    
    for i in range(10,0,-1):
      print(f"Prepare {i} sec", end="\r", flush=True)
      time.sleep(1)
      lcd.text(f"Prepare {i} sec", 1)
    
    listen_calibrate(ref = ref)
    lcd.clear()
    lcd.text("CAL at 94.0dB", 1)
    lcd.text("Done...", 2)
    
    
    while True:
      if GPIO.input(slidePin_CAL) == 0:
          print("CALIBRATE OFF")
          break
      else:
          print("CAL at 94.0dB")
          lcd.clear()
          lcd.text("CAL at 94.0dB", 1)
          lcd.text("Done...", 2)
          time.sleep(3)
          
    os.system('python3 /home/pi/SPsonic_SPL/spl_app.py') # get full path
    #os.system('sudo shutdown -r now')

