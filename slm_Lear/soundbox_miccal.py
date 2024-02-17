import numpy as np
import soundBox_acoustic
import sqlite3
from acoustics.signal import third_octaves
import math
import os

freq_dict_change = {'10 Hz': 0, '12.5 Hz': 1, '16 Hz': 2, '20 Hz': 3,
                    '25 Hz': 4, '30 Hz': 5, '40 Hz': 6, '50 Hz': 7,
                    '63 Hz': 8, '80 Hz': 9, '100 Hz': 10, '125 Hz': 11,
                    '160 Hz': 12, '200 Hz': 13, '250 Hz': 14, '315 Hz': 15, '400 Hz': 16,
                    '500 Hz': 17, '630 Hz': 18, '800 Hz': 19, '1 KHz': 20, '1.25 KHz': 21,
                    '1.6 KHz': 22, '2 KHz': 23, '2.5 KHz': 24, '3.1 KHz': 25, '4 KHz': 26,
                    '5 KHz': 27, '6.3 KHz': 28, '8 KHz': 29, '10 KHz': 30, '12.5 KHz': 31,
                    '16 KHz': 32, '20 KHz': 33
                    }
freq_dict_change_rev = {0: '10 Hz', 1: '12.5 Hz', 2: '16 Hz', 3: '20 Hz',
                        4: '25 Hz', 5: '30 Hz', 6: '40 Hz', 7: '50 Hz',
                        8: '63 Hz', 9: '80 Hz', 10: '100 Hz', 11: '125 Hz',
                        12: '160 Hz', 13: '200 Hz', 14: '250 Hz', 15: '315 Hz', 16: '400 Hz',
                        17: '500 Hz', 18: '630 Hz', 19: '800 Hz', 20: '1 KHz', 21: '1.25 KHz',
                        22: '1.6 KHz', 23: '2 KHz', 24: '2.5 KHz', 25: '3.1 KHz', 26: '4 KHz',
                        27: '5 KHz', 28: '6.3 KHz', 29: '8 KHz', 30: '10 KHz', 31: '12.5 KHz',
                        32: '16 KHz', 33: '20 KHz'
                        }
freq_s = ['6', '8', '10', '12.5', '16', '20', '25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250',
          '315', '400',
          '500', '630', '800', '1k', '1.25k', '1.6k', '2k', '2.5k', '3.1k', '4k', '5k', '6.3k', '8k', '10k', '12.5k',
          '16k', '20k']
freq_dict = dict(enumerate(freq_s))
x_temp = list(freq_dict.keys())

class soundBox_micCal():
    def __init__(self, parent=None):
        # self.Freq_combobox.currentIndexChanged.connect(self.selectionFreq)
        # self.freq_select = "10 Hz"
        self.sound = soundBox_acoustic.soundbox_acounstic(rate=44400, updatesPerSecond=10)
        self.sound.stream_start()
        self.calibration = 0
        self.calibration_level = 0
        self.miccal_db = sqlite3.connect("/home/pi/SPsonic/db/mic_cal.db")

        miccal_db = sqlite3.connect("/home/pi/SPsonic/db/mic_cal.db")
        result_for_check = miccal_db.execute("SELECT * FROM cal_mic_table")
        ref_check = result_for_check.fetchall()
        ref_pres = ref_check[0]
        self.ref = ref_pres[0]
        if self.ref is None:
            self.ref = 2.0e-5
        print(self.ref)

    def mic_cal_top(self, db_ref):
        self.printProgressBar(0, 100, prefix = 'Progress:', suffix = ' ', length = 50)
        while self.calibration != "Finish":
            self.mic_cal_dep("1 Khz", db_ref)
        self.finish_fcn()


    def mic_cal_dep(self, freq, db_ref):
        self.db_ref = float(db_ref)
        if not self.sound.data is None and not self.sound.fft is None:
            _, self.input_octave = self.get_onethird(self.sound.data, self.sound.rate, ref=self.ref)
            if freq == "250 Hz":
                self.spl_data_now = self.input_octave[14]
            else:
                self.spl_data_now = self.input_octave[20]
            # self.input_octave = self.input_octave + int(self.db_ref)
            if self.spl_data_now != (self.db_ref):
                if self.spl_data_now <= (self.db_ref):
                    # self.ref = self.ref - 0.0000001
                    self.ref = self.ref - 0.000050
                else:
                    # self.ref = self.ref + 0.0000001
                    self.ref = self.ref + 0.000050
                #print(self.spl_data_now)
                #print("Calibrating ...")
                val = int(100 - (100 * (abs(self.db_ref - int(self.spl_data_now))) / self.db_ref))
                self.printProgressBar(int(val), 100, prefix = 'Progress:', suffix = ' ', length = 50)
                
            string = "SPL now = " + str(self.spl_data_now)
            #print(string)
            # self.spl_now.setText(string)

            if np.abs((self.spl_data_now) - (self.db_ref)) < 0.05 :
                string = "SPL now = " + str(self.spl_data_now) + " Finish calibration"
                #print(string)
                self.printProgressBar(100, 100, prefix = 'Progress:', suffix = 'Complete', length = 50)
                self.calibration = "Finish"

                
    def finish_fcn(self):
        if self.calibration == "Finish":
            self.miccal_db.execute("UPDATE cal_mic_table set `ref_pres`=?", (self.ref,))
            self.miccal_db.commit()
            #print("update db")

            

    def get_onethird(self, data, rate, ref):
        freq, sig = third_octaves(data, fs=rate, ref=ref)
        #print("Hello MIcCal getonethird")
        return np.round(freq.center), sig
        
    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '¦', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
   
         
    
    

if __name__ == "__main__":
    form = soundBox_micCal()
    import os
    os.system('clear')
    db_ref = input('Enter Reference dB: ')
    form.mic_cal_top(db_ref)
    os._exit(0)
