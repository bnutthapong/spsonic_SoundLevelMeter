import pyaudio, time, threading, audioop
import datetime
import numpy as np
from acoustics.signal import third_octaves
import sqlite3
import serial

A_scale_weight_db = [-70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1,
                     -13.4, -10.9,
                     -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6, 1, 1.2, 1.3, 1.2, 1, 0.5, -0.1, -1.1, -2.5, -4.3, -6.6,
                     -9.3]

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

def get_onethird_cal(data, rate, ref):
    if ref is None:
        ref = 2.0e-5
    freq, sig = third_octaves(data, fs=rate, ref=ref)

def get_onethird(data, rate):
    miccal_db = sqlite3.connect("/home/pi/SPsonic/db/mic_cal.db")
    result_for_check = miccal_db.execute("SELECT * FROM cal_mic_table")
    ref_check = result_for_check.fetchall()
    ref_pres = ref_check[0]
    ref = ref_pres

    if ref is None:
        ref = 2.0e-5
    freq, sig = third_octaves(data, fs=rate, ref=ref[0])
    return np.round(freq.center), (np.round(sig / 1)), sig

def getFFT(data,rate):
    """Given some data and rate, returns FFTfreq and FFT (half)."""
    data=data*np.hamming(len(data))
    fft=np.fft.fft(data)
    fft=np.abs(fft)
    freq=np.fft.fftfreq(len(fft),1.0/rate)
    return freq[:int(len(freq)/2)],fft[:int(len(fft)/2)]

class soundbox_acounstic():
    """
    The SWHear class is provides access to continuously recorded
    (and mathematically processed) microphone data.

    Arguments:

        device - the number of the sound card input to use. Leave blank
        to automatically detect one.

        rate - sample rate to use. Defaults to something supported.

        updatesPerSecond - how fast to record new data. Note that smaller
        numbers allow more data to be accessed and therefore high
        frequencies to be analyzed if using a FFT later
    """

    def __init__(self, device=None, rate=None, updatesPerSecond=10):
        self.p = pyaudio.PyAudio()
        self.chunk = 65536  # gets replaced automatically
        self.updatesPerSecond = updatesPerSecond
        self.chunksRead = 0
        self.device = device
        self.rate = rate
        self.datax = np.arange(self.chunk) / float(self.rate)
        self.info = 0
        self.ref = 2.0e-5
        self.spl_meter = 0
        self.splDeci = "0.0"


    def valid_low_rate(self, device):
        """set the rate to the lowest supported audio rate."""
        for testrate in [44100]:
            if self.valid_test(device, testrate):
                return testrate
        print("SOMETHING'S WRONG! I can't figure out how to use DEV", device)
        return None

    def valid_test(self, device, rate=44100):
        """given a device ID and a rate, return TRUE/False if it's valid."""
        try:
            self.info = self.p.get_device_info_by_index(device)
            if not self.info["maxInputChannels"] > 0:
                return False
            stream = self.p.open(format=pyaudio.paInt16, channels=1,
                                 input_device_index=device, frames_per_buffer=self.chunk,
                                 rate=int(self.info["defaultSampleRate"]), input=True)
            stream.close()
            return True
        except:
            return False

    def valid_input_devices(self):
        """
        See which devices can be opened for microphone input.
        call this when no PyAudio object is loaded.
        """
        mics = []
        for device in range(self.p.get_device_count()):
            if self.valid_test(device):
                mics.append(device)
        if len(mics) == 0:
            print("no microphone devices found!")
        else:
            print("found %d microphone devices: %s" % (len(mics), mics))
        return mics

    ### SETUP AND SHUTDOWN
    def initiate(self):
        """run this after changing settings (like rate) before recording"""
        if self.device is None:
            self.device = self.valid_input_devices()[0]  # pick the first one
        if self.rate is None:
            self.rate = self.valid_low_rate(self.device)
        self.chunk = int(self.rate / self.updatesPerSecond)  # hold one tenth of a second in memory
        if not self.valid_test(self.device, self.rate):
            print("guessing a valid microphone device/rate...")
            self.device = self.valid_input_devices()[0]  # pick the first one
            self.rate = self.valid_low_rate(self.device)
        self.datax = np.arange(self.chunk) / float(self.rate)
        msg = 'recording from "%s" ' % self.info["name"]
        msg += '(device %d) ' % self.device
        msg += 'at %d Hz' % self.rate
        #print(msg)


    def stream_readchunk(self):
        """reads some audio and re-launches itself"""
        try:
            datetime_object = datetime.datetime.now()
            datetime_object = datetime_object.strftime("%S")
            self.data = np.frombuffer(self.stream.read(self.chunk,exception_on_overflow = False), dtype=np.int16)
            #self.data2 = moving_average(self.data, 100)
            _, self.int_fft, self.fft = get_onethird(self.data, self.rate)
            #print(self.fft)
            self.spl_temp = 0
            for k in range(0, 34):
                self.spl_temp = self.spl_temp + int(10 ** ((self.fft[k] + A_scale_weight_db[k] )/ 10))
            self.spl_meter = 10 * np.log10(self.spl_temp)
            self.splDeci = '%.1f'%(self.spl_meter)
            #print(self.splDeci)
            self.chunksRead += 1
              
        except Exception as E:
            print(" -- exception! terminating... --")
            print(E, "\n" * 5)
            self.keepRecording = False
        if self.keepRecording:
            # print(self.p)
            self.stream_thread_new()
        else:
            self.stream.close()
            self.p.terminate()
            # print(self.p)
            print(" -- stream STOP --")
        

    def stream_thread_new(self):
        self.t = threading.Thread(target=self.stream_readchunk)
        self.t.start()

    def stream_start(self):
        """adds data to self.data until termination signal"""
        self.initiate()
        print(" -- starting stream -- ")
        self.keepRecording = True  # set this to False later to terminate stream
        self.data = None  # will fill up with threaded recording data
        self.fft = None
        self.dataFiltered = None  # same
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1,
                                  rate=self.rate, input=True, frames_per_buffer=self.chunk)
                                  #stream_callback=self.callback)
        self.stream_thread_new()


if __name__ == "__main__":
    ear = soundbox_acounstic(rate=96000, updatesPerSecond=10)  # optinoally set sample rate here
    ear.stream_start()  # goes forever
    lastRead = ear.chunksRead
    while True:
        while lastRead == ear.chunksRead:
          time.sleep(.01)
        #print(ear.chunksRead, ear.fft)

        lastRead = ear.chunksRead
    print("DONE")
