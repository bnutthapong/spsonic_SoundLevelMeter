# Constants
REF_PRESSURE = 20e-6  # Reference pressure in pascals
LEQ_INTERVAL = 60     # Seconds (can be configured)
SAMPLE_RATE = 48000
CHUNK_SIZE = 4096  # Number of samples per frame
TIME_WEIGHTING = "fast"  # Options: "fast", "slow", "none"
REF_DB = 94

ACTIVE_CALIBRATION_GAIN = 25.0000  # Default calibration gain for microphone UMIK-2

error_counter = 0
silent_frame_counter = 0
error_threshold = 10  # Number of consecutive silent or bad frames before abort
optionCAL = 0  # 0: FFT, 1: Time Domain