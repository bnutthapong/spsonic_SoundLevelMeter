import time
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont
import logging
logger = logging.getLogger(__name__)

# Initialize OLED once
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial)

# Load fonts
try:
    font_small = ImageFont.truetype("DejaVuSans.ttf", 10)
    font_medium = ImageFont.truetype("DejaVuSans.ttf", 12)
    font_large = ImageFont.truetype("DejaVuSans.ttf", 20)
except:
    logger.warning("DejaVuSans.ttf not found. Using default font.")
    font_small = font_medium = font_large = ImageFont.load_default()


# Wi-Fi symbol
def draw_wifi_icon(draw, x=0, y=0, scale=0.5):
    """
    Draw a small Wi-Fi symbol at (x, y)
    scale <1 makes it smaller
    """
    # Dot
    dot_radius = int(2 * scale)
    draw.ellipse([x, y + int(8*scale), x + 2*dot_radius, y + int(8*scale) + 2*dot_radius], fill=255)

    # Three arcs
    for i in range(3):
        radius = int((i + 1) * 6 * scale)
        bbox = [x - radius + dot_radius, y - radius + int(8*scale) + dot_radius,
                x + radius + dot_radius, y + radius + int(8*scale) + dot_radius]
        draw.arc(bbox, start=200, end=340, fill=255, width=1)


def display_slm(wifi=True, mode="SLOW", SPLA=0, Lmin="-", Lmax="-", Leq="-"):
    # Create blank image
    image = Image.new("1", (device.width, device.height), "black")
    draw = ImageDraw.Draw(image)

    # --- Line 1: Wi-Fi + Centered title ---
    if wifi:
        draw_wifi_icon(draw, x=8, y=5, scale=0.3)

    title_text = "SONITA SLM"
    bbox = draw.textbbox((5,5), title_text, font=font_small)
    w = bbox[2] - bbox[0]
    x_center = (device.width - w)//2
    draw.text((x_center, 0), title_text, font=font_small, fill=255)

    # ---------- Line 2: Mode ----------
    mode_text = mode
    bbox = draw.textbbox((0,0), mode_text, font=font_medium)
    w = bbox[2] - bbox[0]
    draw.text(((device.width - w)//2, 14), mode_text, font=font_medium, fill=255)

    # ---------- Line 3: SPLA dBA ----------
    spl_text = f"{SPLA:.1f} dBA" if isinstance(SPLA, (int, float)) else str(SPLA)
    bbox = draw.textbbox((0,0), spl_text, font=font_large)
    w = bbox[2] - bbox[0]
    draw.text(((device.width - w)//2, 28), spl_text, font=font_large, fill=255)

    # ---------- Line 4: Lmin, Lmax, Leq ----------
    # Replace None or empty with '-'
    lmin_text = f"{Lmin:.1f}" if isinstance(Lmin, (int, float)) else str(Lmin)
    lmax_text = f"{Lmax:.1f}" if isinstance(Lmax, (int, float)) else str(Lmax)
    leq_text   = f"{Leq:.1f}" if isinstance(Leq, (int, float)) else str(Leq)
    # l_text = f"Lmin:{lmin_text} Lmax:{lmax_text} Leq:{leq_text}"
    l_text = f"Lmin:{lmin_text} Lmax:{lmax_text}"

    bbox = draw.textbbox((0,0), l_text, font=font_small)
    w = bbox[2] - bbox[0]
    draw.text(((device.width - w)//2, 54), l_text, font=font_small, fill=255)

    # Display on OLED
    device.display(image)
    time.sleep(0.5)  # Small delay to ensure update


def display_calibration(countdown=3, wifi=True):
    """
    Display calibration screen on OLED.
    countdown: integer 3, 2, or 1
    """
    # Blank image
    image = Image.new("1", (device.width, device.height), "black")
    draw = ImageDraw.Draw(image)

    # ---------- Line 1: Wi-Fi + title ----------
    if wifi:
        draw_wifi_icon(draw, x=8, y=5, scale=0.3)

    title_text = "SONITA SLM"
    bbox = draw.textbbox((0,0), title_text, font=font_small)
    w = bbox[2] - bbox[0]
    draw.text(((device.width - w)//2, 0), title_text, font=font_small, fill=255)

    # ---------- Line 2: Calibration instruction ----------
    calib_text = "Calibrate 94dB at 1kHz"
    bbox = draw.textbbox((0,0), calib_text, font=font_small)
    w = bbox[2] - bbox[0]
    draw.text(((device.width - w)//2, 18), calib_text, font=font_small, fill=255)

    # ---------- Line 3: Countdown ----------
    if countdown == -1:
        countdown = "Rec"
    elif countdown == -2:    
        countdown = "Done"
        
    countdown_text = str(countdown)
    bbox = draw.textbbox((0,0), countdown_text, font=font_large)
    w = bbox[2] - bbox[0]
    draw.text(((device.width - w)//2, 38), countdown_text, font=font_large, fill=255)

    # Display on OLED
    device.display(image)
    

def display_reboot(wifi=True):
    """
    Display Reboot screen on OLED.
    """
    # Blank image
    image = Image.new("1", (device.width, device.height), "black")
    draw = ImageDraw.Draw(image)

    # ---------- Line 1: Wi-Fi + title ----------
    if wifi:
        draw_wifi_icon(draw, x=8, y=5, scale=0.3)

    title_text = "SONITA SLM"
    bbox = draw.textbbox((0,0), title_text, font=font_small)
    w = bbox[2] - bbox[0]
    draw.text(((device.width - w)//2, 0), title_text, font=font_small, fill=255)

    # ---------- Line 2: Calibration instruction ----------
    calib_text = "Rebooting..."
    bbox = draw.textbbox((0,0), calib_text, font=font_small)
    w = bbox[2] - bbox[0]
    draw.text(((device.width - w)//2, 18), calib_text, font=font_small, fill=255)

    # Display on OLED
    device.display(image)