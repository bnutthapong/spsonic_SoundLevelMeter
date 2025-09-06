import logging
logger = logging.getLogger(__name__)

import subprocess
from src.slm_hotspot_server import update_dietpi_wifi


def enable_ap_mode():
    logger.info("Disabling Wi-Fi client mode...")
    subprocess.run(["sudo", "systemctl", "stop", "wpa_supplicant"], check=True)
    subprocess.run(["sudo", "systemctl", "disable", "wpa_supplicant"], check=True)
    subprocess.run(["sudo", "killall", "wpa_supplicant"], check=True)

    logger.info("Enabling AP mode...")
    subprocess.run(["sudo", "ip", "addr", "flush", "dev", "wlan0"], check=True)
    subprocess.run(["sudo", "systemctl", "unmask", "hostapd"], check=True)
    subprocess.run(["sudo", "systemctl", "enable", "hostapd"], check=True)
    subprocess.run(["sudo", "systemctl", "enable", "dnsmasq"], check=True)
    subprocess.run(["sudo", "ip", "addr", "add", "192.168.4.1/24", "dev", "wlan0"], check=True)
    subprocess.run(["sudo", "ip", "link", "set", "wlan0", "up"], check=True)
    subprocess.run(["sudo", "systemctl", "start", "hostapd"], check=True)
    subprocess.run(["sudo", "systemctl", "start", "dnsmasq"], check=True)
    logger.info("AP mode enabled.")
    subprocess.run(["ip", "addr", "show", "wlan0"], check=True)

def disable_ap_mode():
    logger.info("Disabling AP mode...")
    subprocess.run(["sudo", "systemctl", "stop", "hostapd"], check=True)
    subprocess.run(["sudo", "systemctl", "stop", "dnsmasq"], check=True)
    subprocess.run(["sudo", "ip", "addr", "flush", "dev", "wlan0"], check=True)
    subprocess.run(["sudo", "systemctl", "disable", "hostapd"], check=True)
    subprocess.run(["sudo", "systemctl", "disable", "dnsmasq"], check=True)

    logger.info("Re-enabling Wi-Fi client mode...")
    subprocess.run(["sudo", "systemctl", "enable", "wpa_supplicant"], check=True)
    #subprocess.run(["sudo", "ip", "addr", "add", "192.168.1.79/24", "dev", "wlan0"], check=True) # no need to set IP manully
    subprocess.run(["sudo", "dhclient", "wlan0"], check=True)
    logger.info("Wi-Fi client mode enabled.")
    update_dietpi_wifi()
    subprocess.run(["ip", "addr", "show", "wlan0"], check=True)
    subprocess.run(["sudo", "reboot", "-n"], check=True)
    
