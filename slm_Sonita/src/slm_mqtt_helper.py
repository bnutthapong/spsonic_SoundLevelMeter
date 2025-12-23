import os
import json
import logging
import paho.mqtt.client as mqtt
import ssl

logger = logging.getLogger(__name__)

def setup_mqtt():
    try:
        mqtt_cfg = read_mqtt_config()  # e.g., {'broker': 'broker.hivemq.com', 'port': 1883, 'topic': 'slm/topic', 'node_id': 'slm_node'}
        logger.info(f"Connecting to: {mqtt_cfg['broker']}")
        mqtt_client = mqtt.Client(
                    client_id=mqtt_cfg["node_id"],
                    protocol=mqtt.MQTTv311,
                    transport="tcp",
                    userdata=None,
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
                )

        # Authentication
        mqtt_client.username_pw_set(mqtt_cfg["username"], mqtt_cfg["password"])
        
        # Secure TLS connection
        mqtt_client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
        mqtt_client.tls_insecure_set(False)

        # Connect and start network loop
        mqtt_client.connect(mqtt_cfg["broker"], mqtt_cfg["port"], keepalive=60)
        mqtt_client.loop_start()

    except Exception as e:
        logger.warning(f"MQTT setup failed: {e}")
        mqtt_client = None
        mqtt_cfg = None
    
    return mqtt_client, mqtt_cfg


def read_mqtt_config() -> dict:
    try:
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        config_path = os.path.join(config_dir, 'slm_config.json')
        with open(config_path, "r") as f:
            cfg = json.load(f)
            
        # Basic validation + defaults
        mqtt_broker = cfg.get("mqtt_broker", "").strip()
        mqtt_port = int(cfg.get("mqtt_port", 1883))
        mqtt_topic = cfg.get("mqtt_topic", "").strip()
        mqtt_username = cfg.get("mqtt_username", "").strip()
        mqtt_password = cfg.get("mqtt_password", "").strip()
        if not mqtt_broker:
            raise ValueError("mqtt_broker is required in slm_config.json")
        return {
            "broker": mqtt_broker,
            "port": mqtt_port,
            "topic": mqtt_topic,
            "username": mqtt_username,
            "password": mqtt_password,
            "node_id": cfg.get("node_id", "slm_node"),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to read MQTT config: {e}") from e


def publish_leq(timestamp, mqtt_client, mqtt_topic, node_id, leq_val, lmax_val, lmin_val, l90_val, spl_current):
    """
    Publish SPL/LEQ values to MQTT broker in JSON format.

    Parameters:
    - mqtt_client : paho.mqtt.client.Client
    - mqtt_topic  : str, topic to publish
    - node_id     : str, unique node identifier
    - leq_val     : float, Leq value
    - lmax_val    : float, Lmax value
    - lmin_val    : float, Lmin value
    - l90_val     : float, L90 value
    - spl_current : float, current SPL (A)
    """
    if mqtt_client is None:
        logger.warning("MQTT client not initialized. Skipping publish.")
        return

    payload = {
        "TimeStamp": timestamp,
        "node": node_id,
        "Leq": leq_val,
        "Lmax": lmax_val,
        "Lmin": lmin_val,
        "L90": l90_val,
        "SPL": spl_current
    }
    
    qos = 1  # Quality of Service level (0, 1, or 2)

    try:
        # Publish the message
        result = mqtt_client.publish(mqtt_topic, json.dumps(payload), qos=qos)
        
        # Optional: check result
        status = result[0]
        if status == mqtt.MQTT_ERR_SUCCESS:
            logger.info(f"Published MQTT payload to {mqtt_topic}: {payload}")
        else:
            logger.info(f"Failed to send message to topic {topic}")
            
    except Exception as e:
        logger.exception(f"Failed to publish MQTT payload: {e}")

def test_publish():
    try:
        cfg = read_mqtt_config()
        print("Connecting to:", cfg["broker"])

        client = mqtt.Client(
            client_id=cfg["node_id"],
            protocol=mqtt.MQTTv311,
            transport="tcp",
            userdata=None,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )

        # Authentication
        client.username_pw_set(cfg["username"], cfg["password"])
        print("Username:", cfg["username"], cfg["password"])
        
        # Secure TLS connection
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
        client.tls_insecure_set(False)

        # Connect and start network loop
        client.connect(cfg["broker"], cfg["port"], keepalive=60)
        client.loop_start()
        
        topic = cfg["topic"]  # e.g., "slm/data"
        payload = '{"level": 75.5, "unit": "dB"}'
        qos = 0  # Quality of Service level (0, 1, or 2)

        # Publish the message
        result = client.publish(topic, payload, qos=qos)

        # Optional: check result
        status = result[0]
        if status == mqtt.MQTT_ERR_SUCCESS:
            print(f"Message sent to topic {topic}")
        else:
            print(f"Failed to send message to topic {topic}")

    except RuntimeError as err:
        print("Runtime error:", err)
    except Exception as e:
        print("Unexpected error:", e)

if __name__ == "__main__":
    test_publish()
