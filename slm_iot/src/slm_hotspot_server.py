from flask import Flask, render_template_string, request, redirect
import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'slm_config.json')
app = Flask(__name__)

HTML_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SLM Configuration</title>
  <style>
    body {
        font-family: Arial;
        margin: 40px;
        max-width: 600px;
    }

    form {
        display: flex;
        flex-direction: column;
    }

    .form-group {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }

    .form-group label {
        width: 150px;
        font-weight: bold;
    }

    .form-group input[type="text"],
    .form-group input[type="number"] {
        flex: 1;
        padding: 8px;
        font-size: 14px;
    }

    button {
        align-self: flex-start;
        margin-top: 20px;
        padding: 10px 20px;
        font-size: 14px;
    }
  </style>
</head>
<body>
  <h2>SLM 1.0 Configuration</h2>
<form method="POST" action="/">
  <div class="form-group">
    <label for="node_id">Node ID:</label>
    <input type="text" name="node_id" id="node_id" value="{{ config.get('node_id', '') }}" required>
  </div>

  <div class="form-group">
    <label for="wifi_ssid">Wi-Fi SSID:</label>
    <input type="text" name="wifi_ssid" id="wifi_ssid" value="{{ config.get('wifi_ssid', '') }}" >
  </div>

  <div class="form-group">
    <label for="wifi_password">Wi-Fi Password:</label>
    <input type="text" name="wifi_password" id="wifi_password" value="{{ config.get('wifi_password', '') }}" >
  </div>

  <div class="form-group">
    <label for="mqtt_broker">MQTT Broker IP:</label>
    <input type="text" name="mqtt_broker" id="mqtt_broker" value="{{ config.get('mqtt_broker', '') }}" >
  </div>

  <div class="form-group">
    <label for="mqtt_port">MQTT Port:</label>
    <input type="number" name="mqtt_port" id="mqtt_port" value="{{ config.get('mqtt_port', 1883) }}" >
  </div>

  <div class="form-group">
    <label for="http_endpoint">HTTP Endpoint URL:</label>
    <input type="text" name="http_endpoint" id="http_endpoint" value="{{ config.get('http_endpoint', '') }}" >
  </div>

  <button type="submit">Save Configuration</button>
</form>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def edit_config():
    default_config = {
        "node_id": "",
        "wifi_ssid": "",
        "wifi_password": "",
        "mqtt_broker": "",
        "mqtt_port": 1883,
        "http_endpoint": ""
    }

    # Ensure config directory exists
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Create config file if missing
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=2)

    # Load config
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # Handle form submission
    if request.method == 'POST':
        for key in default_config.keys():
            value = request.form.get(key)
            if key == "mqtt_port":
                try:
                    config[key] = int(value)
                except ValueError:
                    config[key] = default_config["mqtt_port"]
            else:
                config[key] = value
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return redirect('/')

    return render_template_string(HTML_FORM, config=config)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
