import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from shapely.geometry import Point
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType

# Configurações de Filtro
INDUSTRIAL_SITES = [
    {"name": "Kremenchuk", "lat": 49.12, "lon": 33.48},
    {"name": "Zaporizhzhia", "lat": 47.51, "lon": 34.58},
    {"name": "Odessa Port", "lat": 46.50, "lon": 30.73}
]
MIN_FRP = 30.0
STATE_FILE = "bot_state.json"

def get_config():
    config = SHConfig()
    config.sh_client_id = os.getenv('CDSE_CLIENT_ID')
    config.sh_client_secret = os.getenv('CDSE_CLIENT_SECRET')
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    return config

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"processed_fires": []}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def is_industrial(lat, lon):
    point = (lat, lon)
    for site in INDUSTRIAL_SITES:
        dist = ((lat - site['lat'])**2 + (lon - site['lon'])**2)**0.5 # Aproximação simples
        if dist < 0.02: # Aprox. 2km
            return True
    return False

def fetch_firms_data():
    api_key = os.getenv('NASA_API_KEY')
    # Bbox Ucrânia: 22, 44, 40, 52
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{api_key}/VIIRS_SNPP/UKR/1"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    
    df = pd.read_csv(requests.utils.io.StringIO(response.text))
    new_fires = []
    state = load_state()

    for _, row in df.iterrows():
        f_id = f"{row['latitude']}_{row['longitude']}_{row['acq_date']}"
        if row['frp'] > MIN_FRP and not is_industrial(row['latitude'], row['longitude']):
            if f_id not in state['processed_fires']:
                new_fires.append(row)
                state['processed_fires'].append(f_id)
    
    # Limitar histórico de estado para não crescer infinitamente
    state['processed_fires'] = state['processed_fires'][-500:]
    save_state(state)
    return new_fires

def get_satellite_image(lat, lon):
    config = get_config()
    # Criar uma BBox de 2km ao redor do ponto
    offset = 0.01 
    bbox = BBox(bbox=[lon-offset, lat-offset, lon+offset, lat+offset], crs=CRS.WGS84)
    
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B04", "B03", "B02"],
        output: { bands: 3 }
      };
    }
    function evaluatePixel(sample) {
      return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data_obj(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
            )
        ],
        responses=[SentinelHubRequest.output_obj('default', MimeType.PNG)],
        bbox=bbox,
        size=(800, 800),
        config=config
    )
    
    response = request.get_data()
    if response:
        with open("fire_spot.png", "wb") as f:
            f.write(response[0])
        return "fire_spot.png"
    return None

def send_telegram(caption, image_path):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    with open(image_path, 'rb') as photo:
        requests.post(url, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}, files={'photo': photo})

def main():
    fires = fetch_firms_data()
    for fire in fires[:3]: # Processar apenas os 3 mais recentes para não travar
        img = get_satellite_image(fire['latitude'], fire['longitude'])
        if img:
            caption = (
                f"🔥 *STRIKE ALERT - UKRAINE*\n\n"
                f"🎯 *Confidence:* {fire['confidence']}%\n"
                f"🚀 *FRP:* {fire['frp']}\n"
                f"📍 *Location:* `{fire['latitude']}, {fire['longitude']}`\n"
                f"🕒 *Time:* {fire['acq_date']} {fire['acq_time']} UTC\n\n"
                f"🔗 [Open Google Maps](https://www.google.com/maps?q={fire['latitude']},{fire['longitude']})"
            )
            send_telegram(caption, img)

if __name__ == "__main__":
    main()