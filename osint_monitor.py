import os, json, requests, io
import pandas as pd
from datetime import datetime, timedelta, timezone
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType

# --- CONFIGURAÇÕES ---
INDUSTRIAL_SITES = [
    {"name": "Kremenchuk", "lat": 49.12, "lon": 33.48},
    {"name": "Zaporizhzhia", "lat": 47.51, "lon": 34.58},
    {"name": "Odessa Port", "lat": 46.50, "lon": 30.73}
]
MIN_FRP = 35.0  # Aumentado levemente para evitar alarmes falsos
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
    for site in INDUSTRIAL_SITES:
        dist = ((lat - site['lat'])**2 + (lon - site['lon'])**2)**0.5
        if dist < 0.015: # Reduzido para ~1.5km para ser mais preciso
            return True
    return False

def fetch_firms_data():
    api_key = os.getenv('NASA_API_KEY')
    # Pegando dados de 24h para garantir cobertura
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{api_key}/VIIRS_SNPP/UKR/1"
    
    try:
        response = requests.get(url, timeout=30)
        df = pd.read_csv(io.StringIO(response.text))
        new_fires = []
        state = load_state()
        
        # Ordenar por FRP (mais potentes primeiro)
        df = df.sort_values(by='frp', ascending=False)

        for _, row in df.iterrows():
            f_id = f"{row['latitude']}_{row['longitude']}" # ID simplificado para evitar o mesmo ponto em horas diferentes
            
            if row['frp'] >= MIN_FRP and not is_industrial(row['latitude'], row['longitude']):
                if f_id not in state['processed_fires']:
                    new_fires.append(row)
                    state['processed_fires'].append(f_id)
        
        state['processed_fires'] = state['processed_fires'][-300:] # Mantém histórico recente
        save_state(state)
        return new_fires
    except Exception as e:
        print(f"Erro FIRMS: {e}")
        return []

def get_satellite_image(lat, lon):
    try:
        config = get_config()
        offset = 0.012 # Zoom levemente maior
        bbox = BBox(bbox=[lon-offset, lat-offset, lon+offset, lat+offset], crs=CRS.WGS84)
        
        # EVALSCRIPT MELHORADO: B12 (SWIR) destaca calor, B08 (NIR) destaca vegetação, B04 (Red)
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B12", "B08", "B04"],
            output: { bands: 3 }
          };
        }
        function evaluatePixel(sample) {
          // Se o SWIR (B12) for muito alto, pinta de laranja/vermelho brilhante (fogo)
          return [sample.B12 * 2.5, sample.B08 * 1.5, sample.B04 * 1.5];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data_obj(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(datetime.now(timezone.utc) - timedelta(days=3), datetime.now(timezone.utc)),
                    maxcc=0.4 # Filtro: Ignora se houver mais de 40% de nuvens
                )
            ],
            responses=[SentinelHubRequest.output_obj('default', MimeType.PNG)],
            bbox=bbox,
            size=(800, 800),
            config=config
        )
        
        response = request.get_data()
        if response:
            path = "fire_spot.png"
            with open(path, "wb") as f: f.write(response[0])
            return path
    except: return None
    return None

def send_telegram(caption, image_path):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    with open(image_path, 'rb') as photo:
        requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", 
                      data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}, 
                      files={'photo': photo})

def main():
    fires = fetch_firms_data()
    for _, fire in pd.DataFrame(fires).head(2).iterrows(): # Processa os 2 mais importantes
        img = get_satellite_image(fire['latitude'], fire['longitude'])
        if img:
            caption = (
                f"🔥 *POTENTIAL STRIKE DETECTED*\n\n"
                f"🚀 *Power (FRP):* `{fire['frp']}`\n"
                f"📍 *Coords:* `{fire['latitude']}, {fire['longitude']}`\n"
                f"🛰️ *Sensor:* VIIRS (NASA)\n\n"
                f"🔗 [View on Google Maps](https://www.google.com/maps?q={fire['latitude']},{fire['longitude']})"
            )
            send_telegram(caption, img)

if __name__ == "__main__":
    main()
