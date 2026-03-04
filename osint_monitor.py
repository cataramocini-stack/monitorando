import os
import json
import requests
import pandas as pd
import io
from datetime import datetime, timedelta, timezone
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType

# --- CONFIGURAÇÕES ---
INDUSTRIAL_SITES = [
    {"name": "Kremenchuk", "lat": 49.12, "lon": 33.48},
    {"name": "Zaporizhzhia", "lat": 47.51, "lon": 34.58},
    {"name": "Odessa Port", "lat": 46.50, "lon": 30.73}
]
MIN_FRP = 30.0  # Filtro de potência do fogo
STATE_FILE = "bot_state.json"

def get_config():
    config = SHConfig()
    config.sh_client_id = os.getenv('CDSE_CLIENT_ID')
    config.sh_client_secret = os.getenv('CDSE_CLIENT_SECRET')
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    return config

def load_state():
    default_state = {"processed_fires": []}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                if not isinstance(state, dict) or "processed_fires" not in state:
                    return default_state
                return state
        except Exception as e:
            print(f"⚠️ Erro ao ler estado: {e}")
            return default_state
    return default_state

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        print("💾 Estado salvo com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao salvar estado: {e}")

def is_industrial(lat, lon):
    for site in INDUSTRIAL_SITES:
        dist = ((lat - site['lat'])**2 + (lon - site['lon'])**2)**0.5
        if dist < 0.02: # Aproximadamente 2km
            return True
    return False

def fetch_firms_data():
    print("🛰️ Conectando à API NASA FIRMS...")
    api_key = os.getenv('NASA_API_KEY')
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{api_key}/VIIRS_SNPP/UKR/1"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"❌ Erro NASA: {response.status_code}")
            return []
        
        df = pd.read_csv(io.StringIO(response.text))
        print(f"📊 {len(df)} pontos térmicos totais encontrados nas últimas 24h.")
        
        new_fires = []
        state = load_state()
        
        for _, row in df.iterrows():
            f_id = f"{row['latitude']}_{row['longitude']}_{row['acq_date']}"
            
            # Filtro de Potência
            if row['frp'] < MIN_FRP:
                continue
            
            # Filtro Industrial
            if is_industrial(row['latitude'], row['longitude']):
                print(f"🏭 Ignorado (Zona Industrial): {row['latitude']}, {row['longitude']}")
                continue
                
            # Filtro de Duplicata
            if f_id not in state['processed_fires']:
                print(f"🔥 NOVO FOCO DETECTADO: Lat {row['latitude']}, Lon {row['longitude']} (FRP: {row['frp']})")
                new_fires.append(row)
                state['processed_fires'].append(f_id)
        
        state['processed_fires'] = state['processed_fires'][-500:]
        save_state(state)
        return new_fires
    except Exception as e:
        print(f"❌ Erro ao buscar dados FIRMS: {e}")
        return []

def get_satellite_image(lat, lon):
    print(f"📸 Solicitando imagem Sentinel-2 para {lat}, {lon}...")
    try:
        config = get_config()
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
                    time_interval=(datetime.now(timezone.utc) - timedelta(days=5)).strftime('%Y-%m-%d'),
                )
            ],
            responses=[SentinelHubRequest.output_obj('default', MimeType.PNG)],
            bbox=bbox,
            size=(800, 800),
            config=config
        )
        
        response = request.get_data()
        if response:
            filename = "fire_spot.png"
            with open(filename, "wb") as f:
                f.write(response[0])
            print("✅ Imagem de satélite obtida.")
            return filename
    except Exception as e:
        print(f"⚠️ Erro no Copernicus: {e}")
    return None

def send_telegram(caption, image_path):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    try:
        with open(image_path, 'rb') as photo:
            r = requests.post(url, data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}, files={'photo': photo}, timeout=30)
            if r.status_code == 200:
                print("📤 Mensagem enviada ao Telegram!")
            else:
                print(f"❌ Erro Telegram: {r.text}")
    except Exception as e:
        print(f"❌ Erro ao enviar: {e}")

def main():
    print("🚀 --- INICIANDO MONITORAMENTO OSINT ---")
    fires = fetch_firms_data()
    
    if not fires:
        print("💤 Nenhum novo incêndio relevante encontrado.")
        return

    print(f"🎯 Processando {min(len(fires), 3)} alertas...")
    for _, fire in pd.DataFrame(fires).head(3).iterrows():
        img = get_satellite_image(fire['latitude'], fire['longitude'])
        if img:
            caption = (
                f"🔥 *STRIKE ALERT - UKRAINE*\n\n"
                f"🎯 *Confidence:* {fire['confidence']}\n"
                f"🚀 *FRP:* {fire['frp']}\n"
                f"📍 *Location:* `{fire['latitude']}, {fire['longitude']}`\n"
                f"🕒 *Time:* {fire['acq_date']} {fire['acq_time']} UTC\n\n"
                f"🔗 [Open Google Maps](https://www.google.com/maps?q={fire['latitude']},{fire['longitude']})"
            )
            send_telegram(caption, img)
        else:
            print("☁️ Imagem não disponível (nuvens ou falha no satélite).")
    
    print("🏁 --- MONITORAMENTO CONCLUÍDO ---")

if __name__ == "__main__":
    main()
