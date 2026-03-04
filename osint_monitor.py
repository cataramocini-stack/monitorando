import os
import json
import requests
import io
import logging
import time
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType
from geopy.distance import geodesic

INDUSTRIAL_SITES = [
    {"name": "Kremenchuk", "lat": 49.12, "lon": 33.48},
    {"name": "Zaporizhzhia", "lat": 47.51, "lon": 34.58},
    {"name": "Odessa Port", "lat": 46.50, "lon": 30.73}
]
MIN_FRP = 35.0
STATE_FILE = "bot_state.json"
MAX_RECORDS = 300
MAX_ALERTS = 2
CLOUD_THRESHOLD = 0.2
IMAGE_DIR = "fire_images"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def get_config():
    config = SHConfig()
    config.sh_client_id = os.getenv("CDSE_CLIENT_ID")
    config.sh_client_secret = os.getenv("CDSE_CLIENT_SECRET")
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return config

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}
    processed = data.get("processed_fires", [])
    normalized = []
    for item in processed:
        if isinstance(item, dict) and "id" in item:
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append({"id": item})
    return {
        "schema_version": data.get("schema_version", 1),
        "processed_fires": normalized
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def processed_ids(state):
    return {item["id"] for item in state.get("processed_fires", []) if "id" in item}

def request_with_retries(url, attempts=3, timeout=30):
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            if attempt >= attempts:
                logger.error("FIRMS request failed after %s attempts: %s", attempts, exc)
                raise
            backoff = 2 ** (attempt - 1)
            logger.error("FIRMS request error on attempt %s: %s", attempt, exc)
            time.sleep(backoff)

def is_industrial(lat, lon):
    for site in INDUSTRIAL_SITES:
        distance_km = geodesic((lat, lon), (site["lat"], site["lon"])).km
        if distance_km <= 1.5:
            return True
    return False

def parse_fire_timestamp(acq_date, acq_time):
    if not acq_date:
        return datetime.now(timezone.utc)
    try:
        acq_time = str(int(acq_time)).zfill(4) if acq_time == acq_time else "0000"
        return datetime.strptime(f"{acq_date}{acq_time}", "%Y-%m-%d%H%M").replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def build_fire_id(lat, lon, acq_date, acq_time):
    time_part = str(int(acq_time)).zfill(4) if acq_time == acq_time else "0000"
    date_part = acq_date if acq_date else "unknown"
    return f"{lat:.5f}_{lon:.5f}_{date_part}_{time_part}"

def fetch_firms_data(state):
    api_key = os.getenv("NASA_API_KEY")
    if not api_key:
        logger.error("NASA_API_KEY not configured")
        return []
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{api_key}/VIIRS_SNPP/UKR/1"
    logger.info("Starting FIRMS check")
    try:
        response = request_with_retries(url)
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as exc:
        logger.error("FIRMS fetch failed: %s", exc)
        return []
    if df.empty:
        logger.info("No FIRMS data returned")
        return []
    df = df.sort_values(by="frp", ascending=False)
    seen_ids = processed_ids(state)
    new_fires = []
    for _, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        if row["frp"] < MIN_FRP:
            continue
        if is_industrial(lat, lon):
            continue
        fire_id = build_fire_id(lat, lon, row.get("acq_date"), row.get("acq_time"))
        if fire_id in seen_ids:
            continue
        fire = row.to_dict()
        fire["fire_id"] = fire_id
        fire["fire_timestamp"] = parse_fire_timestamp(row.get("acq_date"), row.get("acq_time")).isoformat()
        new_fires.append(fire)
    logger.info("FIRMS fetched: %s new fires", len(new_fires))
    return new_fires

def get_evalscript():
    return """
//VERSION=3
function setup() {
  return {
    input: ["B12", "B08", "B04"],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  var swir = sample.B12;
  var nir = sample.B08;
  var red = sample.B04;
  var fire = Math.max(0, swir - 0.1) * 4.0;
  return [
    Math.min(1.0, red * 1.5 + fire),
    Math.min(1.0, nir * 1.2),
    Math.min(1.0, red * 0.8)
  ];
}
"""

def request_sentinel_image(lat, lon, data_collection):
    config = get_config()
    offset = 0.012
    bbox = BBox(bbox=[lon - offset, lat - offset, lon + offset, lat + offset], crs=CRS.WGS84)
    request = SentinelHubRequest(
        evalscript=get_evalscript(),
        input_data=[
            SentinelHubRequest.input_data_obj(
                data_collection=data_collection,
                time_interval=(datetime.now(timezone.utc) - timedelta(days=3), datetime.now(timezone.utc)),
                maxcc=CLOUD_THRESHOLD
            )
        ],
        responses=[SentinelHubRequest.output_obj("default", MimeType.PNG)],
        bbox=bbox,
        size=(800, 800),
        config=config
    )
    response = request.get_data()
    if response:
        return response[0]
    return None

def get_satellite_image(fire):
    lat = float(fire["latitude"])
    lon = float(fire["longitude"])
    os.makedirs(IMAGE_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    filename = f"fire_{timestamp}_{lat:.4f}_{lon:.4f}.png"
    path = os.path.join(IMAGE_DIR, filename)
    try:
        data = request_sentinel_image(lat, lon, DataCollection.SENTINEL2_L2A)
        if not data:
            logger.info("L2A image unavailable, falling back to L1C for %s", fire["fire_id"])
            data = request_sentinel_image(lat, lon, DataCollection.SENTINEL2_L1C)
        if not data:
            logger.error("Sentinel image unavailable for %s", fire["fire_id"])
            return None
        with open(path, "wb") as f:
            f.write(data)
        logger.info("Image generated: %s", path)
        return path
    except Exception as exc:
        logger.error("Sentinel image failed for %s: %s", fire["fire_id"], exc)
        return None

def escape_markdown_v2(text):
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", str(text))

def escape_markdown_v2_url(url):
    return url.replace("\\", "\\\\").replace(")", "\\)").replace("(", "\\(")

def build_caption(fire):
    lat = float(fire["latitude"])
    lon = float(fire["longitude"])
    frp = fire.get("frp")
    frp_text = escape_markdown_v2(f"{frp}")
    coords_text = escape_markdown_v2(f"{lat:.4f}, {lon:.4f}")
    url = escape_markdown_v2_url(f"https://www.google.com/maps?q={lat},{lon}")
    return (
        "🔥 *POTENTIAL STRIKE DETECTED*\n\n"
        f"🚀 *Power (FRP):* `{frp_text}`\n"
        f"📍 *Coords:* `{coords_text}`\n"
        "🛰️ *Sensor:* VIIRS (NASA)\n\n"
        f"🔗 [View on Google Maps]({url})"
    )

def send_telegram(caption, image_path):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.error("Telegram credentials not configured")
        return False
    try:
        with open(image_path, "rb") as photo:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "MarkdownV2"},
                files={"photo": photo},
                timeout=30
            )
        if response.status_code >= 400:
            logger.error("Telegram send failed: %s", response.text)
            return False
        logger.info("Telegram sent for image %s", image_path)
        return True
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False

def build_state_entry(fire):
    return {
        "id": fire["fire_id"],
        "lat": fire.get("latitude"),
        "lon": fire.get("longitude"),
        "acq_date": fire.get("acq_date"),
        "acq_time": fire.get("acq_time"),
        "processed_at": datetime.now(timezone.utc).isoformat()
    }

def process_fire(fire):
    image_path = get_satellite_image(fire)
    if not image_path:
        return None
    caption = build_caption(fire)
    if send_telegram(caption, image_path):
        return build_state_entry(fire)
    return None

def main():
    state = load_state()
    fires = fetch_firms_data(state)
    if not fires:
        logger.info("No new fires to process")
        return
    fires_to_process = fires[:MAX_ALERTS]
    processed_entries = []
    if len(fires_to_process) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(fires_to_process))) as executor:
            futures = [executor.submit(process_fire, fire) for fire in fires_to_process]
            for future in as_completed(futures):
                entry = future.result()
                if entry:
                    processed_entries.append(entry)
    else:
        entry = process_fire(fires_to_process[0])
        if entry:
            processed_entries.append(entry)
    if processed_entries:
        state["processed_fires"].extend(processed_entries)
        state["processed_fires"] = state["processed_fires"][-MAX_RECORDS:]
        save_state(state)
        logger.info("State updated with %s new entries", len(processed_entries))
    else:
        logger.info("No new entries saved")

if __name__ == "__main__":
    main()
