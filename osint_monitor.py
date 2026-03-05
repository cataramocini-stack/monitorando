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
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

INDUSTRIAL_SITES = [
    {"name": "Kremenchuk", "lat": 49.12, "lon": 33.48},
    {"name": "Zaporizhzhia", "lat": 47.51, "lon": 34.58},
    {"name": "Odessa Port", "lat": 46.50, "lon": 30.73}
]

MIN_FRP = 15.0

STATE_FILE = "bot_state.json"

MAX_RECORDS = 300
MAX_ALERTS = 2

CLOUD_THRESHOLD = 0.2

IMAGE_DIR = "fire_images"
HEATMAP_DIR = "heatmaps"

CLUSTER_DISTANCE_KM = 2
CLUSTER_TIME_MINUTES = 20
MIN_CLUSTER_POINTS = 2

MAX_CLUSTER_SIZE_KM = 8
MAX_AGRI_FRP = 60

AGRI_CLUSTER_POINTS = 6
AGRI_CLUSTER_RADIUS_KM = 5
AGRI_TIME_SPREAD_MIN = 60
AGRI_MEAN_FRP_MAX = 40


# Ucrânia apenas
FIRMS_BBOX = "22,44,40,52"


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------
# SENTINEL CONFIG
# ------------------------------------------------

def get_config():
    config = SHConfig()
    config.sh_client_id = os.getenv("CDSE_CLIENT_ID")
    config.sh_client_secret = os.getenv("CDSE_CLIENT_SECRET")
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return config


# ------------------------------------------------
# STATE
# ------------------------------------------------

def load_state():

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE,"r") as f:
            data=json.load(f)
    else:
        data={}

    processed=data.get("processed_fires",[])

    normalized=[]

    for item in processed:

        if isinstance(item,dict) and "id" in item:
            normalized.append(item)

        elif isinstance(item,str):
            normalized.append({"id":item})

    return {
        "schema_version":data.get("schema_version",1),
        "processed_fires":normalized
    }


def save_state(state):

    with open(STATE_FILE,"w") as f:
        json.dump(state,f,indent=2)


def processed_ids(state):

    return {item["id"] for item in state.get("processed_fires",[]) if "id" in item}


# ------------------------------------------------
# REQUEST
# ------------------------------------------------

def request_with_retries(url,attempts=3,timeout=30):

    for attempt in range(1,attempts+1):

        try:

            response=requests.get(url,timeout=timeout)
            response.raise_for_status()

            return response

        except Exception as exc:

            if attempt>=attempts:
                logger.error("FIRMS request failed after %s attempts: %s",attempts,exc)
                raise

            backoff=2**(attempt-1)

            logger.error("FIRMS request error attempt %s: %s",attempt,exc)

            time.sleep(backoff)


# ------------------------------------------------
# UTIL
# ------------------------------------------------

def is_industrial(lat,lon):

    for site in INDUSTRIAL_SITES:

        if geodesic((lat,lon),(site["lat"],site["lon"])).km<=1.5:
            return True

    return False


def parse_fire_timestamp(acq_date,acq_time):

    try:

        acq_time_str=str(int(acq_time)).zfill(4)

        return datetime.strptime(
            f"{acq_date}{acq_time_str}",
            "%Y-%m-%d%H%M"
        ).replace(tzinfo=timezone.utc)

    except:

        return datetime.now(timezone.utc)


def build_fire_id(lat,lon,acq_date,acq_time):

    return f"{lat:.5f}_{lon:.5f}_{acq_date}_{str(acq_time).zfill(4)}"


# ------------------------------------------------
# CLUSTER DETECTION
# ------------------------------------------------

def detect_clusters(fires):

    clusters=[]
    visited=set()

    for i,fire in enumerate(fires):

        if i in visited:
            continue

        lat=float(fire["latitude"])
        lon=float(fire["longitude"])
        t=parse_fire_timestamp(fire["acq_date"],fire["acq_time"])

        cluster=[fire]
        visited.add(i)

        for j,other in enumerate(fires):

            if j in visited or i==j:
                continue

            lat2=float(other["latitude"])
            lon2=float(other["longitude"])
            t2=parse_fire_timestamp(other["acq_date"],other["acq_time"])

            dist=geodesic((lat,lon),(lat2,lon2)).km
            dt=abs((t-t2).total_seconds())/60

            if dist<=CLUSTER_DISTANCE_KM and dt<=CLUSTER_TIME_MINUTES:

                cluster.append(other)
                visited.add(j)

        if len(cluster)>=MIN_CLUSTER_POINTS:

            clusters.append(cluster)

    return clusters


# ------------------------------------------------
# AGRICULTURAL FILTER
# ------------------------------------------------

def is_agricultural_pattern(cluster):

    lats=[float(f["latitude"]) for f in cluster]
    lons=[float(f["longitude"]) for f in cluster]
    frps=[float(f["frp"]) for f in cluster]

    times=[
        parse_fire_timestamp(f["acq_date"],f["acq_time"])
        for f in cluster
    ]

    center=(sum(lats)/len(lats),sum(lons)/len(lons))

    max_dist=max(
        geodesic(center,(lats[i],lons[i])).km
        for i in range(len(cluster))
    )

    mean_frp=sum(frps)/len(frps)

    time_spread=max(times)-min(times)
    time_spread_minutes=time_spread.total_seconds()/60

    if len(cluster)>=AGRI_CLUSTER_POINTS:
        if max_dist>AGRI_CLUSTER_RADIUS_KM:
            return True

    if mean_frp<AGRI_MEAN_FRP_MAX and time_spread_minutes>AGRI_TIME_SPREAD_MIN:
        return True

    if max_dist>MAX_CLUSTER_SIZE_KM and max(frps)<MAX_AGRI_FRP:
        return True

    return False


# ------------------------------------------------
# HEATMAP
# ------------------------------------------------

def generate_heatmap(fires):

    if not fires:
        return

    os.makedirs(HEATMAP_DIR,exist_ok=True)

    lats=[float(f["latitude"]) for f in fires]
    lons=[float(f["longitude"]) for f in fires]

    plt.figure(figsize=(6,8))

    plt.hexbin(lons,lats,gridsize=40,cmap="inferno")

    plt.colorbar(label="Strike density")

    plt.title("Ukraine Strike Heatmap")

    week=datetime.now().strftime("%Y_%W")

    path=f"{HEATMAP_DIR}/heatmap_{week}.png"

    plt.savefig(path,dpi=200)

    plt.close()

    logger.info("Heatmap generated: %s",path)


# ------------------------------------------------
# FIRMS
# ------------------------------------------------

def fetch_dataset(api_key,dataset):

    url=f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{dataset}/{FIRMS_BBOX}/1"

    try:

        r=request_with_retries(url)

        df=pd.read_csv(io.StringIO(r.text))

        return df

    except Exception as e:

        logger.error("Dataset error %s : %s",dataset,e)

        return pd.DataFrame()


def fetch_firms_data(state):

    api_key=os.getenv("NASA_API_KEY")

    if not api_key:
        logger.error("NASA_API_KEY not configured")
        return []

    logger.info("Starting FIRMS check")

    df1=fetch_dataset(api_key,"VIIRS_SNPP_NRT")
    df2=fetch_dataset(api_key,"VIIRS_NOAA20_NRT")

    df=pd.concat([df1,df2],ignore_index=True)

    if df.empty:

        logger.info("No FIRMS data returned")
        return []

    df=df.sort_values(by="frp",ascending=False)

    seen_ids=processed_ids(state)

    new_fires=[]

    for _,row in df.iterrows():

        lat=float(row["latitude"])
        lon=float(row["longitude"])

        if row["frp"]<MIN_FRP:
            continue

        if is_industrial(lat,lon):
            continue

        fire_id=build_fire_id(lat,lon,row["acq_date"],row["acq_time"])

        if fire_id in seen_ids:
            continue

        fire=row.to_dict()

        fire["fire_id"]=fire_id

        fire["fire_timestamp"]=parse_fire_timestamp(
            row["acq_date"],
            row["acq_time"]
        ).isoformat()

        new_fires.append(fire)

    clusters=detect_clusters(new_fires)

    filtered=[]
    clustered_ids=set()

    for cluster in clusters:

        if not is_agricultural_pattern(cluster):
            filtered.extend(cluster)

        for f in cluster:
            clustered_ids.add(f["fire_id"])

    for fire in new_fires:
        if fire["fire_id"] not in clustered_ids:
            filtered.append(fire)

    new_fires=filtered

    generate_heatmap(new_fires)

    logger.info("FIRMS fetched: %s new fires",len(new_fires))

    return new_fires


# ------------------------------------------------
# SENTINEL IMAGE
# ------------------------------------------------

def get_evalscript():
    return """
//VERSION=3
function setup() {
  return {
    input: ["B12","B08","B04"],
    output:{bands:3}
  };
}
function evaluatePixel(sample){

  var swir=sample.B12;
  var nir=sample.B08;
  var red=sample.B04;

  var fire=Math.max(0,swir-0.1)*4.0;

  return[
    Math.min(1.0,red*1.5+fire),
    Math.min(1.0,nir*1.2),
    Math.min(1.0,red*0.8)
  ];
}
"""


def request_sentinel_image(lat,lon,data_collection):

    config=get_config()

    offset=0.012

    bbox=BBox(
        bbox=[lon-offset,lat-offset,lon+offset,lat+offset],
        crs=CRS.WGS84
    )

    now=datetime.now(timezone.utc)

    intervals=[
        (now-timedelta(days=3),now),
        (now-timedelta(days=7),now),
        (now-timedelta(days=14),now)
    ]

    for interval in intervals:

        request=SentinelHubRequest(

            evalscript=get_evalscript(),

            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=interval,
                    maxcc=CLOUD_THRESHOLD
                )
            ],

            responses=[SentinelHubRequest.output_response("default",MimeType.PNG)],

            bbox=bbox,

            size=(800,800),

            config=config
        )

        data=request.get_data()

        if data:
            return data[0]

    return None


def get_satellite_image(fire):

    lat=float(fire["latitude"])
    lon=float(fire["longitude"])

    os.makedirs(IMAGE_DIR,exist_ok=True)

    ts=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    filename=f"fire_{fire['fire_id']}_{ts}.png"

    path=os.path.join(IMAGE_DIR,filename)

    data=request_sentinel_image(lat,lon,DataCollection.SENTINEL2_L2A)

    if not data:
        return None

    with open(path,"wb") as f:
        f.write(data)

    logger.info("Image generated: %s",path)

    return path


# ------------------------------------------------
# TELEGRAM
# ------------------------------------------------

def escape_markdown_v2(text):
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])",r"\\\1",str(text))


def escape_markdown_v2_url(url):
    return url.replace("\\","\\\\").replace(")","\\)").replace("(","\\(")


def build_caption(fire):

    lat=float(fire["latitude"])
    lon=float(fire["longitude"])

    frp=fire.get("frp")

    frp_text=escape_markdown_v2(f"{frp}")
    coords_text=escape_markdown_v2(f"{lat:.4f}, {lon:.4f}")

    url=escape_markdown_v2_url(f"https://www.google.com/maps?q={lat},{lon}")

    return(
        "🔥 *POTENTIAL STRIKE DETECTED*\n\n"
        f"🚀 *Power (FRP):* `{frp_text}`\n"
        f"📍 *Coords:* `{coords_text}`\n"
        "🛰️ *Sensor:* VIIRS (NASA)\n\n"
        f"🔗 [View on Google Maps]({url})"
    )


def send_telegram(caption,image_path):

    token=os.getenv("TELEGRAM_TOKEN")
    chat_id=os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.error("Telegram credentials not configured")
        return False

    try:

        with open(image_path,"rb") as photo:

            response=requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={
                    "chat_id":chat_id,
                    "caption":caption,
                    "parse_mode":"MarkdownV2"
                },
                files={"photo":photo},
                timeout=30
            )

        if response.status_code>=400:
            logger.error("Telegram send failed: %s",response.text)
            return False

        logger.info("Telegram sent for image %s",image_path)
        return True

    except Exception as exc:

        logger.error("Telegram send failed: %s",exc)
        return False


# ------------------------------------------------
# PROCESS
# ------------------------------------------------

def is_fire_confirmed(fire):
    return fire.get("frp",0)>=MIN_FRP


def build_state_entry(fire):

    return {
        "id":fire["fire_id"],
        "lat":fire.get("latitude"),
        "lon":fire.get("longitude"),
        "acq_date":fire.get("acq_date"),
        "acq_time":fire.get("acq_time"),
        "processed_at":datetime.now(timezone.utc).isoformat()
    }


def process_fire(fire):

    if not is_fire_confirmed(fire):
        return None

    image_path=get_satellite_image(fire)

    if not image_path:
        return None

    caption=build_caption(fire)

    if send_telegram(caption,image_path):
        return build_state_entry(fire)

    return None


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    try:

        state=load_state()

        fires=fetch_firms_data(state)

        if not fires:
            logger.info("No new fires to process")
            return

        fires_to_process=fires[:MAX_ALERTS]

        processed_entries=[]

        with ThreadPoolExecutor(max_workers=min(4,len(fires_to_process))) as executor:

            futures=[executor.submit(process_fire,fire) for fire in fires_to_process]

            for future in as_completed(futures):

                entry=future.result()

                if entry:
                    processed_entries.append(entry)

        if processed_entries:

            state["processed_fires"].extend(processed_entries)

            state["processed_fires"]=state["processed_fires"][-MAX_RECORDS:]

            save_state(state)

            logger.info("State updated with %s new entries",len(processed_entries))

        else:

            logger.info("No new entries saved")

    except Exception as exc:

        logger.exception("Unhandled error in main: %s",exc)


if __name__ == "__main__":
    main()
