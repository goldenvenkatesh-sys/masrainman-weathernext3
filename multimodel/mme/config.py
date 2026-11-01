# ============================================================
# MASRAINMAN MULTI-MODEL — SETTINGS
# ============================================================

import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY_DIR = os.path.join(HERE, "boundaries")
CACHE_DIR = os.environ.get("MME_CACHE", os.path.join(HERE, "cache"))
SITE_DIR = os.environ.get("SITE_DIR", os.path.join(HERE, "site"))

# Common analysis grid (all models are regridded to this)
GRID_LAT = (0.0, 40.0)
GRID_LON = (60.0, 100.0)
GRID_STEP = 0.1

# Tracking grid (North Indian Ocean) and lead times
TRACK_LAT = (-5.0, 35.0)
TRACK_LON = (40.0, 105.0)
TRACK_STEP = 0.25
TRACK_STEPS = list(range(0, 169, 12))          # T+0 ... T+168 h, every 12 h
DETECT_AREA = (45.0, 100.0, 0.0, 27.0)         # W, E, S, N where lows are searched

# Members used for tracking (pressure only). More members = longer downloads.
TRACK_MEMBERS = {
    "ecmwf_ens": int(os.environ.get("TRACK_ECMWF_ENS_MEMBERS", "25")),
    "aifs_ens": int(os.environ.get("TRACK_AIFS_ENS_MEMBERS", "25")),
    "gefs": int(os.environ.get("TRACK_GEFS_MEMBERS", "30")),
    "geps": 20,
}

# Rainfall day = 08:30 IST -> 08:30 IST = 03 UTC -> 03 UTC
DAY_END_UTC_HOUR = 3
N_DAYS = 15

# ------------------------------------------------------------
# MODEL GROUPS AND WEIGHTS
# Weights are renormalised over the models available for each day.
# ------------------------------------------------------------

GROUPS = {
    "ens": {
        "title": "Ensemble Blend",
        "models": {"ecmwf_ens": 0.30, "wn3": 0.25, "aifs_ens": 0.20, "gefs": 0.15, "geps": 0.10},
    },
    "det": {
        "title": "Deterministic Blend",
        "models": {"ecmwf_hres": 0.35, "aifs_single": 0.25, "icon": 0.20, "gfs": 0.20},
    },
}

MODEL_NAMES = {
    "ecmwf_ens": "ECMWF ENS", "wn3": "WeatherNext 3", "aifs_ens": "ECMWF AIFS ENS",
    "gefs": "NCEP GEFS", "geps": "CMC GEPS", "ecmwf_hres": "ECMWF IFS HRES",
    "aifs_single": "ECMWF AIFS Single", "icon": "DWD ICON", "gfs": "NCEP GFS",
}

# Members used from each ensemble (None = all). Fewer members = faster downloads.
ENS_MEMBERS = {
    "ecmwf_ens": int(os.environ.get("ECMWF_ENS_MEMBERS", "50")),
    "aifs_ens": int(os.environ.get("AIFS_ENS_MEMBERS", "50")),
}

# Minimum number of models needed before a blended day is drawn.
# ECMWF runs stop at 360 h, so the last days may rest on fewer models;
# the models actually used are printed on every map.
MIN_MODELS_PER_DAY = 2
MIN_MODELS_LATE = 1      # for days >= LATE_DAY
LATE_DAY = 13

# ------------------------------------------------------------
# REGIONS — key: (label, map extent [W, E, S, N], state or None)
# ------------------------------------------------------------

REGIONS = {
    "south_india": ("South India", [72.5, 85.5, 7.5, 20.0], None),
    "north_india": ("North India", [68.0, 97.5, 21.0, 37.2], None),
    "tamilnadu": ("Tamil Nadu", [76.0, 80.6, 7.9, 13.7], "Tamil Nadu"),
    "kerala": ("Kerala", [74.6, 77.7, 8.0, 12.9], "Kerala"),
    "karnataka": ("Karnataka", [73.8, 78.8, 11.4, 18.6], "Karnataka"),
    "ap": ("Andhra Pradesh", [76.6, 84.9, 12.5, 19.3], "Andhra Pradesh"),
    "telangana": ("Telangana", [77.1, 81.5, 15.7, 19.98], "Telangana"),
}

SOUTH_CITIES = [
    ("Chennai", 13.08, 80.27), ("Bengaluru", 12.97, 77.59), ("Hyderabad", 17.39, 78.49),
    ("Thiruvananthapuram", 8.52, 76.94), ("Kochi", 9.93, 76.27), ("Kozhikode", 11.26, 75.78),
    ("Mangaluru", 12.91, 74.86), ("Mysuru", 12.30, 76.64), ("Coimbatore", 11.02, 76.96),
    ("Madurai", 9.93, 78.12), ("Tiruchirappalli", 10.80, 78.69), ("Puducherry", 11.93, 79.83),
    ("Visakhapatnam", 17.69, 83.22), ("Vijayawada", 16.51, 80.65), ("Tirupati", 13.63, 79.42),
    ("Kurnool", 15.83, 78.04), ("Warangal", 17.97, 79.59), ("Hubballi", 15.36, 75.12),
    ("Belagavi", 15.85, 74.50), ("Panaji", 15.49, 73.83), ("Mumbai", 19.08, 72.88),
    ("Pune", 18.52, 73.86), ("Kanyakumari", 8.08, 77.54), ("Tirunelveli", 8.71, 77.76),
    ("Salem", 11.66, 78.15), ("Nellore", 14.44, 79.99), ("Kalaburagi", 17.33, 76.83),
]

NORTH_CITIES = [
    ("New Delhi", 28.61, 77.21), ("Jaipur", 26.91, 75.79), ("Lucknow", 26.85, 80.95),
    ("Kolkata", 22.57, 88.36), ("Patna", 25.59, 85.14), ("Bhopal", 23.26, 77.41),
    ("Ahmedabad", 23.02, 72.57), ("Chandigarh", 30.73, 76.78), ("Srinagar", 34.08, 74.80),
    ("Leh", 34.15, 77.58), ("Shimla", 31.10, 77.17), ("Dehradun", 30.32, 78.03),
    ("Varanasi", 25.32, 82.97), ("Guwahati", 26.14, 91.74), ("Ranchi", 23.34, 85.31),
    ("Raipur", 21.25, 81.63), ("Indore", 22.72, 75.86), ("Jodhpur", 26.24, 73.02),
    ("Gangtok", 27.33, 88.61), ("Shillong", 25.58, 91.89), ("Imphal", 24.82, 93.94),
    ("Agartala", 23.83, 91.29), ("Itanagar", 27.08, 93.61), ("Amritsar", 31.63, 74.87),
    ("Gwalior", 26.22, 78.18), ("Prayagraj", 25.44, 81.85), ("Nagpur", 21.15, 79.09),
    ("Jammu", 32.73, 74.86), ("Kohima", 25.67, 94.11), ("Bikaner", 28.02, 73.31),
]

# MASRainman house colour bars (from the MASRainman ECMWF maps), a number on every box.
# period: (boundaries, colours between them, colour below, colour above, label)
RAIN_24H = (
    [0.1, 0.3, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 150, 200, 300],
    ["#ffffff", "#d8d0fc", "#bcd6ec", "#9cc0e4", "#6aa2da", "#3a7cc4", "#1452a0", "#1e6e98",
     "#2e8a3c", "#60b04c", "#a8c84a", "#e2e822", "#f8d820", "#f49a1e", "#ea6414", "#dc3c28",
     "#c02626", "#901438", "#b4145a", "#e01e64", "#ee5a8c", "#c4b4ba", "#808080"],
    "#ffffff", "#606060", "Rainfall Intensity (mm)")
RAIN_ACC = (
    [0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250,
     300, 400, 500, 600, 800],
    ["#f0f0f0", "#cce0ff", "#96c8ff", "#5aaaf8", "#1e8cf0", "#0a4ca0", "#1e5a28", "#00c850",
     "#50d228", "#c8f000", "#ffd700", "#ffaa00", "#fc6e14", "#fc8c00", "#ff5a82", "#ff2a5a",
     "#ff0a3c", "#dc0014", "#78209c", "#6400ff", "#c800ff", "#d282ff", "#ecd2fa", "#dcdcdc",
     "#a0a0a0", "#787878", "#505050"],
    "#ffffff", "#404040", "Rainfall Accumulation (mm)")
RAIN_BARS = {"24h": RAIN_24H, "7day": RAIN_ACC, "15day": RAIN_ACC}
