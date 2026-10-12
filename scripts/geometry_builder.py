"""Rebuild cached OSM geometry traces; scheduling is evaluated by run_icc.py.
Building data: OpenStreetMap contributors, ODbL.
"""
import os, re, json, time, math
import numpy as np
import requests
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.path import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.bbox': 'tight', 'savefig.dpi': 300,
})
BLUE, RED, GREEN, ORANGE, PURPLE, GREY, BLACK = (
    '#2166AC', '#B2182B', '#1B7837', '#F1A340', '#762A83', '#999999', '#222222')

from types import SimpleNamespace
def los_blocked(ux,uy,uz, gx,gy,gz, Hgrid, cell, span, steps=60):
    """True if the UAV->GS segment is blocked by any building (3D geometric LoS)."""
    for a in np.linspace(0.02,0.98,steps):
        x=ux+a*(gx-ux); y=uy+a*(gy-uy); z=uz+a*(gz-uz)
        i=int(x//cell); j=int(y//cell)
        if 0<=i<Hgrid.shape[0] and 0<=j<Hgrid.shape[1]:
            if Hgrid[i,j] > z: return True
    return False


def autocorr(x, lags=30):
    x=x.astype(float)-x.mean(); ac=[np.corrcoef(x[:-k],x[k:])[0,1] for k in range(1,lags)]
    return np.array(ac)

# --- schedulers on a given LoS trace (belief uses a Markov-estimated model) ---

U=SimpleNamespace(TAU=0.1,los_blocked=los_blocked,autocorr=autocorr)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); FIG = os.path.join(HERE, "figs")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# ------------------------- named district + real bbox -------------------------
CITY = "Midtown Manhattan (Rockefeller Center core), New York City"
CITY_SHORT = "Midtown Manhattan"
SLUG = "midtown_manhattan"
# Overpass / standard bbox order = (south, west, north, east) = (minlat,minlon,maxlat,maxlon)
# ~1.2 km dense Midtown core (approx. W44th-W54th St, 5th-7th Ave), centered near
# Rockefeller Center (40.7539 N, 73.9840 W); matches the paper's operational scale.
MINLAT, MINLON, MAXLAT, MAXLON = 40.7485, -73.9911, 40.7593, -73.9769
CACHE = os.path.join(DATA, f"osm_{SLUG}.json")

CELL = 6.0        # height-grid cell size (m); task target 5-8 m
N_UAV = 12; T_SLOTS = 2000; GEO_SEED = 1; RUN_SEEDS = range(4)   # match e12_urban.py

# ------------------------- Overpass fetch (cache + mirrors + retry) -----------
def fetch_or_load():
    """Load cached raw JSON if present (offline-reproducible); else query
    Overpass (two mirrors, 3 attempts, 20 s backoff) and cache the raw JSON."""
    if os.path.exists(CACHE):
        with open(CACHE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[cache]    loaded {os.path.relpath(CACHE, HERE)} "
              f"({os.path.getsize(CACHE)//1024} KiB) -> OFFLINE reproduction")
        return data, 'cache'
    q = (f'[out:json][timeout:60];'
         f'(way["building"]({MINLAT},{MINLON},{MAXLAT},{MAXLON}););out body geom;')
    mirrors = ['https://overpass-api.de/api/interpreter',
               'https://overpass.kumi.systems/api/interpreter']
    last = None
    for attempt in range(3):
        for url in mirrors:
            try:
                print(f"[overpass] POST {url} (attempt {attempt+1}/3)")
                r = requests.post(url, data={'data': q}, timeout=120,
                                  headers={'User-Agent':
                                           'infocom-research/1.0 (OSM building footprints; ODbL)'})
                if r.status_code == 200:
                    data = r.json()
                    with open(CACHE, 'w', encoding='utf-8') as f:
                        json.dump(data, f)
                    print(f"[overpass] OK -> cached {os.path.relpath(CACHE, HERE)} "
                          f"({len(r.content)//1024} KiB)")
                    return data, 'overpass'
                print(f"[overpass] HTTP {r.status_code} from {url}")
                last = f"HTTP {r.status_code}"
            except Exception as e:
                print(f"[overpass] ERROR {url}: {e!r}")
                last = repr(e)
        if attempt < 2:
            print("[overpass] all mirrors failed; waiting 20 s then retrying...")
            time.sleep(20)
    raise SystemExit(
        f"[FATAL] Overpass unreachable ({last}) and no local cache present. "
        f"Re-run when online; the raw data will then be cached at {CACHE} for "
        f"offline reuse. (No data fabricated.)")

# ------------------------- building parsing -----------------------------------
_NUM = re.compile(r'-?\d+(?:\.\d+)?')
def _first_float(s):
    m = _NUM.search(str(s)); return float(m.group()) if m else None

def parse_height(tags):
    """Return (height_m, source) with source in {'tagged','levels','default'}.
    Priority: explicit 'height' tag (meters; feet converted) > building:levels*3 m
    > 10 m default. Values clipped to a sane [2,500] m."""
    if 'height' in tags:
        raw = str(tags['height']).strip().lower().replace(',', '.')
        v = _first_float(raw)
        if v is not None:
            if ('ft' in raw) or ("'" in raw) or ('feet' in raw):
                v *= 0.3048
            return float(np.clip(v, 2.0, 500.0)), 'tagged'
    if 'building:levels' in tags:
        v = _first_float(tags['building:levels'])
        if v is not None and v > 0:
            return float(np.clip(v * 3.0, 2.0, 500.0)), 'levels'
    return 10.0, 'default'

def parse_buildings(data):
    ways = [e for e in data.get('elements', [])
            if e.get('type') == 'way' and e.get('geometry')]
    polys, heights, src = [], [], {'tagged': 0, 'levels': 0, 'default': 0}
    for e in ways:
        g = e['geometry']
        if len(g) < 3:
            continue
        lat = np.array([p['lat'] for p in g], float)
        lon = np.array([p['lon'] for p in g], float)
        h, s = parse_height(e.get('tags', {}))
        polys.append((lat, lon)); heights.append(h); src[s] += 1
    return polys, np.array(heights, float), src

# ------------------------- projection + rasterization -------------------------
def build_grid(polys, heights):
    """Equirectangular projection about the bbox-center latitude (origin at the
    SW corner so all coords >=0), then rasterize building footprints to a height
    grid H[ix,iy] = max building height over cell centers inside the footprint.
    Indexing matches urban_blockage.los_blocked (i=x//cell, j=y//cell)."""
    lat_c = 0.5 * (MINLAT + MAXLAT)
    kx = 111320.0 * math.cos(math.radians(lat_c)); ky = 110540.0
    lon0, lat0 = MINLON, MINLAT
    span_x = (MAXLON - MINLON) * kx; span_y = (MAXLAT - MINLAT) * ky
    nx = int(math.ceil(span_x / CELL)); ny = int(math.ceil(span_y / CELL))
    H = np.zeros((nx, ny), np.float32)
    for (lat, lon), h in zip(polys, heights):
        x = (lon - lon0) * kx; y = (lat - lat0) * ky
        i0 = max(int(x.min() // CELL), 0); i1 = min(int(x.max() // CELL), nx - 1)
        j0 = max(int(y.min() // CELL), 0); j1 = min(int(y.max() // CELL), ny - 1)
        if i1 < i0 or j1 < j0:
            continue
        ii = np.arange(i0, i1 + 1); jj = np.arange(j0, j1 + 1)
        XX, YY = np.meshgrid((ii + 0.5) * CELL, (jj + 0.5) * CELL, indexing='ij')
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        inside = Path(np.column_stack([x, y])).contains_points(pts).reshape(XX.shape)
        if inside.any():
            sub = H[i0:i1 + 1, j0:j1 + 1]
            np.maximum(sub, inside * np.float32(h), out=sub)
        else:  # footprint smaller than a cell: stamp its centroid cell
            ci = int(np.clip(x.mean() // CELL, 0, nx - 1))
            cj = int(np.clip(y.mean() // CELL, 0, ny - 1))
            H[ci, cj] = max(H[ci, cj], np.float32(h))
    return H, CELL, span_x, span_y, (kx, ky, lat_c)

def place_gs(H, cell):
    """Ground station on a central mid-rise rooftop: nearest-to-center cell whose
    building height is in [30,50] m; mast top clipped to [35,50] m."""
    nx, ny = H.shape; ci, cj = nx // 2, ny // 2
    roof = np.argwhere((H >= 30.0) & (H <= 50.0))
    if len(roof):
        gi, gj = roof[np.argmin((roof[:, 0] - ci) ** 2 + (roof[:, 1] - cj) ** 2)]
        gh = float(H[gi, gj])
    else:
        gi, gj, gh = ci, cj, float(H[ci, cj])
    gx = (gi + 0.5) * cell; gy = (gj + 0.5) * cell
    gz = float(np.clip((gh if gh >= 30 else 42.0) + 3.0, 35.0, 50.0))
    return np.array([gx, gy, gz]), (int(gi), int(gj), gh)

# ------------------------- trace generation (mirrors U.gen_traces) ------------
def gen_traces_osm(H, cell, span_x, span_y, gs, N=N_UAV, T=T_SLOTS, seed=GEO_SEED):
    """Straight patrol trajectories at 80-120 m wrapping in the district, with an
    exact 3D LoS ray-march to the GS over the REAL height grid via
    urban_blockage.los_blocked (reused unchanged; step count adapted to the link
    length so the march resolves the ~6 m grid)."""
    rng = np.random.default_rng(seed)
    los = np.zeros((T, N), bool); dist = np.zeros((T, N))
    for n in range(N):
        h = rng.uniform(80, 120)
        p0 = np.array([rng.uniform(0, span_x), rng.uniform(0, span_y), h])
        ang = rng.uniform(0, 2 * np.pi); v = rng.uniform(6, 14) * U.TAU     # m/slot
        vel = np.array([np.cos(ang), np.sin(ang), 0.0]) * v
        for t in range(T):
            p = p0 + vel * t
            p[0] = p[0] % span_x; p[1] = p[1] % span_y                      # wrap in district
            seg = float(np.linalg.norm(p - gs))
            steps = int(np.clip(seg / (cell * 0.7), 80, 400))               # resolve the grid
            blk = U.los_blocked(p[0], p[1], p[2], gs[0], gs[1], gs[2], H, cell, span_x, steps=steps)
            los[t, n] = (not blk); dist[t, n] = seg
    return los, dist

def autocorr_avg(los):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nanmean([U.autocorr(los[:, n]) for n in range(los.shape[1])], axis=0)

def load_traces():
    """Build (or load cached) the OSM LoS/distance traces so downstream figure
    scripts reproduce fast/offline without re-ray-marching."""
    tc = os.path.join(DATA, f"osm_traces_{SLUG}.npz")
    if os.path.exists(tc):
        z = np.load(tc); return z['los'], z['dist']
    data, _ = fetch_or_load(); polys, heights, _ = parse_buildings(data)
    H, cell, sx, sy, _ = build_grid(polys, heights); gs, _ = place_gs(H, cell)
    los, dist = gen_traces_osm(H, cell, sx, sy, gs)
    np.savez_compressed(tc, los=los, dist=dist); return los, dist


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(); parser.add_argument('--city',choices=['midtown_manhattan','city_of_london'],required=True)
 args=parser.parse_args()
 if args.city=='city_of_london':
  MINLAT,MINLON,MAXLAT,MAXLON=51.5100,-0.0900,51.5200,-0.0740
  SLUG='city_of_london'; CACHE=os.path.join(DATA,'osm_city_of_london.json')
 if not os.path.exists(CACHE): raise SystemExit('Included raw map cache is required; no online replacement is used.')
 data,_=fetch_or_load(); polys,heights,src=parse_buildings(data)
 Hg,cell,sx,sy,_=build_grid(polys,heights); gs,_=place_gs(Hg,cell)
 lo,di=gen_traces_osm(Hg,cell,sx,sy,gs)
 dest=os.path.join(DATA,'rebuilt_'+SLUG+'.npz')
 np.savez_compressed(dest,los=lo,dist=di)
 old=np.load(os.path.join(DATA,'osm_traces_'+SLUG+'.npz'))
 print(SLUG,len(polys),src,'LoS matches:',np.array_equal(lo,old['los']),'distance matches:',np.allclose(di,old['dist']), 'saved',dest)
