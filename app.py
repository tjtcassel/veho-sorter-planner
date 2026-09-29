"""
Veho Inbound Routing & Execution Planner
========================================
Decides, shipment by shipment, whether inbound volume goes to sortation
conveyance ("Sorter") or manual sortation, then builds an execution plan across
32 sorter diverts (54 destination markets) and 8 manual lanes (A–H).

Run:
    pip install streamlit pandas
    streamlit run app.py

Planning horizon
----------------
Sorter throughput is expressed in parcels/hour, so one "plan" represents one
sort wave of roughly one hour. Divert (1,000) and lane capacities are per wave.

Placeholder data
----------------
Market names, the top-6 market list, and shipper names are illustrative.
Replace them in the NETWORK DATA section with real network data.
"""

import math
import random
from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st

# =============================================================================
# CONFIGURATION — default thresholds (all adjustable live in the sidebar)
# =============================================================================

# --- Routing rule thresholds -------------------------------------------------
SORTER_VOLUME_THRESHOLD = 150       # Rule 1: volume >= this -> Sorter candidate
CUTOFF_OVERRIDE_MINUTES = 120       # Rule 2: minutes to cutoff <= this -> Manual
VIP_SORTER_MIN_VOLUME = 75          # Rule 3: VIP with volume >= this -> Sorter
LARGE_BOX_MANUAL_MIN_VOLUME = 200   # Rule 5: large boxes >= this prefer Manual (unless VIP)

# --- Sorter throughput (parcels/hour) ----------------------------------------
SORTER_CAPACITY_SMALL_MEDIUM = 4800  # realistic range 4,500–5,000
SORTER_CAPACITY_LARGE = 3800         # realistic range 3,500–4,000

# How Rule 4 handles a wave that mixes box sizes:
#   "blended" – a large parcel consumes 4800/3800 ≈ 1.26 sorter "slots", so the
#               cumulative load is tracked in small/medium-equivalent units and
#               compared against the 4,800/hr ceiling. Handles mixed waves.
#   "literal" – exactly as the original spec: capacity = 3800 if THIS shipment
#               is large, else 4800, compared against cumulative raw units.
DEFAULT_CAPACITY_MODE = "blended"

# --- Sorter divert rules -----------------------------------------------------
DIVERT_MAX_UNITS = 1000        # no divert should exceed this expected load
SUBDIVERT_FLAG_UNITS = 600     # flag subdivert groups above this
MAX_MARKETS_PER_DIVERT = 6     # subdiverts feed 4–6 downstream sub-markets

# --- Manual lanes ------------------------------------------------------------
DEFAULT_MANUAL_LANES = 8       # Lanes A–H
MAX_MANUAL_LANES = 26          # Lanes A–Z
FRONT_LANE_COUNT = 3           # Lanes A–C sit closest; they get top manual markets
FRONT_LANE_MARKET_COUNT = 6    # number of highest-volume manual markets sent to A–C
MANUAL_LANE_CAPACITY = 600     # units per lane per wave (assumption; drives util %)
UNITS_PER_FTE = 400            # staffing: 1 FTE per 400 units
LANE_BALANCE_SLACK = 1.15      # a lane may run 15% above average before we look elsewhere

# --- Peak-day context --------------------------------------------------------
PEAK_DAY_PACKAGES = 185_000
PEAK_DAY_OPERATING_HOURS = 20  # assumption; used to express peak as an hourly rate

# --- Site sorter profiles ----------------------------------------------------
# Picking a profile in the sidebar loads these values; "Custom" keeps whatever
# is currently set. Edit or add profiles here.
SITE_PROFILES = {
    "Large sorter (32 diverts)": dict(has_sorter=True, cap_sm=SORTER_CAPACITY_SMALL_MEDIUM,
                                      cap_lg=SORTER_CAPACITY_LARGE, n_diverts=32, n_hv=6,
                                      n_lanes=DEFAULT_MANUAL_LANES),
    "Small sorter (16 diverts)": dict(has_sorter=True, cap_sm=2400, cap_lg=1900,
                                      n_diverts=16, n_hv=3, n_lanes=10),
    "No sorter (manual only)":   dict(has_sorter=False, cap_sm=4800, cap_lg=3800,
                                      n_diverts=32, n_hv=6, n_lanes=16),
    "Custom": None,
}
DEFAULT_PROFILE = "Large sorter (32 diverts)"
MIN_DIVERTS, MAX_DIVERTS = 2, 64

# =============================================================================
# NETWORK DATA — markets, regions, diverts, shippers (illustrative placeholders)
# =============================================================================

# 54 destination markets grouped by region (9 per region).
MARKETS_BY_REGION = {
    "Northeast": ["New York City", "Boston", "Providence", "Hartford", "Newark",
                  "Albany", "Buffalo", "Rochester", "Portland ME"],
    "Mid-Atlantic": ["Philadelphia", "Baltimore", "Washington DC", "Richmond",
                     "Pittsburgh", "Harrisburg", "Wilmington", "Norfolk", "Allentown"],
    "South": ["Atlanta", "Miami", "Orlando", "Tampa", "Charlotte", "Raleigh",
              "Nashville", "Jacksonville", "Memphis"],
    "Midwest": ["Chicago", "Detroit", "Columbus", "Cleveland", "Indianapolis",
                "Milwaukee", "Minneapolis", "St. Louis", "Kansas City"],
    "Southwest": ["Dallas", "Houston", "Austin", "San Antonio", "Phoenix",
                  "Tucson", "Albuquerque", "Oklahoma City", "El Paso"],
    "West": ["Los Angeles", "San Diego", "San Francisco", "Sacramento",
             "Las Vegas", "Denver", "Salt Lake City", "Seattle", "Portland OR"],
}
MARKET_TO_REGION = {m: r for r, ms in MARKETS_BY_REGION.items() for m in ms}
REGIONS = list(MARKETS_BY_REGION)

# Top-volume markets, ranked. The first N (N = number of high-velocity diverts)
# each get a dedicated divert that NEVER changes, regardless of the day's volume.
TOP_MARKETS = [
    "New York City", "Philadelphia", "Atlanta", "Chicago", "Dallas", "Los Angeles",
    "Houston", "Washington DC", "Miami", "Phoenix", "Boston", "Detroit",
]

assert len(MARKET_TO_REGION) == 54, "Network must cover 54 markets"


def build_divert_network(n_diverts, n_hv):
    """
    Build the divert layout for a site.

    Returns (hv, groups):
      hv     – {divert_id: market} for the fixed high-velocity diverts (D01..)
      groups – list of (label, [regions], [divert_ids]) for regional diverts.
               Regional diverts are shared out in proportion to each region's
               market count (at least one per region). If there are fewer
               regional diverts than regions, neighboring regions share one.
    """
    n_hv = max(0, min(n_hv, len(TOP_MARKETS), n_diverts - 1))  # keep >= 1 regional divert
    hv = {f"D{i + 1:02d}": TOP_MARKETS[i] for i in range(n_hv)}
    hv_markets = set(hv.values())
    n_reg = n_diverts - n_hv

    region_counts = {r: len([m for m in ms if m not in hv_markets])
                     for r, ms in MARKETS_BY_REGION.items()}
    regions = [r for r in REGIONS if region_counts[r] > 0]

    # Decide which regions share which number of diverts
    if n_reg < len(regions):
        base, extra = divmod(len(regions), n_reg)
        plan, i = [], 0
        for g in range(n_reg):
            size = base + (1 if g < extra else 0)
            plan.append((regions[i:i + size], 1))
            i += size
    else:
        total = sum(region_counts[r] for r in regions)
        spare = n_reg - len(regions)                     # 1 each guaranteed
        shares = {r: spare * region_counts[r] / total for r in regions}
        counts = {r: 1 + int(shares[r]) for r in regions}
        leftover = n_reg - sum(counts.values())
        for r in sorted(regions, key=lambda r: -(shares[r] - int(shares[r])))[:leftover]:
            counts[r] += 1
        plan = [([r], counts[r]) for r in regions]

    groups, n = [], n_hv + 1
    for regs, count in plan:
        ids = [f"D{n + k:02d}" for k in range(count)]
        n += count
        groups.append((" + ".join(regs), regs, ids))
    return hv, groups


def lane_ids(n_lanes):
    return [f"Lane {chr(ord('A') + i)}" for i in range(n_lanes)]

# Shipper -> default box size class (used when box size is set to "Auto").
SHIPPER_BOX_SIZE = {
    "Apparel Co": "small",
    "Beauty Box Co": "small",
    "Book & Media Co": "small",
    "Supplement Co": "small",
    "Meal Kit Co": "medium",
    "Electronics Co": "medium",
    "Outdoor Gear Co": "medium",
    "Home Goods Co": "large",
    "Pet Supply Co": "large",
    "Furniture Co": "large",
}

CUSTOMER_TIERS = ["VIP", "Standard", "Economy"]
TIER_PRIORITY = {"VIP": 0, "Standard": 1, "Economy": 2}  # tie-break in processing order
SERVICE_LEVELS = ["Same Day", "Next Day", "2-Day", "Standard Ground"]


# =============================================================================
# DATA HELPERS
# =============================================================================

def make_shipment(shipment_id, arrival_time, destination_market, customer_tier,
                  shipper, box_size_class, volume_units, cutoff_time,
                  service_level, special_handling):
    """Create one shipment record. Region is derived from the market."""
    return {
        "shipment_id": shipment_id,
        "arrival_time": arrival_time,
        "destination_market": destination_market,
        "region": MARKET_TO_REGION[destination_market],
        "customer_tier": customer_tier,
        "shipper": shipper,
        "box_size_class": box_size_class,
        "volume_units": int(volume_units),
        "cutoff_time": cutoff_time,
        "service_level": service_level,
        "special_handling": bool(special_handling),
    }


def generate_sample_shipments(n=60, seed=7):
    """Realistic-looking sample wave: skewed toward top markets, mixed tiers/sizes."""
    rng = random.Random(seed)
    base = datetime.combine(date.today(), time(5, 0))
    top_markets = TOP_MARKETS[:6]
    all_markets = list(MARKET_TO_REGION)
    shippers = list(SHIPPER_BOX_SIZE)
    rows = []
    for i in range(n):
        market = rng.choice(top_markets) if rng.random() < 0.35 else rng.choice(all_markets)
        shipper = rng.choice(shippers)
        tier = rng.choices(CUSTOMER_TIERS, weights=[0.2, 0.55, 0.25])[0]
        volume = rng.randint(40, 149) if rng.random() < 0.45 else rng.randint(150, 450)
        arrival = base + timedelta(minutes=rng.randint(0, 180))
        cutoff = arrival + timedelta(minutes=rng.choice([90, 150, 240, 300, 360, 480]))
        rows.append(make_shipment(
            f"SHP-{1001 + i}", arrival, market, tier, shipper,
            SHIPPER_BOX_SIZE[shipper], volume, cutoff,
            rng.choice(SERVICE_LEVELS), rng.random() < 0.1,
        ))
    return rows


# =============================================================================
# ROUTING DECISION ENGINE
# =============================================================================

def to_sorter_equivalent(volume, box_size_class, cfg):
    """Convert parcels into small/medium-equivalent sorter slots."""
    if box_size_class == "large":
        return volume * cfg["cap_small_medium"] / cfg["cap_large"]
    return float(volume)


def sorter_capacity_for(box_size_class, cfg):
    """Spec capacity logic: large boxes run slower through the sorter."""
    if box_size_class == "large":
        return cfg["cap_large"]
    return cfg["cap_small_medium"]


def run_routing_engine(shipments, cfg):
    """
    Apply Rules 1–5 in order to each shipment and return:
        routed DataFrame, cumulative raw sorter units, cumulative equivalent units

    Processing order: arrival time, then VIP first, then earliest cutoff.
    Sorter load is only committed once a shipment's FINAL decision is Sorter.
    """
    df = pd.DataFrame(shipments)
    df["tier_rank"] = df["customer_tier"].map(TIER_PRIORITY)
    df = df.sort_values(["arrival_time", "tier_rank", "cutoff_time"]).reset_index(drop=True)

    load_raw, load_eq = 0, 0.0
    rows = []

    for s in df.to_dict("records"):
        vol = int(s["volume_units"])
        box = s["box_size_class"]
        is_vip = s["customer_tier"] == "VIP"
        minutes_to_cutoff = (s["cutoff_time"] - s["arrival_time"]).total_seconds() / 60
        reasons = []

        # ---- Rule 0: site has no sorter -------------------------------------
        if not cfg["has_sorter"]:
            rows.append({
                **s,
                "minutes_to_cutoff": round(minutes_to_cutoff),
                "routing_decision": "Manual",
                "reasons": "R0 No sorter at this site → Manual",
                "cum_sorter_load": 0,
            })
            continue

        # ---- Rule 1: Volume threshold ---------------------------------------
        if vol >= cfg["vol_threshold"]:
            decision = "Sorter"
            reasons.append(f"R1 Volume {vol} ≥ {cfg['vol_threshold']} → Sorter candidate")
        else:
            decision = "Manual"
            reasons.append(f"R1 Volume {vol} < {cfg['vol_threshold']} → Manual candidate")

        # ---- Rule 2: Cutoff proximity ---------------------------------------
        if minutes_to_cutoff <= cfg["cutoff_minutes"]:
            note = "cutoff already passed" if minutes_to_cutoff < 0 else f"{minutes_to_cutoff:.0f} min to cutoff"
            action = "override to Manual" if decision == "Sorter" else "keep Manual"
            reasons.append(f"R2 Cutoff: {note} (≤ {cfg['cutoff_minutes']}) → {action}")
            decision = "Manual"

        # ---- Rule 3: Customer tier (VIP) ------------------------------------
        if is_vip and vol >= cfg["vip_min_volume"]:
            action = "override to Sorter" if decision == "Manual" else "confirm Sorter"
            reasons.append(f"R3 VIP with {vol} ≥ {cfg['vip_min_volume']} → {action}")
            decision = "Sorter"

        # ---- Rule 4: Sorter capacity ----------------------------------------
        if decision == "Sorter":
            if cfg["capacity_mode"] == "blended":
                need = to_sorter_equivalent(vol, box, cfg)
                cap = cfg["cap_small_medium"]
                if load_eq + need > cap:
                    decision = "Manual"
                    reasons.append(
                        f"R4 Capacity: {load_eq:,.0f} + {need:,.0f} eq. units > {cap:,}/hr → override to Manual")
            else:
                cap = sorter_capacity_for(box, cfg)
                if load_raw + vol > cap:
                    decision = "Manual"
                    reasons.append(
                        f"R4 Capacity: {load_raw:,} + {vol} > {cap:,}/hr ({box}) → override to Manual")

        # ---- Rule 5: Box size / shipper -------------------------------------
        if box == "large" and vol >= cfg["large_min_volume"]:
            if decision == "Sorter":
                if is_vip:
                    reasons.append(f"R5 Large box ({s['shipper']}) ≥ {cfg['large_min_volume']}, VIP exception → keep Sorter")
                else:
                    decision = "Manual"
                    reasons.append(f"R5 Large box ({s['shipper']}) ≥ {cfg['large_min_volume']} → prefer Manual")
            else:
                reasons.append(f"R5 Large box ({s['shipper']}) ≥ {cfg['large_min_volume']} → Manual preferred (already Manual)")

        # ---- Optional Rule 6: Special handling (off by default) -------------
        if cfg["special_handling_rule"] and s["special_handling"] and decision == "Sorter":
            decision = "Manual"
            reasons.append("R6 Special handling → override to Manual (optional rule)")

        # Commit sorter load only for final Sorter decisions
        if decision == "Sorter":
            load_raw += vol
            load_eq += to_sorter_equivalent(vol, box, cfg)

        rows.append({
            **s,
            "minutes_to_cutoff": round(minutes_to_cutoff),
            "routing_decision": decision,
            "reasons": " | ".join(reasons),
            "cum_sorter_load": round(load_eq if cfg["capacity_mode"] == "blended" else load_raw),
        })

    routed = pd.DataFrame(rows).drop(columns=["tier_rank"])
    return routed, load_raw, load_eq


# =============================================================================
# SORTER DIVERT ASSIGNMENT
# =============================================================================

def assign_diverts(routed, cfg):
    """
    1. Top markets -> their fixed high-velocity divert (never changes).
    2. Other markets -> diverts within their region group, highest volume first,
       each going to the least-loaded divert that stays under the per-divert
       cap (i.e. when a divert is full, move to the next one).
    3. Zero-volume markets are still mapped so all 54 markets are covered.
    Returns an empty table when the site has no sorter.
    """
    if not cfg["has_sorter"]:
        return pd.DataFrame()

    hv_map, groups = build_divert_network(cfg["n_diverts"], cfg["n_hv"])
    hv_markets = set(hv_map.values())
    sorter = routed[routed["routing_decision"] == "Sorter"]
    market_vol = sorter.groupby("destination_market")["volume_units"].sum().to_dict()
    cap = cfg["divert_max_units"]
    max_mkts = cfg["max_markets_per_divert"]
    diverts = {}

    # High-velocity (fixed) diverts
    for d, m in hv_map.items():
        v = int(market_vol.get(m, 0))
        diverts[d] = {"type": "High-velocity (fixed)", "region": MARKET_TO_REGION[m],
                      "markets": {m: v}, "volume": v, "overflow": []}

    # Regional diverts
    for label, regs, ids in groups:
        for d in ids:
            diverts[d] = {"type": "Regional", "region": label,
                          "markets": {}, "volume": 0, "overflow": []}

        group_markets = [m for r in regs for m in MARKETS_BY_REGION[r] if m not in hv_markets]
        with_volume = sorted([m for m in group_markets if market_vol.get(m, 0) > 0],
                             key=lambda m: -market_vol[m])
        without_volume = [m for m in group_markets if market_vol.get(m, 0) == 0]

        for m in with_volume:
            v = int(market_vol[m])
            open_diverts = [d for d in ids if len(diverts[d]["markets"]) < max_mkts] or ids
            fits = [d for d in open_diverts if diverts[d]["volume"] + v <= cap]
            if fits:
                target = min(fits, key=lambda d: diverts[d]["volume"])  # balance load
            else:
                target = min(open_diverts, key=lambda d: diverts[d]["volume"])
                diverts[target]["overflow"].append(m)
            diverts[target]["markets"][m] = v
            diverts[target]["volume"] += v

        for m in without_volume:
            open_diverts = [d for d in ids if len(diverts[d]["markets"]) < max_mkts] or ids
            target = min(open_diverts, key=lambda d: (len(diverts[d]["markets"]), diverts[d]["volume"]))
            diverts[target]["markets"][m] = 0

    # Smallest active primary (high-velocity) divert, for the subdivert check
    hv_active = [diverts[d]["volume"] for d in hv_map if diverts[d]["volume"] > 0]
    min_primary = min(hv_active) if hv_active else None

    rows = []
    for d in sorted(diverts):
        info = diverts[d]
        n_markets = len(info["markets"])
        is_subdivert = info["type"] == "Regional" and n_markets > 1
        if info["type"] == "Regional":
            info["type"] = "Regional – subdivert group" if is_subdivert else "Regional – single market"

        flags = []
        if info["volume"] > cap:
            if d in hv_map:
                flags.append(f"Over {cap:,} cap (fixed divert: meter inbound or spill to manual)")
            else:
                flags.append(f"Over {cap:,} cap: {', '.join(info['overflow']) or 'region full'}")
        if n_markets > max_mkts:
            flags.append(f"{n_markets} markets > {max_mkts} max (add diverts)")
        if is_subdivert and info["volume"] > cfg["subdivert_flag_units"]:
            flags.append(f"Subdivert > {cfg['subdivert_flag_units']}")
        if is_subdivert and min_primary is not None and info["volume"] >= min_primary:
            flags.append("Subdivert ≥ smallest primary divert")

        markets_str = ", ".join(f"{m} ({v:,})" for m, v in
                                sorted(info["markets"].items(), key=lambda kv: -kv[1]))
        rows.append({
            "Divert ID": d,
            "Type": info["type"],
            "Region": info["region"],
            "Assigned Markets": markets_str,
            "# Markets": n_markets,
            "Expected Volume": info["volume"],
            "Utilization %": round(100 * info["volume"] / cap, 1),
            "Flags": "; ".join(flags) if flags else "OK",
        })
    return pd.DataFrame(rows)


# =============================================================================
# MANUAL LANE ASSIGNMENT
# =============================================================================

def assign_manual_lanes(routed, cfg):
    """
    1. The highest-volume manual markets go to front lanes (A–C) to cut walking.
       A market too big for one lane is split across several lanes.
    2. Remaining markets are placed region by region (biggest region first) so a
       region's markets cluster in the same lane, while keeping lanes balanced
       near the average load.
    3. Staffing = ceil(volume / units_per_fte).
    """
    manual = routed[routed["routing_decision"] == "Manual"]
    all_lanes = lane_ids(cfg["n_lanes"])
    lanes = {lid: {"markets": {}, "regions": set(), "volume": 0} for lid in all_lanes}
    n_front = min(FRONT_LANE_COUNT, len(all_lanes) - 1) if len(all_lanes) > 1 else 1
    front = all_lanes[:n_front]
    back = all_lanes[n_front:] or front

    if not manual.empty:
        market_vol = (manual.groupby("destination_market")["volume_units"].sum()
                      .sort_values(ascending=False))
        target = market_vol.sum() / len(all_lanes)
        ceiling = target * LANE_BALANCE_SLACK

        def place_one(label, region, vol, pool):
            same_region = [l for l in pool
                           if region in lanes[l]["regions"] and lanes[l]["volume"] + vol <= ceiling]
            lane = min(same_region or pool, key=lambda l: lanes[l]["volume"])
            lanes[lane]["markets"][label] = int(vol)
            lanes[lane]["regions"].add(region)
            lanes[lane]["volume"] += int(vol)

        def place(market, vol, pool):
            """A market too big for one lane is split across several lanes."""
            region = MARKET_TO_REGION[market]
            k = min(len(all_lanes), math.ceil(vol / target)) if vol > ceiling else 1
            if k <= 1:
                place_one(market, region, vol, pool)
                return
            base, extra = divmod(int(vol), k)
            for i in range(k):
                part = base + (1 if i < extra else 0)
                place_one(f"{market} [{i + 1}/{k}]", region, part, pool if i == 0 else all_lanes)

        # Step 1: top manual markets -> front lanes, as long as a front lane is
        # empty or stays under the balance ceiling (keeps A–C from overloading)
        top = []
        for m in market_vol.index[:cfg["front_lane_markets"]]:
            lightest_front = min(front, key=lambda l: lanes[l]["volume"])
            if lanes[lightest_front]["volume"] == 0 or lanes[lightest_front]["volume"] + market_vol[m] <= ceiling:
                place(m, market_vol[m], front)
                top.append(m)

        # Step 2: everything else, grouped by region
        rest = market_vol.drop(top)
        if not rest.empty:
            rest_regions = rest.index.map(MARKET_TO_REGION)
            region_totals = rest.groupby(rest_regions).sum().sort_values(ascending=False)
            for region in region_totals.index:
                region_markets = rest[rest_regions == region].sort_values(ascending=False)
                for m, v in region_markets.items():
                    lightest_back = min(back, key=lambda l: lanes[l]["volume"])
                    lightest_front = min(front, key=lambda l: lanes[l]["volume"])
                    back_full = lanes[lightest_back]["volume"] + v > ceiling
                    front_lighter = lanes[lightest_front]["volume"] < lanes[lightest_back]["volume"]
                    pool = all_lanes if (back_full and front_lighter) else back
                    place(m, v, pool)

    rows = []
    for lid in all_lanes:
        info = lanes[lid]
        vol = info["volume"]
        rows.append({
            "Lane ID": lid,
            "Zone": "Front (high-volume)" if lid in front else "Back",
            "Regions": ", ".join(sorted(info["regions"])) or "—",
            "Assigned Markets": ", ".join(f"{m} ({v:,})" for m, v in
                                          sorted(info["markets"].items(), key=lambda kv: -kv[1])) or "—",
            "Expected Volume": vol,
            "Utilization %": round(100 * vol / cfg["manual_lane_capacity"], 1),
            "Suggested Staffing (FTE)": math.ceil(vol / cfg["units_per_fte"]) if vol > 0 else 0,
        })
    return pd.DataFrame(rows)


# =============================================================================
# STREAMLIT UI
# =============================================================================

def init_state():
    if "shipments" not in st.session_state:
        st.session_state.shipments = []
    if "results" not in st.session_state:
        st.session_state.results = None
    if "results_sig" not in st.session_state:
        st.session_state.results_sig = None
    if "site_profile" not in st.session_state:
        st.session_state.site_profile = DEFAULT_PROFILE
        for k, v in SITE_PROFILES[DEFAULT_PROFILE].items():
            st.session_state[k] = v


def apply_profile():
    """Load a site profile's sorter settings into the sidebar widgets."""
    profile = SITE_PROFILES[st.session_state.site_profile]
    if profile:
        for k, v in profile.items():
            st.session_state[k] = v


def input_signature(cfg):
    """Detects whether inputs changed since the last engine run."""
    ids = tuple(s["shipment_id"] for s in st.session_state.shipments)
    return (tuple(sorted(cfg.items())), ids)


def sidebar_config():
    st.sidebar.header("Rule settings")
    st.sidebar.caption("Defaults come from the CONFIGURATION block in app.py.")
    cfg = {}
    ss = st.session_state

    with st.sidebar.expander("Site sorter setup", expanded=True):
        st.selectbox("Site profile", list(SITE_PROFILES), key="site_profile", on_change=apply_profile,
                     help="Loads throughput, divert and lane counts. Change any value below to fine-tune.")
        cfg["has_sorter"] = st.checkbox("Site has a sorter", key="has_sorter")
        no_sorter = not cfg["has_sorter"]
        cfg["cap_small_medium"] = st.number_input(
            "Sorter throughput, small/medium (parcels/hr)", 500, 20000, step=100, key="cap_sm", disabled=no_sorter)
        cfg["cap_large"] = st.number_input(
            "Sorter throughput, large (parcels/hr)", 500, 20000, step=100, key="cap_lg", disabled=no_sorter)
        cfg["n_diverts"] = st.number_input(
            "Number of diverts", MIN_DIVERTS, MAX_DIVERTS, step=1, key="n_diverts", disabled=no_sorter)
        max_hv = min(len(TOP_MARKETS), cfg["n_diverts"] - 1)
        if ss.n_hv > max_hv:
            ss.n_hv = max_hv  # keep at least one regional divert
        cfg["n_hv"] = st.number_input(
            "High-velocity (fixed) diverts", 0, max_hv, step=1, key="n_hv", disabled=no_sorter,
            help=f"Each serves one top market, in rank order: {', '.join(TOP_MARKETS)}.")
        cfg["n_lanes"] = st.number_input("Manual lanes", 1, MAX_MANUAL_LANES, step=1, key="n_lanes")
        if no_sorter:
            st.caption("No sorter: every shipment routes to manual lanes.")
        else:
            regional = cfg["n_diverts"] - cfg["n_hv"]
            st.caption(f"{cfg['n_hv']} fixed + {regional} regional diverts covering 54 markets.")

    with st.sidebar.expander("Routing rules", expanded=True):
        cfg["vol_threshold"] = st.number_input("R1 · Sorter volume threshold", 1, 5000, SORTER_VOLUME_THRESHOLD, 10)
        cfg["cutoff_minutes"] = st.number_input("R2 · Cutoff override (minutes)", 0, 1440, CUTOFF_OVERRIDE_MINUTES, 15)
        cfg["vip_min_volume"] = st.number_input("R3 · VIP sorter minimum volume", 1, 5000, VIP_SORTER_MIN_VOLUME, 5)
        cfg["large_min_volume"] = st.number_input("R5 · Large-box manual threshold", 1, 5000, LARGE_BOX_MANUAL_MIN_VOLUME, 10)
        cfg["special_handling_rule"] = st.checkbox(
            "R6 · Send special handling to Manual (optional)", value=False,
            help="Not in the original rule set. Off by default.")
    with st.sidebar.expander("Sorter capacity mode"):
        cfg["capacity_mode"] = st.radio(
            "R4 capacity mode", ["blended", "literal"],
            index=["blended", "literal"].index(DEFAULT_CAPACITY_MODE),
            help="Blended converts large parcels to small/medium-equivalent slots. "
                 "Literal uses the per-shipment capacity exactly as specified.")
    with st.sidebar.expander("Diverts & lanes"):
        cfg["divert_max_units"] = st.number_input("Max units per divert", 100, 5000, DIVERT_MAX_UNITS, 50)
        cfg["subdivert_flag_units"] = st.number_input("Subdivert flag above", 100, 5000, SUBDIVERT_FLAG_UNITS, 50)
        cfg["max_markets_per_divert"] = st.number_input("Max markets per divert", 1, 10, MAX_MARKETS_PER_DIVERT, 1)
        cfg["front_lane_markets"] = st.number_input("Top manual markets sent to front lanes", 0, 20, FRONT_LANE_MARKET_COUNT, 1)
        cfg["manual_lane_capacity"] = st.number_input("Manual lane capacity (units/wave)", 50, 5000, MANUAL_LANE_CAPACITY, 50)
        cfg["units_per_fte"] = st.number_input("Units per FTE", 50, 2000, UNITS_PER_FTE, 25)
    with st.sidebar.expander("Peak-day context"):
        cfg["peak_day_packages"] = st.number_input("Peak-day packages", 1000, 1_000_000, PEAK_DAY_PACKAGES, 5000)
        cfg["peak_ops_hours"] = st.number_input("Operating hours on peak day", 1, 24, PEAK_DAY_OPERATING_HOURS, 1)
    return cfg


def render_input_tab():
    st.subheader("Add an inbound shipment")
    shipments = st.session_state.shipments
    existing_ids = {s["shipment_id"] for s in shipments}

    with st.form("add_shipment"):
        c1, c2, c3 = st.columns(3)
        with c1:
            shipment_id = st.text_input("Shipment ID", value=f"SHP-{1001 + len(shipments)}")
            market = st.selectbox("Destination market", sorted(MARKET_TO_REGION),
                                  format_func=lambda m: f"{m} ({MARKET_TO_REGION[m]})")
            tier = st.selectbox("Customer tier", CUSTOMER_TIERS, index=1)
            service_level = st.selectbox("Service level", SERVICE_LEVELS, index=1)
        with c2:
            shipper = st.selectbox("Shipper", list(SHIPPER_BOX_SIZE))
            box_choice = st.selectbox("Box size class", ["Auto (from shipper)", "small", "medium", "large"])
            volume = st.number_input("Volume units", min_value=1, max_value=20000, value=200, step=10)
            special = st.checkbox("Special handling")
        with c3:
            arr_d = st.date_input("Arrival date", value=date.today())
            arr_t = st.time_input("Arrival time", value=time(6, 0), step=900)
            cut_d = st.date_input("Cutoff date", value=date.today(), key="cut_d")
            cut_t = st.time_input("Cutoff time", value=time(10, 0), step=900, key="cut_t")
        submitted = st.form_submit_button("Add Shipment", type="primary")

    if submitted:
        arrival = datetime.combine(arr_d, arr_t)
        cutoff = datetime.combine(cut_d, cut_t)
        box = SHIPPER_BOX_SIZE[shipper] if box_choice.startswith("Auto") else box_choice
        if not shipment_id.strip():
            st.error("Enter a shipment ID.")
        elif shipment_id in existing_ids:
            st.error(f"{shipment_id} already exists. Use a unique shipment ID.")
        else:
            shipments.append(make_shipment(shipment_id.strip(), arrival, market, tier, shipper,
                                           box, volume, cutoff, service_level, special))
            st.success(f"Added {shipment_id}: {volume} units to {market} "
                       f"({box} boxes, region {MARKET_TO_REGION[market]}).")
            if cutoff <= arrival:
                st.warning("Cutoff is at or before arrival. Rule 2 will force this shipment to Manual.")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Load sample wave (60 shipments)"):
        st.session_state.shipments = generate_sample_shipments()
        st.rerun()
    if b2.button("Clear all shipments"):
        st.session_state.shipments = []
        st.session_state.results = None
        st.rerun()

    st.subheader(f"Inbound shipments ({len(shipments)})")
    if not shipments:
        st.info("No shipments yet. Add one above or load the sample wave.")
        return

    df = pd.DataFrame(shipments)
    show = df.copy()
    show["arrival_time"] = show["arrival_time"].dt.strftime("%m-%d %H:%M")
    show["cutoff_time"] = show["cutoff_time"].dt.strftime("%m-%d %H:%M")
    st.dataframe(show, hide_index=True)
    st.caption(f"Total inbound volume: {df['volume_units'].sum():,} units")

    to_remove = st.multiselect("Remove shipments", df["shipment_id"].tolist())
    if to_remove and st.button("Remove selected"):
        st.session_state.shipments = [s for s in shipments if s["shipment_id"] not in to_remove]
        st.rerun()


def render_routing_tab(cfg):
    st.subheader("Routing decision engine")
    st.markdown(
        f"Rules run in order for each shipment (sorted by arrival, then VIP first, then cutoff):\n"
        f"1. **Volume** ≥ {cfg['vol_threshold']} → Sorter, else Manual\n"
        f"2. **Cutoff** ≤ {cfg['cutoff_minutes']} min away → Manual\n"
        f"3. **VIP** with ≥ {cfg['vip_min_volume']} units → Sorter\n"
        f"4. **Sorter capacity** would be exceeded → Manual ({cfg['capacity_mode']} mode)\n"
        f"5. **Large boxes** with ≥ {cfg['large_min_volume']} units → Manual unless VIP"
    )

    if not cfg["has_sorter"]:
        st.info("This site is set up with no sorter, so every shipment routes to manual lanes (Rule 0).")

    if st.button("Run Routing Decision Engine", type="primary",
                 disabled=not st.session_state.shipments):
        routed, load_raw, load_eq = run_routing_engine(st.session_state.shipments, cfg)
        st.session_state.results = {
            "routed": routed,
            "load_raw": load_raw,
            "load_eq": load_eq,
            "diverts": assign_diverts(routed, cfg),
            "lanes": assign_manual_lanes(routed, cfg),
        }
        st.session_state.results_sig = input_signature(cfg)

    if not st.session_state.shipments:
        st.info("Add shipments in the first tab, then run the engine.")
        return
    res = st.session_state.results
    if res is None:
        st.info("Run the engine to see routing decisions.")
        return
    if st.session_state.results_sig != input_signature(cfg):
        st.warning("Shipments or rule settings changed since the last run. Run the engine again to refresh.")

    routed = res["routed"]
    out = routed.rename(columns={
        "shipment_id": "Shipment ID", "destination_market": "Market", "region": "Region",
        "volume_units": "Volume", "routing_decision": "Routing Decision",
        "reasons": "Reasons (rules fired)", "customer_tier": "Tier",
        "box_size_class": "Box", "minutes_to_cutoff": "Min to Cutoff",
        "cum_sorter_load": "Cum. Sorter Load",
    })[["Shipment ID", "Market", "Region", "Volume", "Tier", "Box", "Min to Cutoff",
        "Routing Decision", "Cum. Sorter Load", "Reasons (rules fired)"]]

    decision_filter = st.radio("Show", ["All", "Sorter", "Manual"], horizontal=True)
    if decision_filter != "All":
        out = out[out["Routing Decision"] == decision_filter]
    st.dataframe(out, hide_index=True, column_config={
        "Reasons (rules fired)": st.column_config.TextColumn(width="large"),
    })

    counts = routed["routing_decision"].value_counts()
    fired = routed["reasons"].str.extractall(r"(R\d)")[0].value_counts().sort_index()
    c1, c2 = st.columns(2)
    c1.markdown(f"**Sorter:** {counts.get('Sorter', 0)} shipments · "
                f"**Manual:** {counts.get('Manual', 0)} shipments")
    c2.markdown("**Rule fire counts:** " + ", ".join(f"{r}: {n}" for r, n in fired.items()))
    st.download_button("Download routing decisions (CSV)", out.to_csv(index=False),
                       "routing_decisions.csv", "text/csv")


def render_plan_tab(cfg):
    res = st.session_state.results
    if res is None:
        st.info("Run the routing engine to generate the execution plan.")
        return
    if st.session_state.results_sig != input_signature(cfg):
        st.warning("Inputs changed since the last run. This plan reflects the previous run.")

    routed, diverts, lanes = res["routed"], res["diverts"], res["lanes"]
    total = int(routed["volume_units"].sum())
    sorter_vol = int(routed.loc[routed["routing_decision"] == "Sorter", "volume_units"].sum())
    manual_vol = total - sorter_vol
    has_sorter = not diverts.empty
    sorter_load = res["load_eq"] if cfg["capacity_mode"] == "blended" else res["load_raw"]
    sorter_pct = 100 * sorter_load / cfg["cap_small_medium"]
    n_lanes = len(lanes)
    lanes_active = int((lanes["Expected Volume"] > 0).sum())
    manual_pct = 100 * manual_vol / (n_lanes * cfg["manual_lane_capacity"])

    # ---- Summary -----------------------------------------------------------
    st.subheader("Wave summary")
    m = st.columns(5)
    m[0].metric("Total inbound", f"{total:,}")
    m[1].metric("Sorter volume", f"{sorter_vol:,}", f"{100 * sorter_vol / total:.0f}% of inbound" if total else None,
                delta_color="off")
    m[2].metric("Manual volume", f"{manual_vol:,}", f"{100 * manual_vol / total:.0f}% of inbound" if total else None,
                delta_color="off")
    if has_sorter:
        m[3].metric("Sorter capacity used", f"{sorter_pct:.0f}%",
                    f"{sorter_load:,.0f} / {cfg['cap_small_medium']:,} per hr", delta_color="off")
    else:
        m[3].metric("Sorter capacity used", "No sorter")
    m[4].metric("Manual lanes used", f"{lanes_active} of {n_lanes}",
                f"{manual_pct:.0f}% of lane capacity", delta_color="off")

    with st.expander("Big-day context", expanded=True):
        peak_hourly = cfg["peak_day_packages"] / cfg["peak_ops_hours"]
        sorter_rate = cfg["cap_small_medium"] if has_sorter else 0
        manual_share_needed = max(0.0, 1 - sorter_rate / peak_hourly)
        lanes_needed = math.ceil(peak_hourly * manual_share_needed / cfg["manual_lane_capacity"])
        if has_sorter:
            sorter_line = (f"- The sorter tops out near **{cfg['cap_small_medium']:,}/hr** (small/medium) and "
                           f"**{cfg['cap_large']:,}/hr** (large), so at peak roughly "
                           f"**{100 * manual_share_needed:.0f}%** of hourly volume must go manual or be metered.\n")
        else:
            sorter_line = "- This site has **no sorter**, so all peak volume is sorted manually.\n"
        wave_line = (f"- This wave is **{100 * total / peak_hourly:.0f}%** of a peak hour and "
                     f"**{100 * total / cfg['peak_day_packages']:.1f}%** of a peak day. "
                     f"Manual share this wave: **{100 * manual_vol / total:.0f}%**." if total else "")
        st.markdown(
            f"- A peak day runs **{cfg['peak_day_packages']:,} packages**, about "
            f"**{peak_hourly:,.0f}/hr** over {cfg['peak_ops_hours']} operating hours.\n"
            + sorter_line
            + f"- At {cfg['manual_lane_capacity']:,} units per lane per hour, peak needs about "
              f"**{lanes_needed} manual lanes** (this site is set up with {n_lanes}).\n"
            + wave_line
        )

    # ---- Sorter divert plan -------------------------------------------------
    if has_sorter:
        render_divert_plan(diverts)
    else:
        st.subheader("Sorter divert plan")
        st.info("No sorter at this site. All volume is planned in the manual lanes below.")

    # ---- Manual sortation plan ---------------------------------------------
    render_lane_plan(lanes, cfg)


def render_divert_plan(diverts):
    st.subheader(f"Sorter divert plan ({len(diverts)} diverts, 54 markets)")
    flagged = diverts[diverts["Flags"] != "OK"]
    if not flagged.empty:
        st.warning(f"{len(flagged)} divert(s) flagged: " + ", ".join(flagged["Divert ID"]))
    st.dataframe(diverts, hide_index=True, column_config={
        "Assigned Markets": st.column_config.TextColumn(width="large"),
        "Utilization %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
    })
    st.markdown("**Volume per divert**")
    st.bar_chart(diverts.set_index("Divert ID")["Expected Volume"])
    st.download_button("Download divert plan (CSV)", diverts.to_csv(index=False),
                       "divert_plan.csv", "text/csv")


def render_lane_plan(lanes, cfg):
    st.subheader(f"Manual sortation plan ({lanes['Lane ID'].iloc[0]} to {lanes['Lane ID'].iloc[-1]})")
    st.dataframe(lanes, hide_index=True, column_config={
        "Assigned Markets": st.column_config.TextColumn(width="large"),
        "Utilization %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
    })
    st.caption(f"Total suggested manual staffing: {int(lanes['Suggested Staffing (FTE)'].sum())} FTE "
               f"(1 FTE per {cfg['units_per_fte']} units).")
    st.markdown("**Volume per manual lane**")
    st.bar_chart(lanes.set_index("Lane ID")["Expected Volume"])
    st.download_button("Download manual lane plan (CSV)", lanes.to_csv(index=False),
                       "manual_lane_plan.csv", "text/csv")


def main():
    st.set_page_config(page_title="Veho Inbound Routing Planner", page_icon="📦", layout="wide")
    init_state()
    cfg = sidebar_config()

    st.title("Veho inbound routing & execution planner")
    st.caption("Decide sorter vs. manual for each inbound shipment, then plan diverts and manual lanes "
               "for one sort wave (~1 hour of sorter throughput).")

    tab1, tab2, tab3 = st.tabs(["Inbound shipments", "Routing engine", "Execution plan"])
    with tab1:
        render_input_tab()
    with tab2:
        render_routing_tab(cfg)
    with tab3:
        render_plan_tab(cfg)


if __name__ == "__main__":
    main()
