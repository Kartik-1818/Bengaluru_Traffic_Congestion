
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Bengaluru Traffic Response v2",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0f1117; }
  .metric-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 4px solid var(--card-color, #4a90d9);
    margin-bottom: 12px;
  }
  .metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 28px; font-weight: 700; color: #fff; margin-top: 4px; }
  .metric-sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
  .conf-bar-bg  { background: #2a2e3e; border-radius: 8px; height: 12px; margin-top: 6px; }
  .conf-bar     { border-radius: 8px; height: 12px; transition: width 0.5s; }
  .badge        { display: inline-block; padding: 4px 14px; border-radius: 14px;
                  font-weight: 700; font-size: 14px; }
  .section-header { font-size: 13px; color: #888; text-transform: uppercase;
                    letter-spacing: 1px; margin: 20px 0 10px 0; border-bottom: 1px solid #2a2e3e; padding-bottom: 6px; }
  .alert-box { border-radius: 10px; padding: 14px 18px; margin: 10px 0; font-size: 14px; }
  .stButton > button { border-radius: 10px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ──── Load Model Bundle ──────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    # Try v2 first (XGBoost + feature engineering)
    v2_path = 'models/recommendation_engine_bundle_v2.pkl'
    v1_path = 'models/recommendation_engine_bundle.pkl'
    if os.path.exists(v2_path):
        B = joblib.load(v2_path)
        B['_version'] = 'v2'
    else:
        B = joblib.load(v1_path)
        B['_version'] = 'v1'
    return B

B = load_bundle()
MODEL_VER = B.get('_version', 'v1')

# ──── Feature Engineering ────────────────────────────────────────────────────
HOTSPOTS = B.get('hotspots', {
    'mg_road':        (12.9766, 77.6075),
    'silk_board':     (12.9174, 77.6228),
    'hebbal':         (13.0358, 77.5970),
    'marathahalli':   (12.9563, 77.7010),
    'whitefield':     (12.9698, 77.7500),
    'electronic_city':(12.8399, 77.6770),
})

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def engineer_event(event: dict) -> dict:
    """Add engineered features to a raw event dict."""
    ev = event.copy()
    h = ev.get('hour', 12)
    m = ev.get('month_num', 6)
    dow = ev.get('day_of_week', 0)
    lat = ev['latitude']
    lon = ev['longitude']

    ev['hour_sin']    = np.sin(2 * np.pi * h / 24)
    ev['hour_cos']    = np.cos(2 * np.pi * h / 24)
    ev['month_sin']   = np.sin(2 * np.pi * m / 12)
    ev['month_cos']   = np.cos(2 * np.pi * m / 12)
    ev['dow_sin']     = np.sin(2 * np.pi * dow / 7)
    ev['dow_cos']     = np.cos(2 * np.pi * dow / 7)

    cause_score = B['cause_score_map'].get(ev.get('event_cause', 'others'), 2)
    ev['cause_score'] = cause_score

    dists = [haversine_km(lat, lon, hlat, hlon) for hlat, hlon in HOTSPOTS.values()]
    for name, d in zip(HOTSPOTS.keys(), dists):
        ev[f'dist_{name}'] = d
    ev['min_hotspot_dist'] = min(dists)
    ev['nearest_hotspot']  = list(HOTSPOTS.keys())[int(np.argmin(dists))]

    ev['peak_x_cause']    = ev.get('is_peak_hour', 0) * cause_score
    ev['weekend_x_cause'] = ev.get('is_weekend', 0)   * cause_score
    return ev

def nearest_police_station(lat, lon, k=5):
    sc = B['station_coords']
    d = haversine_km(lat, lon, sc['latitude'].values, sc['longitude'].values)
    idx = np.argsort(d)[:k]
    return sc.iloc[idx]['police_station'].mode()[0], round(d[idx].min(), 2)

def nearest_corridor(lat, lon, k=5):
    cc = B['corridor_coords']
    d = haversine_km(lat, lon, cc['latitude'].values, cc['longitude'].values)
    idx = np.argsort(d)[:k]
    return cc.iloc[idx]['corridor'].mode()[0]

def risk_bucket(score):
    if score <= 4:  return 'Low'
    elif score <= 7: return 'Medium'
    elif score <= 9: return 'High'
    else:            return 'Critical'

def encode_event(event_dict, cat_cols, num_cols, target_columns, extra=None):
    row = {c: event_dict.get(c) for c in cat_cols + num_cols}
    if extra:
        row.update(extra)
    df = pd.DataFrame([row])
    enc = pd.get_dummies(df, columns=cat_cols)
    return enc.reindex(columns=target_columns, fill_value=0)

# ──── Inference Engine ───────────────────────────────────────────────────────
def recommend_resources(raw_event: dict) -> dict:
    ev = raw_event.copy()
    # Defaults
    ev.setdefault('zone', 'Unknown')
    ev.setdefault('veh_type', 'unknown')
    ev.setdefault('day_of_week', 0)
    ev.setdefault('is_weekend',  1 if ev.get('day_of_week', 0) in [5, 6] else 0)
    ev.setdefault('is_peak_hour',1 if ev.get('hour', 12) in [7,8,9,17,18,19,20] else 0)
    ev.setdefault('is_night',    1 if (ev.get('hour', 12) >= 22 or ev.get('hour', 12) < 6) else 0)

    # Auto-detect corridor
    ev['corridor'] = nearest_corridor(ev['latitude'], ev['longitude'])

    # Feature engineering (v2 only)
    if MODEL_VER == 'v2':
        ev = engineer_event(ev)

    # ── Priority Model ──
    Xp = encode_event(ev, B['cat_features_fixed'], B['num_features_fixed'], B['priority_feature_cols'])
    p_proba = B['priority_model'].predict_proba(Xp)[0]
    prob_high = float(p_proba[1])
    pred_priority = 'High' if prob_high >= 0.5 else 'Low'
    priority_conf = max(prob_high, 1 - prob_high)

    # ── Closure Model ──
    Xc = encode_event(ev, B['cat_features'], B['num_features'], B['closure_feature_cols'])
    c_proba = B['closure_model'].predict_proba(Xc)[0]
    prob_closure = float(c_proba[1])
    threshold = B.get('closure_threshold', 0.5)
    pred_closure = prob_closure >= threshold
    closure_conf = max(prob_closure, 1 - prob_closure)

    # ── Severity Score ──
    cause_w    = B['cause_score_map'].get(ev.get('event_cause'), 1)
    type_w     = 2 if ev.get('event_type') == 'planned' else 1
    priority_w = 2 if pred_priority == 'High' else 1
    closure_w  = 2 if pred_closure else 0
    est_severity = cause_w + type_w + priority_w + closure_w

    # ── Duration Model ──
    Xd = encode_event(ev, B['cat_features'], B['num_features'], B['duration_feature_cols'],
                      extra={'severity_index': est_severity})
    pred_dur_hrs = float(np.expm1(B['duration_model'].predict(Xd)[0]))
    pred_dur_hrs = max(0.1, min(pred_dur_hrs, 24.0))

    # ── Resource Recommendation ──
    officers = B['manpower_map'].get(est_severity, max(1, est_severity - 1))
    risk = risk_bucket(est_severity)

    if ev.get('zone') not in ('Unknown', None) and ev['zone'] in B['zone_station_map']:
        station = B['zone_station_map'][ev['zone']]
        method = f"historical zone lookup ({ev['zone']})"
    else:
        station, dist = nearest_police_station(ev['latitude'], ev['longitude'])
        method = f"GPS nearest-neighbor (~{dist} km)"

    return {
        'predicted_priority':     pred_priority,
        'priority_confidence':    round(priority_conf, 3),
        'prob_high':              round(prob_high, 3),
        'predicted_road_closure': pred_closure,
        'closure_probability':    round(prob_closure, 3),
        'closure_confidence':     round(closure_conf, 3),
        'predicted_duration_hours': round(pred_dur_hrs, 2),
        'estimated_severity_score': est_severity,
        'risk_level':             risk,
        'recommended_officers':   officers,
        'recommend_barricading':  pred_closure,
        'recommended_police_station': station,
        'station_assignment_method': method,
        'detected_corridor':      ev['corridor'],
        'nearest_hotspot':        ev.get('nearest_hotspot', '—'),
        'min_hotspot_dist':       round(ev.get('min_hotspot_dist', 0), 2),
        'cause_score':            cause_w,
    }

# ──── Color Helpers ──────────────────────────────────────────────────────────
RISK_COLORS = {
    'Low':      '#2ecc71',
    'Medium':   '#f39c12',
    'High':     '#e74c3c',
    'Critical': '#922b21',
}

def badge(text, color):
    return f'<span class="badge" style="background:{color};color:white;">{text}</span>'

def conf_bar(pct, color):
    return f'''
    <div class="conf-bar-bg">
      <div class="conf-bar" style="width:{pct:.0f}%;background:{color};"></div>
    </div>
    <span style="font-size:11px;color:#888;">{pct:.0f}% confidence</span>
    '''

def metric_card(label, value, sub="", color="#4a90d9"):
    return f'''
    <div class="metric-card" style="--card-color:{color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    '''

# ──── UI Layout ──────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="margin:0;font-size:26px;">🚦 Bengaluru Traffic Congestion Response Recommender</h1>

<hr style="border-color:#2a2e3e;margin:10px 0 20px 0">
""", unsafe_allow_html=True)

col_form, col_result = st.columns([1, 1.5], gap="large")

with col_form:
    st.markdown('<div class="section-header">📋 Incoming Event Details</div>', unsafe_allow_html=True)

    event_type  = st.radio("Event Type", ["unplanned", "planned"], horizontal=True,
                            help="Planned = known in advance (rally, construction). Unplanned = sudden (accident, breakdown).")
    event_cause = st.selectbox("Event Cause", list(B['cause_score_map'].keys()),
                                help="Root cause of the traffic event.")

    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude",  value=12.9716, format="%.5f", step=0.001)
    lon = c2.number_input("Longitude", value=77.5946, format="%.5f", step=0.001)

    t1, t2, t3 = st.columns(3)
    hour        = t1.slider("Hour (24h)", 0, 23, 9)
    day_of_week = t2.selectbox("Day", list(range(7)),
                                format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    month_num   = t3.selectbox("Month", list(range(1, 13)), index=2,
                                format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                        'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])

    zone = st.selectbox("Zone (optional)",
                         ['Unknown'] + sorted(B['zone_station_map'].keys()),
                         help="Leave as Unknown to auto-detect from GPS.")

    # Quick presets
    st.markdown('<div class="section-header">⚡ Quick Presets</div>', unsafe_allow_html=True)
    preset_cols = st.columns(3)
    preset_labels = ["Morning Accident\n(MG Road, 8AM)", "VIP Movement\n(City centre, 10AM)", "Night Protest\n(Silk Board, 11PM)"]
    presets = [
        {'lat':12.9766,'lon':77.6075,'hour':8,'cause':'accident','type':'unplanned','dow':0,'month':3},
        {'lat':12.9716,'lon':77.5946,'hour':10,'cause':'vip_movement','type':'planned','dow':2,'month':6},
        {'lat':12.9174,'lon':77.6228,'hour':23,'cause':'protest','type':'unplanned','dow':5,'month':1},
    ]
    for i, (col, label, preset) in enumerate(zip(preset_cols, preset_labels, presets)):
        if col.button(label, use_container_width=True, key=f"preset_{i}"):
            lat          = preset['lat']
            lon          = preset['lon']
            hour         = preset['hour']
            event_cause  = preset['cause']
            event_type   = preset['type']
            day_of_week  = preset['dow']
            month_num    = preset['month']

    st.write("")
    submit = st.button("🔍 Analyse Event & Get Recommendation", type="primary", use_container_width=True)

with col_result:
    if submit:
        event = {
            'event_type': event_type, 'event_cause': event_cause,
            'latitude': lat, 'longitude': lon,
            'hour': hour, 'day_of_week': day_of_week, 'month_num': month_num,
            'zone': zone,
        }
        with st.spinner("Running models…"):
            r = recommend_resources(event)

        # ── Risk Alert Banner ──
        rc = RISK_COLORS[r['risk_level']]
        st.markdown(f'''
        <div class="alert-box" style="background:{rc}22;border-left:4px solid {rc};">
          <b style="color:{rc};font-size:16px;">⚠️ {r["risk_level"]} Risk Event</b>&nbsp;&nbsp;
          Severity {r["estimated_severity_score"]}/11 · Est. {r["predicted_duration_hours"]}h duration
        </div>
        ''', unsafe_allow_html=True)

        # ── Top Metrics Row ──
        m1, m2, m3, m4 = st.columns(4)
        pc = '#e74c3c' if r['predicted_priority'] == 'High' else '#3498db'
        m1.markdown(metric_card("Priority", r['predicted_priority'], color=pc), unsafe_allow_html=True)
        m1.markdown(conf_bar(r['priority_confidence']*100, pc), unsafe_allow_html=True)

        cc = '#e74c3c' if r['predicted_road_closure'] else '#2ecc71'
        m2.markdown(metric_card("Road Closure", "YES" if r['predicted_road_closure'] else "No", color=cc), unsafe_allow_html=True)
        m2.markdown(conf_bar(r['closure_confidence']*100, cc), unsafe_allow_html=True)

        m3.markdown(metric_card("Officers Needed", f"👮 {r['recommended_officers']}", color=rc), unsafe_allow_html=True)
        m3.markdown(f'<span style="font-size:11px;color:#888;">Cause weight: {r["cause_score"]}/5</span>', unsafe_allow_html=True)

        m4.markdown(metric_card("Est. Duration", f"⏱ {r['predicted_duration_hours']}h", color="#9b59b6"), unsafe_allow_html=True)

        # ── Detailed Predictions ──
        st.markdown('<div class="section-header">📊 Model Predictions</div>', unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)
        d1.metric("Priority Prob (High)", f"{r['prob_high']*100:.1f}%",
                   delta=f"{'High' if r['prob_high']>=0.5 else 'Low'} priority")
        d2.metric("Closure Probability", f"{r['closure_probability']*100:.1f}%",
                   delta="closure predicted" if r['predicted_road_closure'] else "no closure")
        d3.metric("Severity Score", f"{r['estimated_severity_score']}/11",
                   delta=r['risk_level'])

        # ── Deployment Info ──
        st.markdown('<div class="section-header">🏢 Deployment Recommendation</div>', unsafe_allow_html=True)
        dep1, dep2 = st.columns(2)

        dep1.markdown(f"""
        **Deploy from:**  
        🏫 `{r['recommended_police_station']}`  
        _{r['station_assignment_method']}_

        **Detected Corridor:**  
        🛣️ `{r['detected_corridor']}`
        """)

        dep2.markdown(f"""
        **Barricading:**  
        {'🚧 **YES — Deploy barricades**' if r['recommend_barricading'] else '✅ No barricades needed'}

        **Nearest Hotspot:**  
        📍 `{r['nearest_hotspot'].replace('_',' ').title()}` ({r['min_hotspot_dist']} km away)
        """)

        # ── Confidence Gauge Table ──
        st.markdown('<div class="section-header">🎯 Confidence Summary</div>', unsafe_allow_html=True)
        conf_df = pd.DataFrame({
            'Model': ['Priority Classifier', 'Closure Classifier'],
            'Prediction': [r['predicted_priority'], 'YES' if r['predicted_road_closure'] else 'No'],
            'Confidence': [f"{r['priority_confidence']*100:.1f}%", f"{r['closure_confidence']*100:.1f}%"],
            'Raw Prob': [f"{r['prob_high']:.3f}", f"{r['closure_probability']:.3f}"],
        })
        st.dataframe(conf_df, hide_index=True, use_container_width=True)

        # ── Map ──
        st.markdown('<div class="section-header">🗺 Location Map</div>', unsafe_allow_html=True)
        map_df = pd.DataFrame({
            'lat': [lat] + [h[0] for h in HOTSPOTS.values()],
            'lon': [lon] + [h[1] for h in HOTSPOTS.values()],
        })
        st.map(map_df, zoom=11, color='#e74c3c')
        st.caption("Red dot = event location · Other dots = known congestion hotspots")

        # ── Action Checklist ──
        st.markdown('<div class="section-header">✅ Recommended Actions</div>', unsafe_allow_html=True)
        actions = []
        if r['risk_level'] in ('High','Critical'):
            actions.append(f"🚨 **Immediate response** — dispatch {r['recommended_officers']} officers from {r['recommended_police_station']}")
        else:
            actions.append(f"📞 Alert {r['recommended_police_station']} — send {r['recommended_officers']} officers")
        if r['recommend_barricading']:
            actions.append("🚧 Deploy barricades along detected corridor")
        if r['predicted_priority'] == 'High':
            actions.append("📡 Notify traffic control room for signal coordination")
        if r['min_hotspot_dist'] < 2.0:
            actions.append(f"⚠️ Event is within 2 km of {r['nearest_hotspot'].replace('_',' ').title()} — expect spillover congestion")
        if event_type == 'planned':
            actions.append("📋 Pre-position resources 30 mins before event start")
        for a in actions:
            st.markdown(f"- {a}")

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#666;">
          <div style="font-size:48px;margin-bottom:16px;">🚦</div>
          <h3 style="color:#888;">Ready to Analyse</h3>
          <p>Fill in the event details on the left and click<br>
          <b style="color:#4a90d9;">Analyse Event & Get Recommendation</b></p>
        </div>
        """, unsafe_allow_html=True)

# ──── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
fc1, fc2, fc3 = st.columns(3)
fc1.caption("🏆 Gridlock Hackathon 2.0 · Theme 2 · Event-Driven Congestion")

fc2.caption("📍 Data: ASTRAM Bengaluru Traffic Events")
