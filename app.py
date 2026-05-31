"""
Streamlit Web App — Smartphone Feature Ranking System
Deploy at: share.streamlit.io
Run locally: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smartphone Ranking System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid rgba(0,255,128,0.15); }
    h1, h2, h3 { color: #e6edf3 !important; font-family: 'Courier New', monospace; }
    p, li, label { color: #8b949e !important; }
    .metric-card {
        background: #161b22;
        border: 1px solid rgba(0,255,128,0.2);
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 32px; font-weight: 800; color: #00ff80; font-family: monospace; }
    .metric-label { font-size: 12px; color: #8b949e; margin-top: 4px; letter-spacing: 0.05em; }
    .rank-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        font-family: monospace;
    }
    .stButton > button {
        background: #00ff80 !important;
        color: #000 !important;
        border: none !important;
        font-weight: 700 !important;
        font-family: monospace !important;
        letter-spacing: 0.05em !important;
        width: 100%;
    }
    .stButton > button:hover { background: #00cc66 !important; }
    [data-testid="stDataFrame"] { border: 1px solid rgba(0,255,128,0.15); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── SMARTPHONERANKER CLASS ───────────────────────────────────────────────────
class SmartphoneRanker:
    def __init__(self, data):
        self.data = data.copy()
        self.normalized_data = None
        self.weighted_data = None
        self.topsis_scores = None

    def normalize_data(self, features):
        normalized = self.data.copy()
        for feature in features:
            sum_sq = np.sqrt(np.sum(self.data[feature] ** 2))
            normalized[feature] = self.data[feature] / sum_sq if sum_sq != 0 else 0
        self.normalized_data = normalized
        return normalized

    def apply_weights(self, features, weights):
        if self.normalized_data is None:
            raise ValueError("Data must be normalized first")
        weighted = self.normalized_data.copy()
        for feature in features:
            weighted[feature] = self.normalized_data[feature] * weights.get(feature, 1.0)
        self.weighted_data = weighted
        return weighted

    def calculate_topsis(self, features,
                          beneficial=None,
                          non_beneficial=None):
        if beneficial is None:
            beneficial = ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'RAM']
        if non_beneficial is None:
            non_beneficial = ['PRICE']
        if self.weighted_data is None:
            raise ValueError("Weights must be applied first")

        ideal_best, ideal_worst = {}, {}
        for f in features:
            if f in beneficial:
                ideal_best[f]  = self.weighted_data[f].max()
                ideal_worst[f] = self.weighted_data[f].min()
            else:
                ideal_best[f]  = self.weighted_data[f].min()
                ideal_worst[f] = self.weighted_data[f].max()

        d_best, d_worst = [], []
        for idx in self.weighted_data.index:
            db = sum((self.weighted_data.loc[idx, f] - ideal_best[f])  ** 2 for f in features)
            dw = sum((self.weighted_data.loc[idx, f] - ideal_worst[f]) ** 2 for f in features)
            d_best.append(np.sqrt(db))
            d_worst.append(np.sqrt(dw))

        d_best  = np.array(d_best)
        d_worst = np.array(d_worst)
        scores  = d_worst / (d_best + d_worst)

        result = self.data.copy()
        result['TOPSIS SCORE'] = np.round(scores, 3)
        result['RANK'] = result['TOPSIS SCORE'].rank(ascending=False, method='min').astype(int)
        return result.sort_values('RANK').reset_index(drop=True)


def categorise(price):
    if price <= 20000:   return 'Budget'
    elif price <= 35000: return 'Mid-Range'
    elif price <= 60000: return 'Premium'
    else:                return 'Flagship'


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### Feature Weights")
    st.caption("Must sum to 1.0")

    w_battery   = st.slider("🔋 Battery",   0.05, 0.40, 0.20, 0.05)
    w_camera    = st.slider("📷 Camera",    0.05, 0.40, 0.20, 0.05)
    w_storage   = st.slider("💾 Storage",   0.05, 0.30, 0.10, 0.05)
    w_processor = st.slider("⚡ Processor", 0.05, 0.40, 0.20, 0.05)
    w_ram       = st.slider("🧠 RAM",       0.05, 0.30, 0.15, 0.05)
    w_price     = st.slider("💰 Price",     0.05, 0.30, 0.15, 0.05)

    total_w = round(w_battery + w_camera + w_storage + w_processor + w_ram + w_price, 2)
    if abs(total_w - 1.0) > 0.001:
        st.error(f"⚠️ Weights sum to {total_w:.2f} — must equal 1.00")
        weights_ok = False
    else:
        st.success(f"✅ Weights sum = {total_w:.2f}")
        weights_ok = True

    st.markdown("---")
    st.markdown("### Data Source")
    data_source = st.radio("Choose dataset", ["Built-in (10 phones)", "Upload CSV"])

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your CSV",
            type=["csv"],
            help="Columns: SMARTPHONENAME, BATTERY, CAMERA, STORAGE, PROCESSOR, RAM, PRICE"
        )

    st.markdown("---")
    st.markdown("### Links")
    st.markdown("[📦 GitHub Repo](https://github.com/Shivajain8449/smartphone-ranking-system)")
    st.markdown("[🌸 GSSoC Issues](https://github.com/Shivajain8449/smartphone-ranking-system/issues)")


# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_builtin():
    return pd.DataFrame({
        'SMARTPHONENAME': ['Galaxy X1','Pixel Pro','Moto One','Redmi Note',
                           'Realme GT','iPhone 14','OnePlus 10','Vivo V25',
                           'Oppo F21','Samsung M32'],
        'BATTERY':    [5000, 4500, 4000, 4500, 4200, 3800, 4500, 4600, 5000, 6000],
        'CAMERA':     [64,   48,   32,   50,   64,   48,   50,   64,   48,   64],
        'STORAGE':    [128,  128,  64,   128,  256,  128,  256,  128,  64,   128],
        'PROCESSOR':  [2.4,  2.8,  2.0,  2.3,  3.0,  3.2,  2.9,  2.5,  2.2,  2.4],
        'RAM':        [8,    12,   6,    8,    12,   6,    12,   8,    6,    8],
        'PRICE':      [20000,25000,18000,15000,30000,80000,35000,22000,16000,14000]
    })

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required = ['SMARTPHONENAME','BATTERY','CAMERA','STORAGE','PROCESSOR','RAM','PRICE']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            df = load_builtin()
    except Exception as e:
        st.error(f"Could not read file: {e}")
        df = load_builtin()
else:
    df = load_builtin()

df['CATEGORY'] = df['PRICE'].apply(categorise)


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:32px 0 8px;'>
  <div style='font-family:monospace;font-size:12px;letter-spacing:0.12em;color:#00ff80;margin-bottom:8px;'>
    ▶ ML PROJECT · GSSOC OPEN SOURCE
  </div>
  <h1 style='font-size:36px;font-weight:800;letter-spacing:-0.02em;margin-bottom:8px;'>
    📱 Smartphone Feature Ranking System
  </h1>
  <p style='font-size:16px;max-width:700px;'>
    Data-driven smartphone rankings using Random Forest Classification + TOPSIS algorithm.
    Adjust weights in the sidebar and hit <strong style='color:#00ff80'>Run Ranking</strong> to re-rank.
  </p>
</div>
""", unsafe_allow_html=True)

run = st.button("▶ RUN RANKING", disabled=not weights_ok)
st.markdown("---")

# ── RUN PIPELINE ─────────────────────────────────────────────────────────────
features = ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'RAM', 'PRICE']
weights  = {
    'BATTERY':   w_battery,
    'CAMERA':    w_camera,
    'STORAGE':   w_storage,
    'PROCESSOR': w_processor,
    'RAM':       w_ram,
    'PRICE':     w_price,
}

if run or 'result' not in st.session_state:
    if weights_ok:
        ranker = SmartphoneRanker(df)
        ranker.normalize_data(features)
        ranker.apply_weights(features, weights)
        result = ranker.calculate_topsis(features)
        st.session_state['result'] = result
    else:
        st.warning("Fix the weights above (must sum to 1.00) before running.")
        st.stop()

result = st.session_state.get('result', None)
if result is None:
    st.info("Configure weights in the sidebar and click ▶ RUN RANKING")
    st.stop()

# ── METRIC CARDS ─────────────────────────────────────────────────────────────
best = result.iloc[0]
budget_df = result[result['PRICE'] <= 20000]
best_budget = budget_df.iloc[0] if not budget_df.empty else result.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value'>{len(result)}</div>
      <div class='metric-label'>PHONES RANKED</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value'>{best['TOPSIS SCORE']:.3f}</div>
      <div class='metric-label'>TOP SCORE</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value' style='font-size:20px;'>{best['SMARTPHONENAME']}</div>
      <div class='metric-label'>🏆 BEST OVERALL</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value' style='font-size:20px;'>{best_budget['SMARTPHONENAME']}</div>
      <div class='metric-label'>💰 BEST UNDER ₹20K</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── RANKINGS TABLE ───────────────────────────────────────────────────────────
st.markdown("### 🏆 Final Rankings")

cat_colors = {'Budget':'#79c0ff','Mid-Range':'#00ff80','Premium':'#e3b341','Flagship':'#ff5f57'}

def style_row(row):
    base = ['background-color: #161b22; color: #e6edf3;'] * len(row)
    if row['RANK'] == 1:
        base = ['background-color: #0d2818; color: #e6edf3;'] * len(row)
    return base

display_df = result[['RANK','SMARTPHONENAME','TOPSIS SCORE','CATEGORY','BATTERY','CAMERA','RAM','PRICE']].copy()
display_df['PRICE'] = display_df['PRICE'].apply(lambda x: f"₹{x:,.0f}")

styled = display_df.style\
    .apply(style_row, axis=1)\
    .format({'TOPSIS SCORE': '{:.3f}'})\
    .set_properties(**{'text-align': 'left'})

st.dataframe(styled, use_container_width=True, hide_index=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── VISUALISATIONS ───────────────────────────────────────────────────────────
st.markdown("### 📊 Visualisations")
tab1, tab2, tab3, tab4 = st.tabs(["📊 TOPSIS Scores", "🔥 Feature Heatmap", "💲 Price vs Score", "🕸 Radar Chart"])

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#8b949e',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})

with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#00ff80' if i == 0 else '#1f6feb' for i in range(len(result))]
    bars = ax.barh(result['SMARTPHONENAME'], result['TOPSIS SCORE'], color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel('TOPSIS Score', fontsize=11)
    ax.set_title('Smartphone Rankings by TOPSIS Score', fontsize=13, pad=12)
    ax.axvline(x=0.5, color='#00ff80', linestyle='--', alpha=0.3, linewidth=1)
    for bar, val in zip(bars, result['TOPSIS SCORE']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10, color='#e6edf3')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    feat_cols = ['BATTERY','CAMERA','STORAGE','PROCESSOR','RAM']
    heatmap_data = result.set_index('SMARTPHONENAME')[feat_cols].copy()
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, linewidths=0.5, linecolor='#0d1117',
                cbar_kws={'label': 'Normalised Value'})
    ax.set_title('Feature Heatmap (Normalised)', fontsize=13, pad=12)
    ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_list = result['CATEGORY'].unique()
    for cat in cat_list:
        sub = result[result['CATEGORY'] == cat]
        ax.scatter(sub['PRICE'], sub['TOPSIS SCORE'],
                   label=cat, s=120, alpha=0.85,
                   color=cat_colors.get(cat, '#888'))
        for _, row in sub.iterrows():
            ax.annotate(row['SMARTPHONENAME'],
                        (row['PRICE'], row['TOPSIS SCORE']),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=9, color='#8b949e')
    ax.set_xlabel('Price (₹)', fontsize=11)
    ax.set_ylabel('TOPSIS Score', fontsize=11)
    ax.set_title('Price vs Performance', fontsize=13, pad=12)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    feat_r = ['BATTERY','CAMERA','STORAGE','PROCESSOR','RAM']
    N = len(feat_r)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    norm_r = result[feat_r].copy()
    for f in feat_r:
        mn, mx = norm_r[f].min(), norm_r[f].max()
        norm_r[f] = (norm_r[f] - mn) / (mx - mn) if mx != mn else 0.5

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True),
                           facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    palette = ['#00ff80','#79c0ff','#e3b341','#ff5f57','#d2a8ff',
               '#ffa657','#56d364','#f85149','#58a6ff','#bc8cff']

    top5 = result.head(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        vals = norm_r.loc[row.name].values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2,
                color=palette[i % len(palette)], label=row['SMARTPHONENAME'], alpha=0.85)
        ax.fill(angles, vals, alpha=0.07, color=palette[i % len(palette)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_r, size=11, color='#8b949e')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25','0.50','0.75','1.00'], size=8, color='#4a5568')
    ax.grid(color='#21262d', linestyle='--', alpha=0.6)
    ax.spines['polar'].set_color('#30363d')
    ax.set_title('Top 5 — Feature Comparison (Normalised)', pad=18,
                 fontsize=13, color='#e6edf3')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
              facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── PREDICT NEW PHONE ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔮 Predict a New Smartphone")
st.caption("Enter specs below — the system will predict its category and rank it against existing phones.")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    n_name      = st.text_input("Phone Name",  value="My Phone")
    n_battery   = st.number_input("Battery (mAh)",   500,  10000, 5000, 100)
    n_camera    = st.number_input("Camera (MP)",       5,    200,   50,   1)
with pc2:
    n_storage   = st.number_input("Storage (GB)",      8,   1024,  128,   8)
    n_processor = st.number_input("Processor (GHz)",  1.0,   4.0,  2.5, 0.1)
    n_ram       = st.number_input("RAM (GB)",           2,    24,    8,   1)
with pc3:
    n_price     = st.number_input("Price (₹)",       1000, 200000, 20000, 500)
    st.markdown("<br/>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 ADD & RANK")

if predict_btn:
    new_row = pd.DataFrame([{
        'SMARTPHONENAME': n_name,
        'BATTERY':   n_battery,
        'CAMERA':    n_camera,
        'STORAGE':   n_storage,
        'PROCESSOR': n_processor,
        'RAM':       n_ram,
        'PRICE':     n_price,
        'CATEGORY':  categorise(n_price),
    }])
    extended = pd.concat([df, new_row], ignore_index=True)
    r2 = SmartphoneRanker(extended)
    r2.normalize_data(features)
    r2.apply_weights(features, weights)
    result2 = r2.calculate_topsis(features)

    phone_row = result2[result2['SMARTPHONENAME'] == n_name].iloc[0]
    rank      = phone_row['RANK']
    total     = len(result2)
    score     = phone_row['TOPSIS SCORE']
    cat       = categorise(n_price)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>#{rank} / {total}</div>
          <div class='metric-label'>PREDICTED RANK</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{score:.3f}</div>
          <div class='metric-label'>TOPSIS SCORE</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        c = cat_colors.get(cat,'#888')
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value' style='color:{c};font-size:22px;'>{cat}</div>
          <div class='metric-label'>CATEGORY</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    disp2 = result2[['RANK','SMARTPHONENAME','TOPSIS SCORE','CATEGORY','PRICE']].copy()
    disp2['PRICE'] = disp2['PRICE'].apply(lambda x: f"₹{x:,.0f}")

    def highlight_new(row):
        if row['SMARTPHONENAME'] == n_name:
            return ['background-color:#1a3a1a; color:#00ff80; font-weight:600;'] * len(row)
        return ['background-color:#161b22; color:#e6edf3;'] * len(row)

    st.dataframe(disp2.style.apply(highlight_new, axis=1), use_container_width=True, hide_index=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:20px 0;font-family:monospace;font-size:12px;color:#4a5568;'>
  Built by <a href='https://github.com/Shivajain8449' style='color:#00ff80;text-decoration:none;'>Shiva Jain</a>
  &nbsp;·&nbsp;
  <a href='https://github.com/Shivajain8449/smartphone-ranking-system' style='color:#00ff80;text-decoration:none;'>GitHub</a>
  &nbsp;·&nbsp; MIT License &nbsp;·&nbsp; GSSoC Open Source Project
</div>
""", unsafe_allow_html=True)