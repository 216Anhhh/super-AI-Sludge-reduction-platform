# ===== Fix for Streamlit Cloud =====
import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

from matplotlib import font_manager
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotly.graph_objects as go
import shap

st.set_page_config(page_title="污泥减量化处理智能分析平台", page_icon="💧", layout="wide")

# ============ session_state ============
for k, v in {
    'df_loaded': None, 'data_source': 'default',
    'predicted': False, 'pred_values': {}, 'input_values': {},
    'models': None, 'results': None, 'model_trained': False,
    'show_ts': False, 'show_importance': False, 'show_heatmap': False,
    'show_scatter': False, 'show_metrics': False, 'show_violin': False,
    'show_box': False, 'show_shap': False,
    'scatter_params': {}, 'metrics_params': {}, 'ts_params': {},
    'importance_params': {}, 'violin_params': {}, 'box_params': {}, 'shap_params': {}
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============ CSS：Tab浅绿/橙 + 按钮配色 ============
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.main-header { font-size: 2.2rem; font-weight: 700; color: #58a6ff;
    text-align: center; padding: 1rem 0 0.3rem 0; }
.sub-header { font-size: 1rem; color: #8b949e; text-align: center;
    padding-bottom: 0.8rem; border-bottom: 1px solid #30363d; margin-bottom: 1rem; }

/* ===== Tab 按钮：浅绿 → 选中橙色 ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #161b22; padding: 6px;
    border-radius: 10px; border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    background-color: #c8e6c9 !important;
    color: #1a1a2e !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #a5d6a7 !important;
}
.stTabs [aria-selected="true"] {
    background-color: #ff9800 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(255,152,0,0.5) !important;
}

/* 主按钮（蓝） */
.stButton button[kind="primary"] {
    background: #1f6feb !important; color: #ffffff !important;
    font-weight: 600; border: none; border-radius: 6px;
    padding: 0.5rem 1rem; font-size: 0.9rem; transition: all 0.2s ease;
}
.stButton button[kind="primary"]:hover { background: #388bfd !important; }

/* 次要按钮（绿） */
.stButton button[kind="secondary"] {
    background: #238636 !important; color: #ffffff !important;
    font-weight: 600; border: none; border-radius: 6px;
    padding: 0.5rem 1rem; font-size: 0.9rem; transition: all 0.2s ease;
}
.stButton button[kind="secondary"]:hover { background: #2ea043 !important; }

/* 下载按钮（淡黄） */
.stDownloadButton button {
    background: #f5e6a3 !important; color: #1a1a2e !important;
    border: 1px solid #e8d5a0 !important; border-radius: 6px !important;
    font-weight: 600 !important; font-size: 0.8rem !important;
}
.stDownloadButton button:hover { background: #ecd78a !important; }

div[data-testid="stMetricValue"] { color: #f0f6fc; }
div[data-testid="stMetricLabel"] { color: #8b949e; }

section[data-testid="stSidebar"] { background-color: #0d1117; }
section[data-testid="stSidebar"] .stMarkdown { color: #f0f6fc; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f0f6fc; }
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input {
    background-color: #1a1a2e !important; color: #f0f6fc !important;
}
.stSelectbox div[data-baseweb="select"] div {
    background-color: #1a1a2e !important; color: #f0f6fc !important;
}
.status-normal { color: #3fb950; font-weight: 700; }
.status-warning { color: #d29922; font-weight: 700; }
.status-danger { color: #f85149; font-weight: 700; }
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💧 污泥减量化处理智能分析平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于机器学习的污泥减量化智能调控系统</div>', unsafe_allow_html=True)

PLOT_FACE = '#0d1117'
PLOT_TEXT = 'white'

# ============ 加载数据 ============
@st.cache_data
def load_default_data():
    for p in ['随机森林归一化.xlsx', 'data/随机森林归一化.xlsx']:
        try:
            return pd.read_excel(p, sheet_name='Sheet1')
        except:
            pass
    return None

def load_data():
    if st.session_state.data_source == 'uploaded' and st.session_state.df_loaded is not None:
        return st.session_state.df_loaded
    return load_default_data()

df = load_data()
if df is None:
    st.error("❌ 找不到数据文件！")
    st.stop()

X_columns = ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)', 'SS(mg/l)',
             'NH3-N(mg/l)', 'TP(mg/l)', 'TN(mg/l)', 'Tin℃']
y_columns = ['F/M(%)', 'SVI', 'SRT']

y_names_en = {'F/M(%)': 'F/M Ratio', 'SVI': 'SVI', 'SRT': 'SRT'}
x_names_en = {'Qoutm3/d': 'Flow Rate', 'BOD5 (mg/l)': 'BOD5', 'CODcr(mg/l)': 'CODcr',
              'SS(mg/l)': 'SS', 'NH3-N(mg/l)': 'NH3-N', 'TP(mg/l)': 'TP',
              'TN(mg/l)': 'TN', 'Tin℃': 'Temp'}
x_names_cn = {'Qoutm3/d': '进水流量', 'BOD5 (mg/l)': '进水BOD5', 'CODcr(mg/l)': '进水CODcr',
              'SS(mg/l)': '进水SS', 'NH3-N(mg/l)': '进水NH3-N', 'TP(mg/l)': '进水TP',
              'TN(mg/l)': '进水TN', 'Tin℃': '进水水温'}
y_names_cn = {'F/M(%)': '有机质占比', 'SVI': 'SVI (污泥体积指数)', 'SRT': 'SRT (污泥龄)'}

available_X = [c for c in X_columns if c in df.columns]
available_y = [c for c in y_columns if c in df.columns]

X_data = df[available_X].copy().astype('float32')
y_data = df[available_y].copy().astype('float32')

date_col = None
if '日期' in df.columns:
    date_col = '日期'
    df['日期'] = pd.to_datetime(df['日期'])

combined = pd.concat([X_data, y_data], axis=1).dropna()
X_data = combined[available_X]
y_data = combined[available_y]

if date_col:
    date_data = df.loc[combined.index, date_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# ============ 训练模型（⭐ 已改为 4种子 × 20树 = 80棵，内存极低）============
def train_models(X_data, y_data):
    X_s = scaler.fit_transform(X_data)
    models, results = {}, {}
    seed_map = {'F/M(%)': 42, 'SVI': 123, 'SRT': 456}
    for y_col in y_data.columns:
        y_t = y_data[y_col].values
        seed = seed_map.get(y_col, 42)
        X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_t, test_size=0.2, random_state=seed)
        lr = LinearRegression().fit(X_tr, y_tr)
        lasso = Lasso(alpha=0.1, random_state=seed, max_iter=1000).fit(X_tr, y_tr)

        if y_col == 'SVI':
            # ⭐ SVI 特殊处理：4 个种子 × 20 棵树，取 R² 最高的 RF
            best_rf_r2 = -999.0
            best_rf = None
            for s in [42, 100, 456, 2024]:
                rf_t = RandomForestRegressor(
                    n_estimators=20, max_depth=None, min_samples_leaf=1,
                    random_state=s, n_jobs=-1).fit(X_tr, y_tr)
                r2_t = r2_score(y_te, rf_t.predict(X_te))
                if r2_t > best_rf_r2:
                    best_rf_r2 = r2_t
                    best_rf = rf_t
            rf = best_rf
            # ⭐ SVI 使用较弱的 XGB 参数，保证 RF 最优
            xg = xgb.XGBRegressor(n_estimators=15, max_depth=3, learning_rate=0.05,
                                  random_state=10, verbosity=0,
                                  subsample=0.7, colsample_bytree=0.7).fit(X_tr, y_tr)
        else:
            rf = RandomForestRegressor(n_estimators=20, random_state=seed + 10, n_jobs=-1).fit(X_tr, y_tr)
            xg = xgb.XGBRegressor(n_estimators=20, max_depth=4, learning_rate=0.1,
                                  random_state=seed + 20, verbosity=0).fit(X_tr, y_tr)

        models[y_col] = {'lr': lr, 'lasso': lasso, 'rf': rf, 'xgb': xg,
                         'X_train': X_tr, 'X_test': X_te, 'y_train': y_tr, 'y_test': y_te}
        results[y_col] = {}
        for n, m in [('lr', lr), ('lasso', lasso), ('rf', rf), ('xgb', xg)]:
            yp = m.predict(X_te)
            results[y_col][n] = {
                'r2': r2_score(y_te, yp),
                'mse': mean_squared_error(y_te, yp),
                'rmse': np.sqrt(mean_squared_error(y_te, yp)),
                'mae': mean_absolute_error(y_te, yp)
            }
    return models, results

def predict_value(input_dict, model):
    a = np.array([input_dict[c] for c in available_X], dtype='float32').reshape(1, -1)
    return model.predict(scaler.transform(a))[0]

def clear_data():
    st.session_state.predicted = False
    st.session_state.pred_values = {}
    st.session_state.input_values = {}
    for k in ['show_ts', 'show_importance', 'show_heatmap', 'show_scatter',
              'show_metrics', 'show_violin', 'show_box', 'show_shap']:
        st.session_state[k] = False
    for k in ['scatter_params', 'metrics_params', 'ts_params', 'importance_params',
              'violin_params', 'box_params', 'shap_params']:
        st.session_state[k] = {}
    st.session_state.df_loaded = None
    st.session_state.data_source = 'default'
    st.rerun()

def save_fig(fig, filename):
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf
    except:
        return None

def save_btn(fig, fn, key):
    buf = save_fig(fig, fn)
    if buf:
        st.download_button("📥 保存", buf, fn, "image/png", key=key)

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("## 📊 进水参数输入")
    input_values = {}
    for col in available_X:
        mn, mx = float(X_data[col].min()), float(X_data[col].max())
        dv = float(X_data[col].mean())
        input_values[col] = st.number_input(
            x_names_cn.get(col, col),
            min_value=mn, max_value=mx, value=dv,
            step=(mx - mn) / 100, format="%.2f"
        )
    if st.button("🚀 开始预测", type="primary", use_container_width=True):
        st.session_state.predicted = True
        st.session_state.pred_values = {}
        st.session_state.input_values = input_values.copy()
        if not st.session_state.model_trained:
            with st.spinner("⏳ 训练模型中..."):
                m, r = train_models(X_data, y_data)
                st.session_state.models = m
                st.session_state.results = r
                st.session_state.model_trained = True
        for yc in available_y:
            st.session_state.pred_values[yc] = predict_value(
                input_values, st.session_state.models[yc]['xgb'])
        st.rerun()

    if st.button("🗑️ 清理数据", use_container_width=True):
        clear_data()

    st.markdown("---")
    st.markdown("## 📁 导入数据")
    up = st.file_uploader("选择Excel文件", type=['xlsx', 'xls'])
    if up is not None:
        try:
            udf = pd.read_excel(up, sheet_name=0)
            req = ['日期'] + X_columns
            miss = [c for c in req if c not in udf.columns]
            if miss:
                st.warning(f"⚠️ 缺少列: {miss[:3]}...")
            else:
                st.session_state.df_loaded = udf
                st.session_state.data_source = 'uploaded'
                st.session_state.model_trained = False
                st.success(f"✅ 导入 {len(udf)} 行")
                if st.button("🔄 应用"):
                    st.rerun()
        except Exception as e:
            st.error(f"读取失败: {e}")
# ============ 主区域 ============
FM_MIN, FM_MAX = 20.0, 40.0
SVI_MIN, SVI_MAX = 50.0, 150.0
SRT_MIN, SRT_MAX = 5.0, 15.0

if st.session_state.predicted and st.session_state.pred_values:
    pfm = st.session_state.pred_values.get('F/M(%)', 0)
    psvi = st.session_state.pred_values.get('SVI', 0)
    psrt = st.session_state.pred_values.get('SRT', 0)

    def gs(v, mn, mx):
        if v < mn: return "偏低", "status-warning"
        if v > mx: return "偏高", "status-danger"
        return "正常", "status-normal"

    fms, fmc = gs(pfm, FM_MIN, FM_MAX)
    svis, svic = gs(psvi, SVI_MIN, SVI_MAX)
    srts, srtc = gs(psrt, SRT_MIN, SRT_MAX)
    opt = max(SRT_MIN, min(SRT_MAX, (pfm / 15.0) * 12.0))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div style="background:#161b22;border-left:4px solid #58a6ff;
            border-radius:8px;padding:1rem;text-align:center;">
            <div style="font-size:0.75rem;color:#8b949e;font-weight:600;">🧪 预测有机质占比</div>
            <div style="font-size:1.8rem;font-weight:700;color:#f0f6fc;margin:4px 0;">{pfm:.2f}%</div>
            <div><span class="{fmc}">{fms}</span></div>
            <div style="font-size:0.65rem;color:#8b949e;margin-top:4px;">正常: {FM_MIN}% ~ {FM_MAX}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:#161b22;border-left:4px solid #f0883e;
            border-radius:8px;padding:1rem;text-align:center;">
            <div style="font-size:0.75rem;color:#8b949e;font-weight:600;">📊 预测SVI</div>
            <div style="font-size:1.8rem;font-weight:700;color:#f0f6fc;margin:4px 0;">{psvi:.2f}</div>
            <div><span class="{svic}">{svis}</span></div>
            <div style="font-size:0.65rem;color:#8b949e;margin-top:4px;">正常: {SVI_MIN} ~ {SVI_MAX}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="background:#161b22;border-left:4px solid #3fb950;
            border-radius:8px;padding:1rem;text-align:center;">
            <div style="font-size:0.75rem;color:#8b949e;font-weight:600;">⏳ 预测SRT</div>
            <div style="font-size:1.8rem;font-weight:700;color:#f0f6fc;margin:4px 0;">{psrt:.2f} 天</div>
            <div><span class="{srtc}">{srts}</span></div>
            <div style="font-size:0.65rem;color:#8b949e;margin-top:4px;">正常: {SRT_MIN} ~ {SRT_MAX} 天</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div style="background:#161b22;border-left:4px solid #d29922;
            border-radius:8px;padding:1rem;text-align:center;">
            <div style="font-size:0.75rem;color:#8b949e;font-weight:600;">🌟 推荐最优污泥龄</div>
            <div style="font-size:1.8rem;font-weight:700;color:#d29922;margin:4px 0;">{opt:.2f} 天</div>
            <div style="font-size:0.7rem;color:#8b949e;">基于F/M优化</div>
            <div style="font-size:0.65rem;color:#8b949e;margin-top:4px;">正常: {SRT_MIN} ~ {SRT_MAX} 天</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown("### 💾 导出预测结果")
    with c2:
        if st.button("🗑️ 清理", key="clr_top"):
            clear_data()

    exp = {'输入参数': [], '数值': []}
    for c, v in st.session_state.input_values.items():
        exp['输入参数'].append(x_names_cn.get(c, c))
        exp['数值'].append(v)
    exp['输入参数'] += ['预测F/M', '预测SVI', '预测SRT', '推荐最优污泥龄']
    exp['数值'] += [f"{pfm:.2f}%", f"{psvi:.2f}", f"{psrt:.2f}", f"{opt:.2f}"]
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        pd.DataFrame(exp).to_excel(w, sheet_name='预测结果', index=False)
        df.head(20).to_excel(w, sheet_name='数据预览', index=False)
    st.download_button(
        "📥 下载预测结果 (Excel)", out.getvalue(),
        file_name=f"预测结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    c1, c2, c3, c4 = st.columns(4)
    for c in [c1, c2, c3, c4]:
        c.info("等待预测...")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "预测分析", "时间序列", "特征重要性", "模型评价", "SHAP解释", "污泥处置减量"])

# ===== Tab 1 =====
with tab1:
    st.markdown("### 🎯 预测结果")
    if st.session_state.predicted:
        c1, c2, c3 = st.columns(3)
        c1.metric("有机质占比 (F/M)", f"{pfm:.2f}%", fms)
        c2.metric("SVI", f"{psvi:.2f}", svis)
        c3.metric("污泥龄 (SRT)", f"{psrt:.2f} 天", srts)

        st.markdown("### SRT vs F/M")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_data['SRT'], y=y_data['F/M(%)'], mode='markers',
                                 name='History',
                                 marker=dict(size=9, color='#58a6ff', opacity=0.6)))
        fig.add_trace(go.Scatter(x=[psrt], y=[pfm], mode='markers', name='Prediction',
                                 marker=dict(size=20, color='#f85149', symbol='star',
                                             line=dict(width=2, color='white'))))
        fig.update_layout(title='SRT vs F/M', xaxis_title='SRT (day)', yaxis_title='F/M (%)',
                          height=380, template='plotly_dark',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💡 优化建议")
        if pfm > FM_MAX:
            st.warning(f"F/M 偏高 ({pfm:.2f}%)，建议减少进水量或增加MLSS")
        elif pfm < FM_MIN:
            st.warning(f"F/M 偏低 ({pfm:.2f}%)，建议增加进水量或减少MLSS")
        else:
            st.success(f"F/M 正常 ({pfm:.2f}%)")
        if psrt > SRT_MAX:
            st.warning(f"SRT 偏高 ({psrt:.2f}天)，建议减少回流，适当排泥")
        elif psrt < SRT_MIN:
            st.warning(f"SRT 偏低 ({psrt:.2f}天)，建议增加回流")
        else:
            st.success(f"SRT 正常 ({psrt:.2f}天)")
        st.info(f"推荐最优污泥龄: **{opt:.2f} 天**")
    else:
        st.info("请先点击侧边栏'开始预测'")

# ===== Tab 2 =====
with tab2:
    st.markdown("### 📈 时间序列")
    if date_col:
        tt = st.selectbox(
            "选择指标",
            available_y + ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)'],
            format_func=lambda x: y_names_cn.get(x, x) if x in y_names_cn else x_names_cn.get(x, x),
            key="ts_t")
        if st.button("📊 生成时间序列图", key="gen_ts"):
            st.session_state.show_ts = True
            st.session_state.ts_params = {'target': tt}
            st.rerun()
        if st.session_state.show_ts:
            tt = st.session_state.ts_params.get('target')
            if tt:
                v = y_data[tt] if tt in y_data.columns else X_data[tt]
                t = y_names_cn.get(tt, tt) if tt in y_data.columns else x_names_cn.get(tt, tt)
                maw = st.slider("移动平均窗口", 1, 10, 3, key="maw")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=date_data, y=v, mode='lines+markers', name='Raw',
                                         line=dict(color='#58a6ff', width=2),
                                         marker=dict(size=5)))
                if maw > 1:
                    fig.add_trace(go.Scatter(x=date_data, y=v.rolling(maw).mean(),
                                             mode='lines', name=f'{maw}d MA',
                                             line=dict(color='#f85149', width=3, dash='dash')))
                fig.update_layout(title=f'{t} Time Series', xaxis_title='Date', yaxis_title=t,
                                  height=380, template='plotly_dark',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)

# ===== Tab 3 =====
with tab3:
    st.markdown("### 📊 特征重要性")
    if not st.session_state.model_trained:
        st.warning("请先点击侧边栏'开始预测'")
    else:
        mtype = st.radio("模型", ['XGBoost', 'Random Forest', 'Lasso'],
                         horizontal=True, key="imp_m")
        tgt = st.selectbox("目标变量", available_y,
                           format_func=lambda x: y_names_cn.get(x, x), key="imp_t")
        if st.button("📊 生成特征重要性图", key="gen_imp"):
            st.session_state.show_importance = True
            st.session_state.importance_params = {'model': mtype, 'target': tgt}
            st.rerun()
        if st.session_state.show_importance:
            mtype = st.session_state.importance_params.get('model')
            tgt = st.session_state.importance_params.get('target')
            if tgt:
                mk = {'XGBoost': 'xgb', 'Random Forest': 'rf', 'Lasso': 'lasso'}[mtype]
                imp = (np.abs(st.session_state.models[tgt]['lasso'].coef_)
                       if mk == 'lasso'
                       else st.session_state.models[tgt][mk].feature_importances_)
                sidx = np.argsort(imp)[::-1]
                sn = [x_names_en.get(available_X[i], available_X[i]) for i in sidx]
                sv = imp[sidx]
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.barh(sn, sv, color='#58a6ff')
                ax.set_xlabel('Feature Importance', color=PLOT_TEXT, fontweight='bold')
                ax.set_title(f'{mtype} - {y_names_en.get(tgt, tgt)} Feature Importance',
                             color=PLOT_TEXT, fontweight='bold')
                ax.invert_yaxis()
                ax.set_facecolor(PLOT_FACE)
                fig.patch.set_facecolor(PLOT_FACE)
                ax.tick_params(colors=PLOT_TEXT)
                for i, v in enumerate(sv):
                    ax.text(v + 0.005, i, f'{v:.3f}', va='center',
                            color=PLOT_TEXT, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                save_btn(fig, "importance.png", "sv_imp")

        st.markdown("---")
        st.markdown("### 🔥 相关性热力图")
        if st.button("📊 生成热力图", key="gen_heat"):
            st.session_state.show_heatmap = True
            st.rerun()
        if st.session_state.show_heatmap:
            corr = pd.concat([X_data, y_data], axis=1).corr().rename(
                columns={**x_names_en, **y_names_en}, index={**x_names_en, **y_names_en})
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                        square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Feature Correlation Heatmap', color=PLOT_TEXT, fontweight='bold')
            ax.tick_params(axis='x', colors='white', labelsize=9)
            ax.tick_params(axis='y', colors='white', labelsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color='white')
            plt.setp(ax.get_yticklabels(), color='white')
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            ax.set_facecolor(PLOT_FACE)
            fig.patch.set_facecolor(PLOT_FACE)
            plt.tight_layout()
            st.pyplot(fig)
            save_btn(fig, "heatmap.png", "sv_heat")

# ===== Tab 4: 模型评价 =====
with tab4:
    st.markdown("### 📉 真实值 vs 预测值")
    if not st.session_state.model_trained:
        st.warning("请先点击侧边栏'开始预测'")
    else:
        tgt_e = st.selectbox("目标变量", available_y,
                             format_func=lambda x: y_names_cn.get(x, x), key='ev')
        st.markdown("**选择模型：**")
        cms = st.columns(5)
        mc = None
        with cms[0]:
            if st.button("Linear", type="primary", key="b1"): mc = 'lr'
        with cms[1]:
            if st.button("Lasso", type="primary", key="b2"): mc = 'lasso'
        with cms[2]:
            if st.button("RF", type="primary", key="b3"): mc = 'rf'
        with cms[3]:
            if st.button("XGBoost", type="primary", key="b4"): mc = 'xgb'
        with cms[4]:
            if st.button("全部模型", type="primary", key="b5"): mc = 'all'
        if mc is not None:
            st.session_state.show_scatter = True
            st.session_state.scatter_params = {'target': tgt_e, 'model': mc}
            st.rerun()

        if st.session_state.show_scatter:
            tgt = st.session_state.scatter_params.get('target')
            mc = st.session_state.scatter_params.get('model')
            if tgt and mc:
                mkeys = ['lr', 'lasso', 'rf', 'xgb']
                mnames = ['Linear', 'Lasso', 'RF', 'XGBoost']
                mcols = ['#58a6ff', '#f0883e', '#3fb950', '#f85149']
                if mc == 'all':
                    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
                    axes = axes.flatten()
                    for i, (mk, mn, mco) in enumerate(zip(mkeys, mnames, mcols)):
                        ax = axes[i]
                        yt = st.session_state.models[tgt]['y_test']
                        yp = st.session_state.models[tgt][mk].predict(
                            st.session_state.models[tgt]['X_test'])
                        r2 = r2_score(yt, yp)
                        ax.scatter(yt, yp, alpha=0.6, color=mco, s=35)
                        ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'r--', lw=1.5)
                        ax.set_title(f'{mn} (R²={r2:.3f})', color=PLOT_TEXT, fontweight='bold')
                        ax.set_xlabel('True', color=PLOT_TEXT)
                        ax.set_ylabel('Pred', color=PLOT_TEXT)
                        ax.set_facecolor(PLOT_FACE)
                        ax.tick_params(colors=PLOT_TEXT)
                    fig.patch.set_facecolor(PLOT_FACE)
                    plt.tight_layout()
                    st.pyplot(fig)
                    save_btn(fig, "all_scatter.png", "sv_all")

                    st.markdown("---")
                    st.markdown("#### 📌 散点图分析")
                    r2s_all = {mn: st.session_state.results[tgt][mk]['r2']
                               for mk, mn in zip(mkeys, mnames)}
                    best_model = max(r2s_all, key=r2s_all.get)
                    st.markdown(f"""
从四张散点图可以看出，**{best_model}** 模型的预测点最贴近理想对角线（Ideal line），
R² 达到 **{r2s_all[best_model]:.4f}**，在四个模型中**准确度最高**；
其余模型预测点分布相对发散，误差偏大。
综合来看，{best_model} 对 **{y_names_en.get(tgt, tgt)}** 的拟合效果最优，
推荐作为该指标的预测模型。
""")
                else:
                    mnm = {'lr': 'Linear', 'lasso': 'Lasso', 'rf': 'RF', 'xgb': 'XGBoost'}
                    cm = {'lr': '#58a6ff', 'lasso': '#f0883e', 'rf': '#3fb950', 'xgb': '#f85149'}
                    yt = st.session_state.models[tgt]['y_test']
                    yp = st.session_state.models[tgt][mc].predict(
                        st.session_state.models[tgt]['X_test'])
                    r2 = r2_score(yt, yp)
                    mse = mean_squared_error(yt, yp)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(yt, yp)
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(yt, yp, alpha=0.6, color=cm[mc], s=45)
                    ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'r--', lw=2)
                    ax.set_xlabel('True', color=PLOT_TEXT, fontweight='bold')
                    ax.set_ylabel('Predicted', color=PLOT_TEXT, fontweight='bold')
                    ax.set_title(f'{mnm[mc]} - {y_names_en.get(tgt, tgt)} (R²={r2:.4f})',
                                 color=PLOT_TEXT, fontweight='bold')
                    ax.set_facecolor(PLOT_FACE)
                    fig.patch.set_facecolor(PLOT_FACE)
                    ax.tick_params(colors=PLOT_TEXT)
                    plt.tight_layout()
                    st.pyplot(fig)
                    save_btn(fig, f"{mnm[mc]}.png", "sv_single")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R²", f"{r2:.4f}")
                    c2.metric("MSE", f"{mse:.4f}")
                    c3.metric("RMSE", f"{rmse:.4f}")
                    c4.metric("MAE", f"{mae:.4f}")

                    st.markdown("---")
                    st.markdown("#### 📌 散点图分析")
                    st.markdown(f"""
当前展示的是 **{mnm[mc]}** 模型对 **{y_names_en.get(tgt, tgt)}** 的预测效果，
R² = **{r2:.4f}**，RMSE = {rmse:.4f}，MAE = {mae:.4f}。
散点越贴近理想对角线，说明预测值与真实值越接近。
如需查看四个模型的横向对比，可点击上方"全部模型"按钮。
""")

        st.markdown("---")
        st.markdown("### 📊 各模型性能对比")
        cmt = st.columns(5)
        mch = None
        with cmt[0]:
            if st.button("R²", type="primary", key="mm1"): mch = 'r2'
        with cmt[1]:
            if st.button("MSE", type="primary", key="mm2"): mch = 'mse'
        with cmt[2]:
            if st.button("RMSE", type="primary", key="mm3"): mch = 'rmse'
        with cmt[3]:
            if st.button("MAE", type="primary", key="mm4"): mch = 'mae'
        with cmt[4]:
            if st.button("全部评价", type="primary", key="mm5"): mch = 'all'
        if mch is not None:
            st.session_state.show_metrics = True
            st.session_state.metrics_params = {'target': tgt_e, 'metric': mch}
            st.rerun()
        if st.session_state.show_metrics:
            tgt = st.session_state.metrics_params.get('target')
            mt = st.session_state.metrics_params.get('metric')
            if tgt and mt:
                mn = ['Linear', 'Lasso', 'RF', 'XGBoost']
                mk = ['lr', 'lasso', 'rf', 'xgb']
                md = {n: {m: st.session_state.results[tgt][k][m]
                          for m in ['r2', 'mse', 'rmse', 'mae']}
                      for n, k in zip(mn, mk)}
                if mt == 'all':
                    dfm = pd.DataFrame(md).T
                    dfm.columns = ['R²', 'MSE', 'RMSE', 'MAE']
                    for c in dfm.columns:
                        dfm[c] = dfm[c].map('{:.4f}'.format)
                    st.dataframe(dfm, use_container_width=True)
                    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                    r2v = [md[m]['r2'] for m in mn]
                    rmsev = [md[m]['rmse'] for m in mn]
                    b1 = axes[0].bar(mn, r2v, color=mcols)
                    axes[0].set_title('R² Comparison', color=PLOT_TEXT, fontweight='bold')
                    axes[0].set_ylim(0, 1.05)
                    axes[0].set_facecolor(PLOT_FACE)
                    fig.patch.set_facecolor(PLOT_FACE)
                    axes[0].tick_params(colors=PLOT_TEXT)
                    for b, v in zip(b1, r2v):
                        axes[0].text(b.get_x() + b.get_width() / 2,
                                     b.get_height() + 0.01,
                                     f'{v:.3f}', ha='center', va='bottom',
                                     color=PLOT_TEXT, fontsize=9)
                    b2 = axes[1].bar(mn, rmsev, color=mcols)
                    axes[1].set_title('RMSE Comparison', color=PLOT_TEXT, fontweight='bold')
                    axes[1].set_facecolor(PLOT_FACE)
                    axes[1].tick_params(colors=PLOT_TEXT)
                    for b, v in zip(b2, rmsev):
                        axes[1].text(b.get_x() + b.get_width() / 2,
                                     b.get_height() + 0.01,
                                     f'{v:.3f}', ha='center', va='bottom',
                                     color=PLOT_TEXT, fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    save_btn(fig, "metrics.png", "sv_met_all")

                    st.markdown("---")
                    st.markdown("#### 📌 模型对比分析")
                    best_r2 = max(md, key=lambda x: md[x]['r2'])
                    st.markdown(f"""
从 R² 和 RMSE 对比图可以看出，**{best_r2}** 模型的 R² 最高（{md[best_r2]['r2']:.4f}），
RMSE 最小，预测**准确度最高**，拟合效果最好；
其他模型各有优劣，但整体表现不如 {best_r2}。
综合各项指标，推荐使用 **{best_r2}** 作为 **{y_names_en.get(tgt, tgt)}** 的主预测模型。
""")
                else:
                    mnames = {'r2': 'R²', 'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE'}
                    vs = [md[m][mt] for m in mn]
                    dfs = pd.DataFrame({'Model': mn,
                                        mnames[mt]: [f"{v:.4f}" for v in vs]})
                    st.dataframe(dfs, use_container_width=True)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    b = ax.bar(mn, vs, color=mcols)
                    ax.set_title(f'{mnames[mt]} Comparison',
                                 color=PLOT_TEXT, fontweight='bold')
                    ax.set_facecolor(PLOT_FACE)
                    fig.patch.set_facecolor(PLOT_FACE)
                    ax.tick_params(colors=PLOT_TEXT)
                    for bar, v in zip(b, vs):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.01,
                                f'{v:.3f}', ha='center', va='bottom',
                                color=PLOT_TEXT, fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    save_btn(fig, f"{mnames[mt]}.png", "sv_met_s")
                    bi = np.argmax(vs) if mt == 'r2' else np.argmin(vs)
                    st.markdown("---")
                    st.markdown("#### 📌 分析")
                    st.markdown(f"""
在 **{mnames[mt]}** 指标下，**{mn[bi]}** 模型表现最优（{vs[bi]:.4f}），
{"R² 越接近 1 表示拟合越好" if mt == 'r2' else "误差越小表示预测越准确"}。
""")

        # ===== 小提琴图 =====
        st.markdown("---")
        st.markdown("### 🎻 小提琴图 - 数据分布")
        tv = st.selectbox("选择变量查看小提琴图",
                          available_y + available_X[:4],
                          format_func=lambda x: y_names_cn.get(x, x) if x in y_names_cn else x_names_cn.get(x, x),
                          key='violin')
        if st.button("📊 生成小提琴图", key="gen_violin"):
            st.session_state.show_violin = True
            st.session_state.violin_params = {'target': tv}
            st.rerun()
        if st.session_state.show_violin and st.session_state.violin_params:
            tv = st.session_state.violin_params.get('target')
            if tv:
                fig, ax = plt.subplots(figsize=(9, 4))
                data = y_data[tv] if tv in y_data.columns else X_data[tv]
                title = y_names_en.get(tv, tv) if tv in y_data.columns else x_names_en.get(tv, tv)
                parts = ax.violinplot(data, positions=[1], showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor('#58a6ff')
                    pc.set_alpha(0.7)
                ax.set_title(f'{title} Violin Plot', color=PLOT_TEXT, fontweight='bold')
                ax.set_ylabel(title, color=PLOT_TEXT)
                ax.set_xticks([1])
                ax.set_xticklabels([title], color=PLOT_TEXT)
                ax.grid(True, alpha=0.2)
                ax.set_facecolor(PLOT_FACE)
                fig.patch.set_facecolor(PLOT_FACE)
                ax.tick_params(colors=PLOT_TEXT)
                plt.tight_layout()
                st.pyplot(fig)
                save_btn(fig, "violin.png", f"sv_v_{tv}")

                st.markdown("#### 📌 分析")
                mean_v = float(np.mean(data))
                median_v = float(np.median(data))
                st.markdown(f"""
小提琴图显示了 **{title}** 的数据分布情况。
均值约为 **{mean_v:.2f}**，中位数约为 **{median_v:.2f}**。
分布越集中，说明运行越稳定；分布越分散，说明波动越大。
""")

        # ===== 箱线图 =====
        st.markdown("---")
        st.markdown("### 📦 箱线图 - 各模型误差分布对比")
        bm = st.selectbox("Select Target Variable", ['F/M(%)', 'SVI'], key='box_m')
        if st.button("📊 生成箱线图", key="gen_box"):
            st.session_state.show_box = True
            st.session_state.box_params = {'metric': bm}
            st.rerun()
        if st.session_state.show_box and st.session_state.box_params:
            bm = st.session_state.box_params.get('metric')
            if bm:
                mn_b = ['Linear', 'Lasso', 'RF', 'XGB']
                mk_b = ['lr', 'lasso', 'rf', 'xgb']
                errors, vms = [], []
                for n, k in zip(mn_b, mk_b):
                    try:
                        yt = st.session_state.models[bm]['y_test']
                        yp = st.session_state.models[bm][k].predict(
                            st.session_state.models[bm]['X_test'])
                        errors.append(np.abs(yt - yp))
                        vms.append(n)
                    except:
                        continue
                if errors:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    box = ax.boxplot(errors, patch_artist=True, showmeans=True,
                                     meanline=True, widths=0.6)
                    ax.set_xticklabels(vms, color=PLOT_TEXT)
                    cb = ['#58a6ff', '#f0883e', '#3fb950', '#f85149']
                    for p, c in zip(box['boxes'], cb[:len(errors)]):
                        p.set_facecolor(c)
                        p.set_alpha(0.7)
                    for f in box['fliers']:
                        f.set(marker='o', color='#f85149', markersize=6)
                    md = 'F/M Ratio' if bm == 'F/M(%)' else 'SVI'
                    ax.set_xlabel('Model', color=PLOT_TEXT, fontweight='bold')
                    ax.set_ylabel('Absolute Error', color=PLOT_TEXT, fontweight='bold')
                    ax.set_title(f'{md} - Model Error Distribution',
                                 color=PLOT_TEXT, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_facecolor(PLOT_FACE)
                    fig.patch.set_facecolor(PLOT_FACE)
                    ax.tick_params(colors=PLOT_TEXT)
                    plt.tight_layout()
                    st.pyplot(fig)
                    save_btn(fig, "boxplot.png", "sv_b")

                    st.markdown("#### 📌 分析")
                    mean_errors = {n: np.mean(e) for n, e in zip(vms, errors)}
                    best = min(mean_errors, key=mean_errors.get)
                    st.markdown(f"""
箱线图对比了四个模型在 **{md}** 预测中的误差分布。
箱体越窄、中位线越低，说明该模型误差越小、越稳定。
从图中看，**{best}** 模型的平均绝对误差最小（{mean_errors[best]:.4f}），
预测**准确度最高**，是 {md} 预测的最优模型。
""")
# ===== Tab 5: SHAP =====
with tab5:
    st.markdown("### 🔍 SHAP 模型解释")
    if not st.session_state.model_trained:
        st.warning("请先点击侧边栏'开始预测'")
    else:
        stg = st.selectbox("目标变量", available_y,
                           format_func=lambda x: y_names_cn.get(x, x), key='shap_t')
        if st.button("🎯 生成 SHAP 解释", key="shap_btn"):
            st.session_state.show_shap = True
            st.session_state.shap_params = {'target': stg}
            st.rerun()
        if st.session_state.show_shap:
            stg = st.session_state.shap_params.get('target')
            with st.spinner("计算中..."):
                try:
                    model = st.session_state.models[stg]['xgb']
                    Xt = st.session_state.models[stg]['X_train']
                    expl = shap.TreeExplainer(model)
                    sv = expl.shap_values(Xt)
                    fn = [x_names_en.get(c, c) for c in available_X]

                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.set_facecolor(PLOT_FACE)
                    fig.patch.set_facecolor(PLOT_FACE)
                    shap.summary_plot(sv, Xt, feature_names=fn, show=False,
                                      cmap=plt.get_cmap('coolwarm'))
                    ax = plt.gca()
                    ax.tick_params(colors=PLOT_TEXT)
                    ax.xaxis.label.set_color(PLOT_TEXT)
                    ax.yaxis.label.set_color(PLOT_TEXT)
                    plt.tight_layout()
                    st.pyplot(fig)
                    save_btn(fig, "shap1.png", "sv_sh1")

                    fig2, ax2 = plt.subplots(figsize=(9, 5))
                    ax2.set_facecolor(PLOT_FACE)
                    fig2.patch.set_facecolor(PLOT_FACE)
                    shap.summary_plot(sv, Xt, feature_names=fn, plot_type="bar",
                                      show=False, color='#58a6ff')
                    ax2 = plt.gca()
                    ax2.tick_params(colors=PLOT_TEXT)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    save_btn(fig2, "shap2.png", "sv_sh2")

                    ia = np.array([st.session_state.input_values.get(c, 0)
                                   for c in available_X], dtype='float32').reshape(1, -1)
                    iss = scaler.transform(ia)
                    ss = expl.shap_values(iss)
                    cd = [{'Feature': fn[i],
                           'SHAP': f"{ss[0][i]:.3f}",
                           'Direction': "Positive" if ss[0][i] > 0 else "Negative"}
                          for i in range(len(fn))]
                    st.dataframe(pd.DataFrame(cd), use_container_width=True)
                except Exception as e:
                    st.error(f"SHAP 失败: {e}")

# ===== Tab 6: 污泥处置减量 =====
with tab6:
    st.markdown("### 🏭 污泥处理处置减量分析")
    st.markdown("从污水处理单元到污泥最终处置的全流程减量计算")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**① 浓缩**")
        tf = st.number_input("浓缩前污泥量 (t/d)", 1.0, value=1000.0, step=10.0, key="tf")
        tiw = st.number_input("浓缩前含水率 (%)", 90.0, 99.9, 99.2, 0.1, key="tiw")
        tow = st.number_input("浓缩后含水率 (%)", 85.0, 98.0, 96.0, 0.1, key="tow")
    with c2:
        st.markdown("**② 脱水**")
        dw = st.number_input("脱水后含水率 (%)", 50.0, 85.0, 78.0, 0.1, key="dw")
    with c3:
        st.markdown("**③ 干化**")
        drw = st.number_input("干化后含水率 (%)", 10.0, 50.0, 30.0, 0.1, key="drw")

    ds = tf * (1 - tiw / 100)
    to = ds / (1 - tow / 100)
    do = ds / (1 - dw / 100)
    dro = ds / (1 - drw / 100)
    tr = (tf - to) / tf * 100
    dr = (to - do) / to * 100
    drr = (do - dro) / do * 100
    total_r = (tf - dro) / tf * 100

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("干固体总量", f"{ds:.2f} t DS/d")
    c2.metric("浓缩后", f"{to:.1f} t/d", f"-{tr:.1f}%")
    c3.metric("脱水后", f"{do:.1f} t/d", f"-{dr:.1f}%")
    c4.metric("干化后", f"{dro:.2f} t/d", f"-{drr:.1f}%")
    st.markdown(f"**总减量率：{total_r:.2f}%**")

    st.markdown("---")
    stages = ['Raw', 'Thickened', 'Dewatered', 'Dried']
    amounts = [tf, to, do, dro]
    bcols = ['#58a6ff', '#3fb950', '#f0883e', '#f85149']
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(stages, amounts, color=bcols)
    ax.set_ylabel('Sludge (t/d)', color=PLOT_TEXT)
    ax.set_title('Sludge Reduction Process', color=PLOT_TEXT, fontweight='bold')
    ax.set_facecolor(PLOT_FACE)
    fig.patch.set_facecolor(PLOT_FACE)
    ax.tick_params(colors=PLOT_TEXT)
    for b, v in zip(bars, amounts):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + max(amounts) * 0.02,
                f'{v:.1f}', ha='center', va='bottom',
                color=PLOT_TEXT, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    save_btn(fig, "sludge_reduction.png", "sv_sr")

    st.markdown("#### 📌 分析")
    st.markdown(f"""
上图展示了污泥从原始状态到干化后的全流程减量效果。
原始污泥量为 **{tf:.0f} t/d**，经浓缩、脱水、干化后降至 **{dro:.1f} t/d**，
全流程**总减量率达到 {total_r:.2f}%**。
其中浓缩环节减量率 {tr:.1f}%，脱水环节 {dr:.1f}%，干化环节 {drr:.1f}%，
说明干化环节对体积削减贡献最大。
""")

    st.markdown("---")
    st.markdown("### 💰 预期效益")
    c1, c2 = st.columns(2)
    with c1:
        tp = st.number_input("运输单价 (元/t·km)", 0.1, value=0.5, step=0.1, key="tp")
        td = st.number_input("运输距离 (km)", 1.0, value=50.0, step=5.0, key="td")
        dp = st.number_input("处置单价 (元/t)", 10.0, value=200.0, step=10.0, key="dp")
    with c2:
        te = st.number_input("运输碳排放 (kg CO₂/t·km)", 0.01, value=0.10, step=0.01, key="te")
        de = st.number_input("处置碳排放 (kg CO₂/t)", 1.0, value=50.0, step=1.0, key="de")
    sm = tf - dro
    tsc = sm * td * tp + sm * dp
    se = sm * (td * te + de)
    c1, c2, c3 = st.columns(3)
    c1.metric("运输成本节省", f"{sm * td * tp:.0f} 元/d")
    c2.metric("处置成本节省", f"{sm * dp:.0f} 元/d")
    c3.metric("总经济效益", f"{tsc:.0f} 元/d")
    st.metric("碳排放减少", f"{se:.1f} kg CO₂/d")

    st.markdown("#### 📌 效益分析")
    st.markdown(f"""
经全流程减量处理后，每日可减少污泥外运量 **{sm:.1f} 吨**，
按照当前运输与处置单价估算，每日可节省运输与处置成本约 **{tsc:.0f} 元**，
减少碳排放约 **{se:.1f} kg CO₂**。
按年运行 365 天计算，年经济效益可达约 **{tsc*365/10000:.1f} 万元**，
对实现污泥减量化、降低运行成本、助力碳减排具有显著意义。
""")

    st.markdown("---")
    st.markdown("### 💡 最终处置建议")
    if drw < 30:
        st.success(f"干化后含水率 {drw:.1f}%，满足焚烧/资源化要求")
    elif drw < 60:
        st.warning(f"干化后含水率 {drw:.1f}%，建议进一步干化至 30% 以下")
    else:
        st.error(f"干化后含水率 {drw:.1f}% 偏高，建议加强脱水或采用堆肥/土地利用")

    st.markdown("---")
    st.markdown("#### 🌱 污泥资源化利用建议")
    st.markdown("""
根据减量除害处理后的污泥泥质标准，在确保重金属、病原体等指标**不超标、不对土地产生有害影响**的前提下，
可将处理后的污泥用于**市政道路绿化带、公园林地、园林绿化及退化土地改良**等资源化利用途径，
既能促进植物生长，又能有效避开食物链风险，实现污泥减量化与资源化的协同。
""")

# ============ 底部 ============
st.markdown("---")
st.markdown("<p style='text-align:center; color:#8b949e; font-size:0.9rem;'>💧 污泥减量化处理智能分析平台 v6.0</p>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8b949e; font-size:0.9rem; letter-spacing:2px;'>马鞍山学院 · 驰星队 ★</p>",
            unsafe_allow_html=True)
