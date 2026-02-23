# Dashboard VALD – upload, gráficos e métricas
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import kaleido
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

st.set_page_config(page_title="VALD / NordBord Trace Viewer", layout="wide", initial_sidebar_state="expanded")

PADRAO_NOME = "**aparelho-teste-nome-sobrenome-export-data.csv**"
EXEMPLO_NOME = "nordbord-isoprone-Bernardo-Germano-export-19_02_2026.csv"

def format_equip(name: str) -> str:
    return (name or "").strip().upper().replace("_", " ")

def parse_filename(filename: str) -> dict:
    if not filename or not filename.lower().endswith(".csv"):
        return {"valid": False, "filename": filename or "(sem nome)"}
    base = filename[:-4].strip()
    parts = [p.strip() for p in base.split("-") if p.strip()]
    if len(parts) < 5:
        return {"valid": False, "filename": filename}
    if parts[-2].lower() != "export":
        return {"valid": False, "filename": filename}
    aparelho = parts[0]
    teste = parts[1]
    nome_sobrenome = " ".join(parts[2:-2])
    data_str = parts[-1].replace("_", "/")
    return {
        "valid": True,
        "filename": filename,
        "aparelho": aparelho,
        "teste": teste,
        "atleta": nome_sobrenome,
        "data": data_str,
        "aparelho_display": format_equip(aparelho),
        "teste_display": format_equip(teste),
    }

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0f1419 0%, #1a2332 50%, #0f1419 100%); }
    .main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
    h1, h2, h3 { font-family: 'Segoe UI', system-ui, sans-serif; color: #e8eaed !important; }
    p, span, label { color: #b8bcc4 !important; }
    .info-card {
        background: linear-gradient(145deg, #1c2738 0%, #232f3f 100%);
        border: 1px solid rgba(74, 158, 255, 0.25);
        border-radius: 16px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .info-card > div { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1.25rem; }
    .info-card .item {
        background: rgba(30, 42, 58, 0.6);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        border-left: 3px solid #7aa2f7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .info-card .item h3 { margin: 0 0 0.35rem 0 !important; font-size: 0.75rem !important; color: #8ab4f8 !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .info-card .valor { font-size: 1.2rem; font-weight: 700; color: #e8eaed; }
    .info-card.invalid { border-color: rgba(242, 139, 130, 0.5); background: linear-gradient(145deg, #2a1e1e 0%, #3d2828 100%); }
    .info-card.invalid h3 { color: #f28b82 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, #1c2738 0%, #232f3f 100%) !important;
        border: 1px solid rgba(74, 158, 255, 0.2) !important;
        border-radius: 14px !important;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    }
    [data-testid="stMetric"] label { color: #8ab4f8 !important; }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e8eaed !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a2332 0%, #0f1419 100%); }
    .metrics-card {
        background: linear-gradient(160deg, #1c2738 0%, #1e2a3a 100%);
        border: 1px solid rgba(74, 158, 255, 0.22);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-top: 0.75rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.03);
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem 1rem;
    }
    .metrics-card.metrics-card-2 { grid-template-columns: repeat(2, 1fr); }
    .metrics-card .m-item {
        background: rgba(30, 42, 58, 0.7);
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        border-left: 3px solid #7aa2f7;
        text-align: center;
    }
    .metrics-card .m-item .m-label { font-size: 0.7rem; color: #8ab4f8; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2rem; }
    .metrics-card .m-item .m-value { font-size: 1.05rem; font-weight: 700; color: #e8eaed; }
</style>
""", unsafe_allow_html=True)

def to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")

def suggest_windows(df: pd.DataFrame, time_col: str, l_col: str, r_col: str):
    t = df[time_col].to_numpy()
    L = df[l_col].to_numpy()
    R = df[r_col].to_numpy()
    F = np.maximum(L, R)
    dt = np.median(np.diff(t))
    win = max(5, int(round(0.1 / dt)))
    F_s = pd.Series(F).rolling(win, center=True, min_periods=1).mean().to_numpy()
    base_mask = t <= (t.min() + 0.5)
    baseline = float(np.nanmean(F_s[base_mask]))
    std = float(np.nanstd(F_s[base_mask]))
    thr = baseline + 5.0 * std
    mask = F_s > thr
    segments = []
    in_seg = False
    start = None
    for ti, mi in zip(t, mask):
        if mi and not in_seg:
            in_seg = True
            start = float(ti)
        elif in_seg and not mi:
            segments.append((start, float(ti)))
            in_seg = False
    if in_seg:
        segments.append((start, float(t[-1])))
    segments = [(a, b) for (a, b) in segments if (b - a) >= 0.3]
    if not segments:
        return (0.0, 1.0), (0.0, 4.0)
    short_t0 = max(t.min(), segments[0][0])
    long_t0 = max(t.min(), segments[-1][0])
    return (float(short_t0), float(short_t0 + 1.0)), (float(long_t0), float(long_t0 + 4.0))

def make_trace_figure(df, time_col, l_col, r_col, t0, t1, title, height=380):
    t = df[time_col]
    L = df[l_col]
    R = df[r_col]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=L, mode="lines", name="Esquerda", line=dict(color="#7aa2f7", width=2)))
    fig.add_trace(go.Scatter(x=t, y=R, mode="lines", name="Direita", line=dict(color="#e0af68", width=2)))
    fig.add_vrect(x0=t0, x1=t1, fillcolor="rgba(120,120,120,0.15)", line_width=0)
    fig.update_layout(
        title=title, xaxis_title="T (s)", yaxis_title="Força", legend_title="Membro",
        margin=dict(l=20, r=20, t=50, b=20), height=height,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,42,58,0.6)",
        font=dict(color="#e8eaed", size=12),
        xaxis=dict(gridcolor="rgba(45,61,79,0.8)"), yaxis=dict(gridcolor="rgba(45,61,79,0.8)"),
    )
    return fig

def make_trace_figure_single(df, time_col, force_col, t0, t1, title, cor):
    t = df[time_col]
    F = df[force_col]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=F, mode="lines", name="Força", line=dict(color=cor, width=2)))
    fig.add_vrect(x0=t0, x1=t1, fillcolor="rgba(120,120,120,0.15)", line_width=0)
    fig.update_layout(
        title=title, xaxis_title="T (s)", yaxis_title="Força", height=380,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,42,58,0.6)",
        font=dict(color="#e8eaed", size=12), showlegend=False,
        xaxis=dict(gridcolor="rgba(45,61,79,0.8)"), yaxis=dict(gridcolor="rgba(45,61,79,0.8)"),
    )
    return fig

def make_figure_cropped_bilateral(df, time_col, l_col, r_col, t0, t1, title, height=320):
    fig = make_trace_figure(df, time_col, l_col, r_col, t0, t1, title)
    fig.update_layout(xaxis_range=[t0, t1], height=height, margin=dict(t=36, b=28, l=40, r=20))
    return fig

def make_figure_cropped_single(df, time_col, force_col, t0, t1, title, cor, height=320):
    fig = make_trace_figure_single(df, time_col, force_col, t0, t1, title, cor)
    fig.update_layout(xaxis_range=[t0, t1], height=height, margin=dict(t=36, b=28, l=40, r=20))
    return fig

def filter_window(df, time_col, t0, t1):
    return df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()

def window_metrics(dfw, l_col, r_col):
    L = dfw[l_col].to_numpy()
    R = dfw[r_col].to_numpy()
    L_peak = float(np.nanmax(L)) if len(L) else np.nan
    R_peak = float(np.nanmax(R)) if len(R) else np.nan
    denom_peak = max(L_peak, R_peak, 1e-9)
    asym_peak = 100.0 * (R_peak - L_peak) / denom_peak
    L_mean = float(np.nanmean(L)) if len(L) else np.nan
    R_mean = float(np.nanmean(R)) if len(R) else np.nan
    denom_mean = max(L_mean, R_mean, 1e-9)
    asym_mean = 100.0 * (R_mean - L_mean) / denom_mean
    return {"L_peak": L_peak, "R_peak": R_peak, "asym_peak": asym_peak, "L_mean": L_mean, "R_mean": R_mean, "asym_mean": asym_mean}

def window_metrics_single(dfw, force_col):
    F = dfw[force_col].to_numpy()
    return {"peak": float(np.nanmax(F)) if len(F) else np.nan, "mean": float(np.nanmean(F)) if len(F) else np.nan}

def build_pdf(parsed, pages, nome_arquivo):
    if not HAS_REPORTLAB:
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    margin = 0.7 * cm
    content_w = w - 2 * margin
    n = len(pages)
    ncols = 2
    cell_w = content_w / ncols
    header_h = 1.35 * cm
    y_top = h - margin - header_h
    content_h = y_top - margin
    if n == 2:
        meta_box_h = 1.5 * cm
        gap = 0.35 * cm
        title_h = 0.5 * cm
        img_h_per = content_h - title_h - gap - meta_box_h - gap
        fig_export_height = 520
        scale_img = 2.5
        block_height = content_h
    else:
        img_h_per = (content_h - 0.4 * cm) / 2 - 1.5 * cm
        fig_export_height = 320
        scale_img = 1.8
        meta_box_h = 1.2 * cm
        block_height = img_h_per + 0.45 * cm + meta_box_h + 0.35 * cm
    ap_display = parsed.get("aparelho_display") or format_equip(parsed.get("aparelho", ""))
    te_display = parsed.get("teste_display") or format_equip(parsed.get("teste", ""))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, h - margin, "Dashboard VALD – Relatório de Teste")
    c.setFont("Helvetica", 10)
    y_sub = h - margin - 0.6 * cm
    if parsed.get("valid"):
        c.setFillColorRGB(0.35, 0.45, 0.6)
        c.drawString(margin, y_sub, f"Aparelho: {ap_display}   •   Teste: {te_display}   •   Atleta: {parsed.get('atleta', '—')}   •   Data: {parsed.get('data', '—')}")
    else:
        c.drawString(margin, y_sub, f"Arquivo: {parsed.get('filename', nome_arquivo)}")
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0.45, 0.55, 0.72)
    c.setLineWidth(0.5)
    c.line(margin, h - margin - 1.0 * cm, w - margin, h - margin - 1.0 * cm)
    for idx, (titulo_pagina, fig, metrics, bilateral) in enumerate(pages):
        col = idx % ncols
        row = idx // ncols
        x0 = margin + col * cell_w
        y0 = y_top - row * block_height
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x0, y0, titulo_pagina[:48] + ("..." if len(titulo_pagina) > 48 else ""))
        y0 -= 0.45 * cm
        img_buf = io.BytesIO()
        try:
            fig.update_layout(height=fig_export_height, margin=dict(t=32, b=28, l=44, r=16))
            fig.write_image(img_buf, format="png", scale=scale_img)
            img_buf.seek(0)
            img = ImageReader(img_buf)
            iw, ih = img.getSize()
            scale = min((cell_w - 0.2 * cm) / iw, img_h_per / ih)
            dw, dh = iw * scale, ih * scale
            c.drawImage(img, x0, y0 - dh, width=dw, height=dh)
            y0 -= dh + 0.3 * cm
        except Exception:
            c.setFont("Helvetica", 8)
            c.drawString(x0, y0, "(Erro ao gerar gráfico)")
            y0 -= 0.5 * cm
        box_y = y0 - meta_box_h
        box_w = cell_w - 0.2 * cm
        c.setFillColorRGB(0.94, 0.95, 0.97)
        c.setStrokeColorRGB(0.6, 0.7, 0.85)
        c.setLineWidth(0.35)
        c.roundRect(x0, box_y, box_w, meta_box_h, 4)
        c.setFillColorRGB(0.15, 0.2, 0.3)
        c.setStrokeColorRGB(0, 0, 0)
        pad = 0.2 * cm
        if bilateral:
            for ox, label, val in [(pad, "Pico Esq.", f"{metrics.get('L_peak', 0):.1f}"), (box_w * 0.33, "Pico Dir.", f"{metrics.get('R_peak', 0):.1f}"), (box_w * 0.66, "Assim.(pico)", f"{metrics.get('asym_peak', 0):.1f}%")]:
                c.setFont("Helvetica", 9)
                c.drawString(x0 + ox, box_y + meta_box_h - 0.42 * cm, label)
                c.setFont("Helvetica-Bold", 10)
                c.drawString(x0 + ox, box_y + meta_box_h - 0.72 * cm, val)
            for ox, label, val in [(pad, "Média Esq.", f"{metrics.get('L_mean', 0):.1f}"), (box_w * 0.33, "Média Dir.", f"{metrics.get('R_mean', 0):.1f}"), (box_w * 0.66, "Assim.(média)", f"{metrics.get('asym_mean', 0):.1f}%")]:
                c.setFont("Helvetica", 9)
                c.drawString(x0 + ox, box_y + 0.48 * cm, label)
                c.setFont("Helvetica-Bold", 10)
                c.drawString(x0 + ox, box_y + 0.18 * cm, val)
        else:
            c.setFont("Helvetica", 9)
            c.drawString(x0 + pad, box_y + meta_box_h - 0.55 * cm, "Pico")
            c.drawString(x0 + pad, box_y + 0.18 * cm, "Média")
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x0 + box_w * 0.5, box_y + meta_box_h - 0.55 * cm, f"{metrics.get('peak', 0):.1f}")
            c.drawString(x0 + box_w * 0.5, box_y + 0.18 * cm, f"{metrics.get('mean', 0):.1f}")
    c.save()
    buf.seek(0)
    return buf.read()

st.markdown("# 🏋️ Dashboard VALD – Força Esquerda vs Direita")
st.markdown("Visualize os testes de contração (curta e longa) e métricas de assimetria.")
st.markdown("---")

uploaded = st.file_uploader("Envie o CSV do teste", type=["csv"], help=f"Padrão do nome: {EXEMPLO_NOME}")

if uploaded is None:
    st.info("👆 Envie um arquivo CSV para começar.")
    st.markdown(f"**Padrão esperado do nome do arquivo:** `{EXEMPLO_NOME}`")
    st.caption("Formato: aparelho-teste-nome-sobrenome-export-data.csv")
    st.stop()

nome_arquivo = uploaded.name if hasattr(uploaded, "name") else "arquivo.csv"
parsed = parse_filename(nome_arquivo)

if parsed["valid"]:
    ap_display = parsed.get("aparelho_display", format_equip(parsed["aparelho"]))
    te_display = parsed.get("teste_display", format_equip(parsed["teste"]))
    st.markdown(f"""
    <div class="info-card">
        <div>
            <div class="item"><h3>Aparelho</h3><span class="valor">{ap_display}</span></div>
            <div class="item"><h3>Teste</h3><span class="valor">{te_display}</span></div>
            <div class="item"><h3>Atleta</h3><span class="valor">{parsed["atleta"]}</span></div>
            <div class="item"><h3>Data</h3><span class="valor">{parsed["data"]}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-card invalid">
        <h3>⚠️ Nome do arquivo fora do padrão</h3>
        <p style="margin: 0.5rem 0 0 0;">O arquivo <code>{parsed["filename"]}</code> não segue o padrão esperado. Os gráficos e métricas continuam disponíveis; para identificar automaticamente <strong>Aparelho</strong>, <strong>Teste</strong> e <strong>Atleta</strong>, renomeie o arquivo assim:</p>
        <p style="margin: 0.75rem 0 0 0;"><code>{EXEMPLO_NOME}</code></p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Formato: <code>aparelho-teste-nome-sobrenome-export-data.csv</code></p>
    </div>
    """, unsafe_allow_html=True)

df = pd.read_csv(uploaded)
time_col = "Seconds" if "Seconds" in df.columns else df.columns[0]
l_col = "Left Force" if "Left Force" in df.columns else df.columns[1]
r_col = "Right Force" if "Right Force" in df.columns else df.columns[2]
df[time_col] = to_numeric(df[time_col])
df[l_col] = to_numeric(df[l_col])
df[r_col] = to_numeric(df[r_col])
df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
t_min = float(df[time_col].min())
t_max = float(df[time_col].max())
(short_s, short_e), (long_s, long_e) = suggest_windows(df, time_col, l_col, r_col)
short_s, short_e = float(np.clip(short_s, t_min, t_max)), float(np.clip(short_e, t_min, t_max))
long_s, long_e = float(np.clip(long_s, t_min, t_max)), float(np.clip(long_e, t_min, t_max))

st.sidebar.caption("Nome do arquivo: **aparelho-teste-nome-sobrenome-export-data.csv**")
st.markdown("### 📊 Contrações e métricas")
unilateral = st.expander("🦵 Teste unilateral (visualizar cada perna separadamente)", expanded=False)
with unilateral:
    modo_unilateral = st.checkbox("Ativar modo unilateral — perna direita em cima, perna esquerda embaixo", value=False, key="modo_unilateral")

if not modo_unilateral:
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**⏱️ Janela contração curta**")
        rng_short = st.slider("Início e fim curta [s]", t_min, t_max, (float(short_s), float(short_e)), step=0.01, key="b_short")
        t0_short, t1_short = min(rng_short[0], rng_short[1]), max(rng_short[0], rng_short[1])
        st.caption(f"Janela: {t0_short:.2f}s → {t1_short:.2f}s (Δ = {t1_short - t0_short:.2f}s)")
        fig1 = make_trace_figure(df, time_col, l_col, r_col, t0_short, t1_short, "Contração curta", height=500)
        st.plotly_chart(fig1, use_container_width=True)
        dfw1 = filter_window(df, time_col, t0_short, t1_short)
        m1 = window_metrics(dfw1, l_col, r_col)
        st.markdown(f"""<div class="metrics-card">
            <div class="m-item"><div class="m-label">Pico Esq.</div><div class="m-value">{m1['L_peak']:.2f}</div></div>
            <div class="m-item"><div class="m-label">Pico Dir.</div><div class="m-value">{m1['R_peak']:.2f}</div></div>
            <div class="m-item"><div class="m-label">Assim. (pico)</div><div class="m-value">{m1['asym_peak']:.1f}%</div></div>
            <div class="m-item"><div class="m-label">Média Esq.</div><div class="m-value">{m1['L_mean']:.2f}</div></div>
            <div class="m-item"><div class="m-label">Média Dir.</div><div class="m-value">{m1['R_mean']:.2f}</div></div>
            <div class="m-item"><div class="m-label">Assim. (média)</div><div class="m-value">{m1['asym_mean']:.1f}%</div></div>
        </div>""", unsafe_allow_html=True)
    with colB:
        st.markdown("**⏱️ Janela contração longa**")
        rng_long = st.slider("Início e fim longa [s]", t_min, t_max, (float(long_s), float(long_e)), step=0.01, key="b_long")
        t0_long, t1_long = min(rng_long[0], rng_long[1]), max(rng_long[0], rng_long[1])
        st.caption(f"Janela: {t0_long:.2f}s → {t1_long:.2f}s (Δ = {t1_long - t0_long:.2f}s)")
        fig2 = make_trace_figure(df, time_col, l_col, r_col, t0_long, t1_long, "Contração longa", height=500)
        st.plotly_chart(fig2, use_container_width=True)
        dfw2 = filter_window(df, time_col, t0_long, t1_long)
        m2 = window_metrics(dfw2, l_col, r_col)
        st.markdown(f"""<div class="metrics-card">
            <div class="m-item"><div class="m-label">Pico Esq.</div><div class="m-value">{m2['L_peak']:.2f}</div></div>
            <div class="m-item"><div class="m-label">Pico Dir.</div><div class="m-value">{m2['R_peak']:.2f}</div></div>
            <div class="m-item"><div class="m-label">Assim. (pico)</div><div class="m-value">{m2['asym_peak']:.1f}%</div></div>
            <div class="m-item"><div class="m-label">Média Esq.</div><div class="m-value">{m2['L_mean']:.2f}</div></div>
            <div class="m-item"><div class="m-label">Média Dir.</div><div class="m-value">{m2['R_mean']:.2f}</div></div>
            <div class="m-item"><div class="m-label">Assim. (média)</div><div class="m-value">{m2['asym_mean']:.1f}%</div></div>
        </div>""", unsafe_allow_html=True)
else:
    COR_DIREITA = "#e0af68"
    COR_ESQUERDA = "#7aa2f7"
    st.markdown("Perna **direita** (acima) e perna **esquerda** (abaixo). Cada janela: início e fim.")
    st.markdown("#### 🦵 Direita")
    colR1, colR2 = st.columns(2)
    with colR1:
        rng_r_short = st.slider("Início e fim curta Dir. [s]", t_min, t_max, (float(short_s), float(short_e)), step=0.01, key="u_short_r")
        t0_short_r, t1_short_r = min(rng_r_short[0], rng_r_short[1]), max(rng_r_short[0], rng_r_short[1])
        st.caption(f"{t0_short_r:.2f}s → {t1_short_r:.2f}s")
        fig_r_short = make_trace_figure_single(df, time_col, r_col, t0_short_r, t1_short_r, "Direita — Contração curta", COR_DIREITA)
        st.plotly_chart(fig_r_short, use_container_width=True)
        mr_short = window_metrics_single(filter_window(df, time_col, t0_short_r, t1_short_r), r_col)
        st.markdown(f"""<div class="metrics-card metrics-card-2"><div class="m-item"><div class="m-label">Pico Dir.</div><div class="m-value">{mr_short['peak']:.2f}</div></div><div class="m-item"><div class="m-label">Média Dir.</div><div class="m-value">{mr_short['mean']:.2f}</div></div></div>""", unsafe_allow_html=True)
    with colR2:
        rng_r_long = st.slider("Início e fim longa Dir. [s]", t_min, t_max, (float(long_s), float(long_e)), step=0.01, key="u_long_r")
        t0_long_r, t1_long_r = min(rng_r_long[0], rng_r_long[1]), max(rng_r_long[0], rng_r_long[1])
        st.caption(f"{t0_long_r:.2f}s → {t1_long_r:.2f}s")
        fig_r_long = make_trace_figure_single(df, time_col, r_col, t0_long_r, t1_long_r, "Direita — Contração longa", COR_DIREITA)
        st.plotly_chart(fig_r_long, use_container_width=True)
        mr_long = window_metrics_single(filter_window(df, time_col, t0_long_r, t1_long_r), r_col)
        st.markdown(f"""<div class="metrics-card metrics-card-2"><div class="m-item"><div class="m-label">Pico Dir.</div><div class="m-value">{mr_long['peak']:.2f}</div></div><div class="m-item"><div class="m-label">Média Dir.</div><div class="m-value">{mr_long['mean']:.2f}</div></div></div>""", unsafe_allow_html=True)
    st.markdown("#### 🦵 Esquerda")
    colL1, colL2 = st.columns(2)
    with colL1:
        rng_l_short = st.slider("Início e fim curta Esq. [s]", t_min, t_max, (float(short_s), float(short_e)), step=0.01, key="u_short_l")
        t0_short_l, t1_short_l = min(rng_l_short[0], rng_l_short[1]), max(rng_l_short[0], rng_l_short[1])
        fig_l_short = make_trace_figure_single(df, time_col, l_col, t0_short_l, t1_short_l, "Esquerda — Contração curta", COR_ESQUERDA)
        st.plotly_chart(fig_l_short, use_container_width=True)
        ml_short = window_metrics_single(filter_window(df, time_col, t0_short_l, t1_short_l), l_col)
        st.markdown(f"""<div class="metrics-card metrics-card-2"><div class="m-item"><div class="m-label">Pico Esq.</div><div class="m-value">{ml_short['peak']:.2f}</div></div><div class="m-item"><div class="m-label">Média Esq.</div><div class="m-value">{ml_short['mean']:.2f}</div></div></div>""", unsafe_allow_html=True)
    with colL2:
        rng_l_long = st.slider("Início e fim longa Esq. [s]", t_min, t_max, (float(long_s), float(long_e)), step=0.01, key="u_long_l")
        t0_long_l, t1_long_l = min(rng_l_long[0], rng_l_long[1]), max(rng_l_long[0], rng_l_long[1])
        fig_l_long = make_trace_figure_single(df, time_col, l_col, t0_long_l, t1_long_l, "Esquerda — Contração longa", COR_ESQUERDA)
        st.plotly_chart(fig_l_long, use_container_width=True)
        ml_long = window_metrics_single(filter_window(df, time_col, t0_long_l, t1_long_l), l_col)
        st.markdown(f"""<div class="metrics-card metrics-card-2"><div class="m-item"><div class="m-label">Pico Esq.</div><div class="m-value">{ml_long['peak']:.2f}</div></div><div class="m-item"><div class="m-label">Média Esq.</div><div class="m-value">{ml_long['mean']:.2f}</div></div></div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📄 Exportar relatório em PDF")
if not HAS_REPORTLAB or not HAS_KALEIDO:
    st.warning("Para exportar em PDF, instale: `pip install reportlab kaleido`")
else:
    if not modo_unilateral:
        fig1_crop = make_figure_cropped_bilateral(df, time_col, l_col, r_col, t0_short, t1_short, "Contração curta")
        fig2_crop = make_figure_cropped_bilateral(df, time_col, l_col, r_col, t0_long, t1_long, "Contração longa")
        pdf_pages = [(f"Contração curta — Janela {t0_short:.2f}s a {t1_short:.2f}s", fig1_crop, m1, True), (f"Contração longa — Janela {t0_long:.2f}s a {t1_long:.2f}s", fig2_crop, m2, True)]
    else:
        fig_r_short_crop = make_figure_cropped_single(df, time_col, r_col, t0_short_r, t1_short_r, "Direita — Curta", COR_DIREITA)
        fig_r_long_crop = make_figure_cropped_single(df, time_col, r_col, t0_long_r, t1_long_r, "Direita — Longa", COR_DIREITA)
        fig_l_short_crop = make_figure_cropped_single(df, time_col, l_col, t0_short_l, t1_short_l, "Esquerda — Curta", COR_ESQUERDA)
        fig_l_long_crop = make_figure_cropped_single(df, time_col, l_col, t0_long_l, t1_long_l, "Esquerda — Longa", COR_ESQUERDA)
        pdf_pages = [(f"Direita — Curta ({t0_short_r:.2f}s a {t1_short_r:.2f}s)", fig_r_short_crop, mr_short, False), (f"Direita — Longa ({t0_long_r:.2f}s a {t1_long_r:.2f}s)", fig_r_long_crop, mr_long, False), (f"Esquerda — Curta ({t0_short_l:.2f}s a {t1_short_l:.2f}s)", fig_l_short_crop, ml_short, False), (f"Esquerda — Longa ({t0_long_l:.2f}s a {t1_long_l:.2f}s)", fig_l_long_crop, ml_long, False)]
    pdf_bytes = build_pdf(parsed, pdf_pages, nome_arquivo)
    if pdf_bytes:
        pdf_filename = f"{nome_arquivo.replace('.csv', '')}_relatorio.pdf" if nome_arquivo.endswith(".csv") else nome_arquivo + "_relatorio.pdf"
        st.download_button("⬇️ Baixar relatório em PDF", data=pdf_bytes, file_name=pdf_filename, mime="application/pdf", use_container_width=False)
    else:
        st.error("Não foi possível gerar o PDF.")

st.markdown("---")
with st.expander("📋 Ver dados (amostra)"):
    st.dataframe(df.head(200))
