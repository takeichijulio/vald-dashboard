# Dashboard ForceFrame – 4 canais de força (Inner L/R + Outer L/R)
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

st.set_page_config(
    page_title="VALD / ForceFrame Trace Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXEMPLO_NOME = "forceframe-isoprone-Bernardo-Germano-export-20_05_2026.csv"

# ---------------------------------------------------------------------------
# Cores por canal
# ---------------------------------------------------------------------------
CORES = {
    "inner_left":  "#7aa2f7",   # azul
    "inner_right": "#e0af68",   # laranja
    "outer_left":  "#9ece6a",   # verde
    "outer_right": "#f7768e",   # vermelho
}

# ---------------------------------------------------------------------------
# CSS (igual ao Dashboard principal)
# ---------------------------------------------------------------------------
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
        box-shadow: 0 4px 14px rgba(0,0,0,0.22);
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
    .channel-badge {
        display: inline-block;
        padding: 0.18rem 0.65rem;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def format_equip(name: str) -> str:
    return (name or "").strip().upper().replace("_", " ")


def parse_filename(filename: str) -> dict:
    if not filename or not filename.lower().endswith(".csv"):
        return {"valid": False, "filename": filename or "(sem nome)"}
    base = filename[:-4].strip()
    parts = [p.strip() for p in base.split("-") if p.strip()]
    if len(parts) < 5 or parts[-2].lower() != "export":
        return {"valid": False, "filename": filename}
    return {
        "valid": True,
        "filename": filename,
        "aparelho": parts[0],
        "teste": parts[1],
        "atleta": " ".join(parts[2:-2]),
        "data": parts[-1].replace("_", "/"),
        "aparelho_display": format_equip(parts[0]),
        "teste_display": format_equip(parts[1]),
    }


def to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def detect_forceframe_columns(df: pd.DataFrame):
    """
    Retorna (time_col, inner_left, inner_right, outer_left, outer_right).
    Tenta por nome primeiro, depois por posição.
    """
    cols = [c.strip().strip('"').strip('﻿') for c in df.columns]
    df.columns = cols

    def _find(candidates):
        for cand in candidates:
            for c in cols:
                if cand.lower() in c.lower():
                    return c
        return None

    time_col   = _find(["second", "time", "tempo"]) or cols[0]
    inner_left  = _find(["inner left", "inner_left"])  or (cols[1] if len(cols) > 1 else None)
    inner_right = _find(["inner right", "inner_right"]) or (cols[2] if len(cols) > 2 else None)
    outer_left  = _find(["outer left", "outer_left"])  or (cols[3] if len(cols) > 3 else None)
    outer_right = _find(["outer right", "outer_right"]) or (cols[4] if len(cols) > 4 else None)

    return time_col, inner_left, inner_right, outer_left, outer_right


def channel_is_active(series: pd.Series, threshold: float = 5.0) -> bool:
    """Canal tem atividade se o pico absoluto superar o threshold."""
    return float(series.abs().max()) > threshold


def suggest_window_for_channel(t: np.ndarray, F: np.ndarray, margin: float = 0.3) -> tuple:
    """Retorna (t0, t1) da maior janela de atividade contínua do canal."""
    if len(F) == 0:
        return (float(t[0]), float(t[-1]))

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0025
    win = max(5, int(round(0.1 / dt)))
    F_abs = np.abs(F)
    F_s = pd.Series(F_abs).rolling(win, center=True, min_periods=1).mean().to_numpy()

    base_mask = t <= (t.min() + 0.5)
    baseline = float(np.nanmean(F_s[base_mask])) if base_mask.any() else 0.0
    std_b    = float(np.nanstd(F_s[base_mask]))  if base_mask.any() else 0.0
    thr = max(baseline + 5.0 * std_b, 5.0)

    mask = F_s > thr
    segments = []
    in_seg, start = False, None
    for ti, mi in zip(t, mask):
        if mi and not in_seg:
            in_seg, start = True, float(ti)
        elif not mi and in_seg:
            segments.append((start, float(ti)))
            in_seg = False
    if in_seg:
        segments.append((start, float(t[-1])))

    # filtra segmentos muito curtos
    segments = [(a, b) for a, b in segments if (b - a) >= 0.2]
    if not segments:
        return (float(t[0]), float(t[-1]))

    # retorna janela do segmento de maior pico
    best = max(segments, key=lambda ab: float(np.nanmax(np.abs(F[(t >= ab[0]) & (t <= ab[1])]))) if ((t >= ab[0]) & (t <= ab[1])).any() else 0)
    t0 = max(float(t[0]), best[0] - margin)
    t1 = min(float(t[-1]), best[1] + margin)
    return (t0, t1)


def window_metrics_single(dfw: pd.DataFrame, col: str) -> dict:
    F = dfw[col].to_numpy()
    if len(F) == 0:
        return {"peak": np.nan, "mean": np.nan}
    return {"peak": float(np.nanmax(F)), "mean": float(np.nanmean(F))}


def asymmetry(v_left: float, v_right: float) -> float:
    denom = max(abs(v_left), abs(v_right), 1e-9)
    return 100.0 * (v_right - v_left) / denom


def filter_window(df: pd.DataFrame, time_col: str, t0: float, t1: float) -> pd.DataFrame:
    return df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(30,42,58,0.6)",
    font=dict(color="#e8eaed", size=12),
    xaxis=dict(gridcolor="rgba(45,61,79,0.8)"),
    yaxis=dict(gridcolor="rgba(45,61,79,0.8)"),
    margin=dict(l=20, r=20, t=50, b=20),
)


def make_overview_figure(df, time_col, channels: dict, height=320) -> go.Figure:
    """Gráfico de visão geral com todos os canais ativos."""
    fig = go.Figure()
    labels = {
        "inner_left":  "Inner Esq.",
        "inner_right": "Inner Dir.",
        "outer_left":  "Outer Esq.",
        "outer_right": "Outer Dir.",
    }
    widths = {"inner_left": 1.5, "inner_right": 1.5, "outer_left": 2.5, "outer_right": 2.5}
    for key, col in channels.items():
        if col is None:
            continue
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[col],
            mode="lines", name=labels.get(key, key),
            line=dict(color=CORES[key], width=widths.get(key, 2)),
        ))
    fig.update_layout(
        title="Visão geral — todos os canais", xaxis_title="T (s)", yaxis_title="Força",
        legend_title="Canal", height=height, **_LAYOUT_BASE,
    )
    return fig


def make_channel_figure(df, time_col, col, t0, t1, title, cor, height=400) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df[col],
        mode="lines", name="Força",
        line=dict(color=cor, width=2),
    ))
    fig.add_vrect(x0=t0, x1=t1, fillcolor="rgba(120,120,120,0.15)", line_width=0)
    fig.update_layout(
        title=title, xaxis_title="T (s)", yaxis_title="Força (N)",
        showlegend=False, height=height, **_LAYOUT_BASE,
    )
    return fig


def make_channel_figure_cropped(df, time_col, col, t0, t1, title, cor, height=300) -> go.Figure:
    fig = make_channel_figure(df, time_col, col, t0, t1, title, cor, height)
    fig.update_layout(
        xaxis_range=[t0, t1],
        margin=dict(t=36, b=28, l=40, r=20),
    )
    return fig


def make_comparison_figure(
    df, time_col,
    col_left, cor_left, label_left,
    col_right, cor_right, label_right,
    t0_left, t1_left, t0_right, t1_right,
    height=380,
) -> go.Figure:
    """Sobreposição dos dois canais (esq e dir) normalizados na janela de cada um, para comparação visual."""
    fig = go.Figure()

    dfw_l = filter_window(df, time_col, t0_left, t1_left)
    dfw_r = filter_window(df, time_col, t0_right, t1_right)

    # Normaliza o tempo de cada janela para 0..1
    def norm_t(series, t0):
        return series - t0

    t_l = norm_t(dfw_l[time_col], t0_left)
    t_r = norm_t(dfw_r[time_col], t0_right)

    fig.add_trace(go.Scatter(x=t_l, y=dfw_l[col_left],  mode="lines", name=label_left,  line=dict(color=cor_left,  width=2)))
    fig.add_trace(go.Scatter(x=t_r, y=dfw_r[col_right], mode="lines", name=label_right, line=dict(color=cor_right, width=2)))
    fig.update_layout(
        title="Comparativo Esq. vs Dir. (tempo relativo)",
        xaxis_title="T relativo (s)", yaxis_title="Força (N)",
        legend_title="Lado", height=height, **_LAYOUT_BASE,
    )
    return fig


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def build_pdf_forceframe(parsed, pdf_pages, nome_arquivo):
    if not HAS_REPORTLAB:
        return None

    def _hex_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    C = {
        "header_bg": "#1a2332", "header_sub": "#8ab4f8", "accent": "#4a7ac4",
        "body_bg": "#ffffff",   "text": "#1c2738",       "muted": "#5a6677",
        "rule": "#c8d0db",      "block_title_bg": "#eef2f7",
        "col_L": "#dce8ff",     "col_R": "#fff3dc",      "col_A": "#f0f2f5",
        "pico_hdr": "#4a7ac4",  "alert": "#c0392b",
    }

    def fill(name): cv.setFillColorRGB(*_hex_rgb(C[name]))
    def stroke(name): cv.setStrokeColorRGB(*_hex_rgb(C[name]))

    buf = io.BytesIO()
    cv = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    mrg = 0.7 * cm
    cw  = w - 2 * mrg

    ap = parsed.get("aparelho_display") or format_equip(parsed.get("aparelho", "ForceFrame"))
    te = parsed.get("teste_display")    or format_equip(parsed.get("teste", ""))
    if parsed.get("valid"):
        sublines = [f"Aparelho: {ap}   •   Teste: {te}   •   Atleta: {parsed.get('atleta','—')}   •   Data: {parsed.get('data','—')}"]
    else:
        sublines = [f"Arquivo: {parsed.get('filename', nome_arquivo)}"]

    hdr_h = 1.05 * cm + len(sublines) * 0.36 * cm + 0.35 * cm
    hdr_top = h - mrg
    body_top = hdr_top - hdr_h
    content_h = body_top - mrg

    # Cabeçalho
    cv.setFillColorRGB(*_hex_rgb(C["header_bg"]))
    cv.rect(mrg, body_top + 2, cw, hdr_h - 2, stroke=0, fill=1)
    cv.setFillColorRGB(*_hex_rgb(C["accent"]))
    cv.rect(mrg, body_top, cw, 2, stroke=0, fill=1)
    cv.setFillColorRGB(1, 1, 1)
    cv.setFont("Helvetica-Bold", 13)
    cv.drawString(mrg + 0.35 * cm, hdr_top - 0.45 * cm, "Dashboard VALD – ForceFrame Relatório")
    cv.setFillColorRGB(*_hex_rgb(C["header_sub"]))
    cv.setFont("Helvetica", 8.5)
    sy = hdr_top - 0.45 * cm - 0.38 * cm
    for line in sublines:
        cv.drawString(mrg + 0.35 * cm, sy, line[:120])
        sy -= 0.36 * cm

    # Fundo branco
    cv.setFillColorRGB(1, 1, 1)
    cv.rect(mrg, mrg, cw, body_top - mrg, stroke=0, fill=1)

    n = len(pdf_pages)
    ncols = 2
    cell_w = cw / ncols
    inner_pad = 0.14 * cm

    if n <= 2:
        block_h = content_h
        scale_img = 2.5
    else:
        block_h = content_h / 2.0
        scale_img = 2.0

    title_h      = 0.48 * cm
    metrics_box_h = 1.6 * cm
    gap          = 0.12 * cm
    img_h        = block_h - title_h - gap - metrics_box_h

    for idx, (titulo, fig, metrics) in enumerate(pdf_pages):
        col = idx % ncols
        row = idx // ncols
        x0  = mrg + col * cell_w + inner_pad
        bcw = cell_w - 2 * inner_pad
        cell_top = body_top - row * block_h

        # Barra de título
        cv.setFillColorRGB(*_hex_rgb(C["block_title_bg"]))
        cv.setStrokeColorRGB(*_hex_rgb(C["rule"]))
        cv.setLineWidth(0.6)
        cv.roundRect(x0 + 0.12 * cm, cell_top - title_h, bcw - 0.24 * cm, title_h, 4, stroke=1, fill=1)
        cv.setFillColorRGB(*_hex_rgb(C["text"]))
        cv.setFont("Helvetica-Bold", 9)
        cv.drawString(x0 + 0.35 * cm, cell_top - title_h + 0.14 * cm, titulo[:52] + ("…" if len(titulo) > 52 else ""))

        img_top = cell_top - title_h
        img_buf = io.BytesIO()
        img_ok  = False
        dw = dh = 0.0
        try:
            fe = go.Figure(fig.to_dict())
            fe.update_layout(
                template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
                font=dict(color="#1c2738", size=11),
                xaxis=dict(gridcolor="#d0d7e3", linecolor="#8a9ab5", tickfont=dict(color="#1c2738"), title_font=dict(color="#1c2738")),
                yaxis=dict(gridcolor="#d0d7e3", linecolor="#8a9ab5", tickfont=dict(color="#1c2738"), title_font=dict(color="#1c2738")),
                legend=dict(font=dict(color="#1c2738"), bgcolor="rgba(255,255,255,0.85)", bordercolor="#c8d0db", borderwidth=1),
                title_font=dict(color="#1c2738"),
                height=max(260, int(img_h * 1.5)),
                margin=dict(t=28, b=24, l=40, r=12),
            )
            for sc in (1, scale_img):
                try:
                    img_buf.seek(0); img_buf.truncate(0)
                    fe.write_image(img_buf, format="png", scale=sc, engine="kaleido")
                    img_buf.seek(0)
                    ir  = ImageReader(img_buf)
                    iw, ih = ir.getSize()
                    slot_w = bcw - 0.15 * cm
                    sc2 = min(slot_w / iw, img_h / ih)
                    dw, dh = iw * sc2, ih * sc2
                    ix = x0 + (bcw - 0.15 * cm - dw) / 2
                    cv.drawImage(ir, ix, img_top - dh, width=dw, height=dh, mask="auto")
                    img_ok = True
                    break
                except Exception:
                    continue
        except Exception:
            pass

        if not img_ok:
            cv.setFont("Helvetica", 8)
            cv.setFillColorRGB(*_hex_rgb(C["muted"]))
            cv.drawString(x0, img_top - img_h * 0.55, "(Gráfico indisponível)")

        box_y = (img_top - dh if img_ok else img_top - img_h) - gap - metrics_box_h
        bw    = bcw
        third = bw / 3.0
        sep_y = box_y + metrics_box_h / 2

        for i, bg_key in enumerate(["col_L", "col_R", "col_A"]):
            cv.setFillColorRGB(*_hex_rgb(C[bg_key]))
            cv.rect(x0 + i * third, box_y, third, metrics_box_h, stroke=0, fill=1)
        cv.setStrokeColorRGB(*_hex_rgb(C["rule"]))
        cv.setLineWidth(0.35)
        for i in (1, 2):
            cv.line(x0 + i * third, box_y, x0 + i * third, box_y + metrics_box_h)
        cv.setLineWidth(0.5)
        cv.line(x0, sep_y, x0 + bw, sep_y)

        fs_lab, fs_val = 7.0, 11

        def _draw_cell(cx, label_top, val_top, lab, val_str):
            cv.setFillColorRGB(*_hex_rgb(C["muted"]))
            cv.setFont("Helvetica", fs_lab)
            cv.drawString(cx + 0.12 * cm, label_top, lab)
            cv.setFillColorRGB(*_hex_rgb(C["text"]))
            cv.setFont("Helvetica-Bold", fs_val)
            cv.drawString(cx + 0.12 * cm, val_top, val_str)

        top_r = box_y + metrics_box_h
        y_lab1 = top_r - 0.38 * cm
        y_val1 = top_r - 0.80 * cm
        y_lab2 = sep_y  - 0.38 * cm
        y_val2 = sep_y  - 0.80 * cm

        peak_l = metrics.get("peak_left",  0.0) or 0.0
        mean_l = metrics.get("mean_left",  0.0) or 0.0
        peak_r = metrics.get("peak_right", 0.0) or 0.0
        mean_r = metrics.get("mean_right", 0.0) or 0.0
        ap_    = metrics.get("asym_peak",  0.0) or 0.0
        am_    = metrics.get("asym_mean",  0.0) or 0.0

        cv.setFillColorRGB(*_hex_rgb(C["pico_hdr"]))
        cv.setFont("Helvetica-Bold", 6.5)
        cv.drawString(x0 + 0.12 * cm, top_r  - 0.10 * cm, "PICO")
        cv.drawString(x0 + 0.12 * cm, sep_y  - 0.10 * cm, "MEDIA")

        _draw_cell(x0,           y_lab1, y_val1, "Pico Esq.",   f"{peak_l:.1f} N")
        _draw_cell(x0,           y_lab2, y_val2, "Media Esq.",  f"{mean_l:.1f} N")
        _draw_cell(x0 + third,   y_lab1, y_val1, "Pico Dir.",   f"{peak_r:.1f} N")
        _draw_cell(x0 + third,   y_lab2, y_val2, "Media Dir.",  f"{mean_r:.1f} N")

        WARN = _hex_rgb(C["alert"])
        for (ly, vy, lab, val) in [(y_lab1, y_val1, "Assim.(pico)", ap_), (y_lab2, y_val2, "Assim.(media)", am_)]:
            cx2 = x0 + 2 * third
            cv.setFillColorRGB(*_hex_rgb(C["muted"]))
            cv.setFont("Helvetica", fs_lab)
            cv.drawString(cx2 + 0.10 * cm, ly, lab)
            val_str = f"{val:.1f}%"
            if abs(val) > 10:
                cv.setFillColorRGB(*WARN)
            else:
                cv.setFillColorRGB(*_hex_rgb(C["text"]))
            cv.setFont("Helvetica-Bold", fs_val)
            cv.drawString(cx2 + 0.10 * cm, vy, val_str)

        cv.setStrokeColorRGB(*_hex_rgb(C["rule"]))
        cv.setLineWidth(0.8)
        cv.roundRect(x0, box_y, bw, metrics_box_h, 3, stroke=1, fill=0)

    cv.save()
    buf.seek(0)
    return buf.read()


# ===========================================================================
# INTERFACE
# ===========================================================================
st.markdown("# 💪 Dashboard ForceFrame – Análise de Força")
st.markdown("Carregue o CSV trace exportado pelo ForceFrame (4 canais de força: Inner L/R + Outer L/R).")
st.markdown("---")

uploaded = st.file_uploader(
    "Envie o CSV do ForceFrame",
    type=["csv"],
    help=f"Padrão do nome: {EXEMPLO_NOME}",
)

if uploaded is None:
    st.info("👆 Envie um arquivo CSV para começar.")
    st.markdown(f"**Padrão esperado:** `{EXEMPLO_NOME}`")
    st.caption("Formato: `aparelho-teste-nome-sobrenome-export-data.csv`")
    st.stop()

nome_arquivo = getattr(uploaded, "name", "arquivo.csv")
parsed = parse_filename(nome_arquivo)

# ── Info do arquivo ──────────────────────────────────────────────────────────
if parsed["valid"]:
    ap_d = parsed.get("aparelho_display", "")
    te_d = parsed.get("teste_display", "")
    st.markdown(f"""
    <div class="info-card">
        <div>
            <div class="item"><h3>Aparelho</h3><span class="valor">{ap_d}</span></div>
            <div class="item"><h3>Teste</h3><span class="valor">{te_d}</span></div>
            <div class="item"><h3>Atleta</h3><span class="valor">{parsed["atleta"]}</span></div>
            <div class="item"><h3>Data</h3><span class="valor">{parsed["data"]}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-card invalid">
        <h3>⚠️ Nome fora do padrão</h3>
        <p>Arquivo: <code>{parsed["filename"]}</code> — análise disponível, identificação automática indisponível.</p>
        <p>Padrão: <code>{EXEMPLO_NOME}</code></p>
    </div>
    """, unsafe_allow_html=True)

# ── Leitura do CSV ───────────────────────────────────────────────────────────
uploaded.seek(0)
df = pd.read_csv(uploaded, encoding="utf-8-sig", sep=None, engine="python")
time_col, il_col, ir_col, ol_col, or_col = detect_forceframe_columns(df)

# Sanitiza valores numéricos
for col in [c for c in [time_col, il_col, ir_col, ol_col, or_col] if c is not None]:
    df[col] = to_numeric(df[col])

df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
t_min = float(df[time_col].min())
t_max = float(df[time_col].max())

# ── Detecta canais ativos ────────────────────────────────────────────────────
active = {}
ALL_CHANNELS = {
    "inner_left":  il_col,
    "inner_right": ir_col,
    "outer_left":  ol_col,
    "outer_right": or_col,
}
for key, col in ALL_CHANNELS.items():
    if col is not None and col in df.columns:
        active[key] = channel_is_active(df[col])

active_cols = {k: v for k, v in ALL_CHANNELS.items() if active.get(k)}
inactive_cols = {k: v for k, v in ALL_CHANNELS.items() if not active.get(k) and v is not None}

LABEL_MAP = {
    "inner_left": "Inner Esquerda",
    "inner_right": "Inner Direita",
    "outer_left": "Outer Esquerda",
    "outer_right": "Outer Direita",
}

if inactive_cols:
    st.sidebar.markdown("**Canais inativos** (pico < 5 N):")
    for k in inactive_cols:
        st.sidebar.caption(f"• {LABEL_MAP[k]}")

# ── Visão geral ──────────────────────────────────────────────────────────────
st.markdown("### 📈 Visão Geral — Sinal Completo")
fig_overview = make_overview_figure(df, time_col, {k: v for k, v in ALL_CHANNELS.items() if v is not None and v in df.columns})
st.plotly_chart(fig_overview, use_container_width=True, key="ff_overview")

st.markdown(f"**Duração total:** {t_max - t_min:.1f}s &nbsp;|&nbsp; **Amostras:** {len(df):,} &nbsp;|&nbsp; **Canais ativos:** {len(active_cols)}")
st.markdown("---")

# ── Seleção do grupo de canais ────────────────────────────────────────────────
st.markdown("### 🎯 Selecione o grupo de canais para análise")

has_outer = ol_col in df.columns and or_col in df.columns
has_inner = il_col in df.columns and ir_col in df.columns

group_options = []
if has_outer:
    group_options.append("Outer (principal)")
if has_inner:
    group_options.append("Inner (secundário)")

if not group_options:
    st.error("Nenhum par de canais L/R encontrado no arquivo.")
    st.stop()

grupo_sel = st.radio(
    "Grupo",
    group_options,
    horizontal=True,
    key="ff_grupo",
    help="'Outer' é o canal de força principal no ForceFrame. 'Inner' captura tensão do lado interno.",
)

if "Outer" in grupo_sel:
    col_left  = ol_col
    col_right = or_col
    cor_left  = CORES["outer_left"]
    cor_right = CORES["outer_right"]
    label_left  = "Outer Esquerda"
    label_right = "Outer Direita"
else:
    col_left  = il_col
    col_right = ir_col
    cor_left  = CORES["inner_left"]
    cor_right = CORES["inner_right"]
    label_left  = "Inner Esquerda"
    label_right = "Inner Direita"

# Sugestão de janelas
t_arr = df[time_col].to_numpy()
F_left  = df[col_left].to_numpy()
F_right = df[col_right].to_numpy()

(l_t0, l_t1) = suggest_window_for_channel(t_arr, F_left)
(r_t0, r_t1) = suggest_window_for_channel(t_arr, F_right)

l_t0, l_t1 = float(np.clip(l_t0, t_min, t_max)), float(np.clip(l_t1, t_min, t_max))
r_t0, r_t1 = float(np.clip(r_t0, t_min, t_max)), float(np.clip(r_t1, t_min, t_max))

st.markdown("---")
st.markdown("### 📊 Análise por lado")
st.caption("Ajuste os sliders para delimitar a janela de melhor esforço de cada lado.")

col_L, col_R = st.columns(2)

# ── LADO ESQUERDO ─────────────────────────────────────────────────────────────
with col_L:
    badge_l = f'<span class="channel-badge" style="background:{cor_left};color:#0f1419;">ESQ</span>'
    st.markdown(f"{badge_l} **{label_left}**", unsafe_allow_html=True)

    rng_l = st.slider(
        f"Janela {label_left} [s]",
        t_min, t_max,
        (l_t0, l_t1),
        step=0.05,
        key="ff_win_left",
    )
    t0_l, t1_l = min(rng_l), max(rng_l)
    st.caption(f"Janela: {t0_l:.2f}s → {t1_l:.2f}s  (Δ = {t1_l - t0_l:.2f}s)")

    fig_l = make_channel_figure(df, time_col, col_left, t0_l, t1_l, label_left, cor_left, height=420)
    st.plotly_chart(fig_l, use_container_width=True, key="ff_chart_left")

    dfw_l = filter_window(df, time_col, t0_l, t1_l)
    m_l   = window_metrics_single(dfw_l, col_left)
    st.markdown(f"""<div class="metrics-card metrics-card-2">
        <div class="m-item"><div class="m-label">Pico Esq. (N)</div><div class="m-value">{m_l['peak']:.1f}</div></div>
        <div class="m-item"><div class="m-label">Média Esq. (N)</div><div class="m-value">{m_l['mean']:.1f}</div></div>
    </div>""", unsafe_allow_html=True)

# ── LADO DIREITO ──────────────────────────────────────────────────────────────
with col_R:
    badge_r = f'<span class="channel-badge" style="background:{cor_right};color:#0f1419;">DIR</span>'
    st.markdown(f"{badge_r} **{label_right}**", unsafe_allow_html=True)

    rng_r = st.slider(
        f"Janela {label_right} [s]",
        t_min, t_max,
        (r_t0, r_t1),
        step=0.05,
        key="ff_win_right",
    )
    t0_r, t1_r = min(rng_r), max(rng_r)
    st.caption(f"Janela: {t0_r:.2f}s → {t1_r:.2f}s  (Δ = {t1_r - t0_r:.2f}s)")

    fig_r = make_channel_figure(df, time_col, col_right, t0_r, t1_r, label_right, cor_right, height=420)
    st.plotly_chart(fig_r, use_container_width=True, key="ff_chart_right")

    dfw_r = filter_window(df, time_col, t0_r, t1_r)
    m_r   = window_metrics_single(dfw_r, col_right)
    st.markdown(f"""<div class="metrics-card metrics-card-2">
        <div class="m-item"><div class="m-label">Pico Dir. (N)</div><div class="m-value">{m_r['peak']:.1f}</div></div>
        <div class="m-item"><div class="m-label">Média Dir. (N)</div><div class="m-value">{m_r['mean']:.1f}</div></div>
    </div>""", unsafe_allow_html=True)

# ── ASSIMETRIA ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚖️ Assimetria Esq. vs Dir.")

asym_peak = asymmetry(m_l["peak"], m_r["peak"])
asym_mean = asymmetry(m_l["mean"], m_r["mean"])
dom_peak  = "Direita" if asym_peak > 0 else "Esquerda"
dom_mean  = "Direita" if asym_mean > 0 else "Esquerda"

asym_col1, asym_col2, asym_col3, asym_col4 = st.columns(4)
with asym_col1:
    st.metric("Pico Esq. (N)", f"{m_l['peak']:.1f}")
with asym_col2:
    st.metric("Pico Dir. (N)", f"{m_r['peak']:.1f}")
with asym_col3:
    delta_color = "normal" if abs(asym_peak) <= 10 else "inverse"
    st.metric("Assim. Pico", f"{asym_peak:.1f}%", delta=f"Domina: {dom_peak}", delta_color=delta_color)
with asym_col4:
    delta_color2 = "normal" if abs(asym_mean) <= 10 else "inverse"
    st.metric("Assim. Média", f"{asym_mean:.1f}%", delta=f"Domina: {dom_mean}", delta_color=delta_color2)

if abs(asym_peak) > 15:
    st.warning(f"⚠️ Assimetria de pico elevada: **{asym_peak:.1f}%** — limiar clínico recomendado: ≤ 10–15%.")
elif abs(asym_peak) > 10:
    st.info(f"ℹ️ Assimetria de pico moderada: **{asym_peak:.1f}%** — monitorar.")
else:
    st.success(f"✅ Assimetria dentro do intervalo aceitável: **{asym_peak:.1f}%**.")

# ── GRÁFICO COMPARATIVO (sobreposição normalizada) ───────────────────────────
st.markdown("---")
with st.expander("📐 Gráfico comparativo (Esq. vs Dir. na mesma escala de tempo)", expanded=True):
    fig_comp = make_comparison_figure(
        df, time_col,
        col_left,  cor_left,  label_left,
        col_right, cor_right, label_right,
        t0_l, t1_l, t0_r, t1_r,
    )
    st.plotly_chart(fig_comp, use_container_width=True, key="ff_chart_comp")
    st.caption("Os dois traços são alinhados pelo início da janela selecionada (t=0 = início de cada janela).")

# ── EXPORTAR PDF ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📄 Exportar relatório em PDF")

if not HAS_REPORTLAB or not HAS_KALEIDO:
    st.warning("Para exportar em PDF, instale: `pip install reportlab kaleido`")
else:
    fig_l_crop = make_channel_figure_cropped(df, time_col, col_left,  t0_l, t1_l, label_left,  cor_left)
    fig_r_crop = make_channel_figure_cropped(df, time_col, col_right, t0_r, t1_r, label_right, cor_right)
    fig_ov_crop = make_overview_figure(df, time_col, {k: v for k, v in ALL_CHANNELS.items() if v is not None and v in df.columns}, height=300)

    pdf_metrics = {
        "peak_left":  m_l["peak"],
        "mean_left":  m_l["mean"],
        "peak_right": m_r["peak"],
        "mean_right": m_r["mean"],
        "asym_peak":  asym_peak,
        "asym_mean":  asym_mean,
    }
    pdf_pages = [
        (f"{label_left} — {t0_l:.2f}s a {t1_l:.2f}s",  fig_l_crop,  pdf_metrics),
        (f"{label_right} — {t0_r:.2f}s a {t1_r:.2f}s", fig_r_crop,  pdf_metrics),
    ]

    pdf_bytes = build_pdf_forceframe(parsed, pdf_pages, nome_arquivo)
    if pdf_bytes:
        pdf_filename = nome_arquivo.replace(".csv", "_relatorio.pdf") if nome_arquivo.endswith(".csv") else nome_arquivo + "_relatorio.pdf"
        st.download_button(
            "⬇️ Baixar relatório em PDF",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=False,
        )
    else:
        st.error("Não foi possível gerar o PDF.")

# ── DADOS BRUTOS ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Ver dados brutos (amostra)"):
    st.dataframe(df.head(200))
