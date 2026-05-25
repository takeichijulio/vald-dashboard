"""ForceFrame v2 – 4 janelas: Esq Curta/Longa + Dir Curta/Longa.
Modo simples (1 arquivo) e modo comparativo (2 arquivos).
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import windows_store

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

st.set_page_config(
    page_title="VALD / ForceFrame",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXEMPLO_NOME = "forceframe-trace-Nome-Sobrenome-export-20_05_2026.csv"

CORES = {
    "inner_left":  "#7aa2f7",
    "inner_right": "#e0af68",
    "outer_left":  "#9ece6a",
    "outer_right": "#f7768e",
}

# Cores para modo comparativo
COR_F1_ESQ = "#9ece6a"    # verde  – Arq1 Esquerda
COR_F1_DIR = "#f7768e"    # rosa   – Arq1 Direita
COR_F2_ESQ = "#7aa2f7"    # azul   – Arq2 Esquerda
COR_F2_DIR = "#e0af68"    # laranja – Arq2 Direita

# ---------------------------------------------------------------------------
# CSS
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
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .info-card > div { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; }
    .info-card .item {
        background: rgba(30, 42, 58, 0.6);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        border-left: 3px solid #7aa2f7;
    }
    .info-card .item h3 { margin: 0 0 0.25rem 0 !important; font-size: 0.72rem !important; color: #8ab4f8 !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .info-card .valor { font-size: 1.1rem; font-weight: 700; color: #e8eaed; }
    .info-card.invalid { border-color: rgba(242,139,130,0.5); background: linear-gradient(145deg,#2a1e1e 0%,#3d2828 100%); }
    .info-card.invalid h3 { color: #f28b82 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, #1c2738 0%, #232f3f 100%) !important;
        border: 1px solid rgba(74, 158, 255, 0.2) !important;
        border-radius: 14px !important;
        padding: 0.9rem 1rem !important;
    }
    [data-testid="stMetric"] label { color: #8ab4f8 !important; }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e8eaed !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a2332 0%, #0f1419 100%); }
    .section-header {
        background: linear-gradient(135deg, #1c2738 0%, #232f3f 100%);
        border-left: 4px solid #4a7ac4;
        border-radius: 0 12px 12px 0;
        padding: 0.6rem 1rem;
        margin: 1.2rem 0 0.8rem 0;
    }
    .section-header h3 { margin: 0 !important; color: #8ab4f8 !important; font-size: 1rem !important; }
    .metrics-2 {
        background: linear-gradient(160deg, #1c2738 0%, #1e2a3a 100%);
        border: 1px solid rgba(74,158,255,0.22);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
    }
    .metrics-3 {
        background: linear-gradient(160deg, #1c2738 0%, #1e2a3a 100%);
        border: 1px solid rgba(74,158,255,0.22);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.5rem;
    }
    .m-item {
        background: rgba(30,42,58,0.7);
        border-radius: 9px;
        padding: 0.55rem 0.75rem;
        border-left: 3px solid #7aa2f7;
        text-align: center;
    }
    .m-label { font-size: 0.67rem; color: #8ab4f8 !important; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.15rem; }
    .m-value { font-size: 1.0rem; font-weight: 700; color: #e8eaed !important; }
    .channel-badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 7px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em; margin-right: 0.35rem;
    }
    .delta-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 0.75rem; }
    .delta-table th { background: #1a2332; color: #8ab4f8 !important; padding: 0.4rem 0.6rem; text-align: left; border-bottom: 1px solid #2d3d4f; }
    .delta-table td { padding: 0.35rem 0.6rem; border-bottom: 1px solid rgba(74,158,255,0.1); color: #b8bcc4 !important; }
    .delta-table tr:nth-child(even) td { background: rgba(30,42,58,0.4); }
    .delta-pos { color: #f7768e !important; font-weight: 600; }
    .delta-neg { color: #9ece6a !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Utilitários
# ===========================================================================
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
    cols = [c.strip().strip('"').strip('﻿') for c in df.columns]
    df.columns = cols

    def _find(candidates):
        for cand in candidates:
            for c in cols:
                if cand.lower() in c.lower():
                    return c
        return None

    time_col    = _find(["second", "time", "tempo"]) or cols[0]
    inner_left  = _find(["inner left",  "inner_left"])  or (cols[1] if len(cols) > 1 else None)
    inner_right = _find(["inner right", "inner_right"]) or (cols[2] if len(cols) > 2 else None)
    outer_left  = _find(["outer left",  "outer_left"])  or (cols[3] if len(cols) > 3 else None)
    outer_right = _find(["outer right", "outer_right"]) or (cols[4] if len(cols) > 4 else None)
    return time_col, inner_left, inner_right, outer_left, outer_right


def channel_is_active(series: pd.Series, threshold: float = 5.0) -> bool:
    return float(series.abs().max()) > threshold


def suggest_two_windows(t: np.ndarray, F: np.ndarray, margin: float = 0.3):
    """Detecta segmentos de atividade e retorna (curta, longa) = ([-2], [-1])."""
    if len(F) == 0 or len(t) == 0:
        return (0.0, 1.0), (0.0, 1.0)
    t_full = (float(t[0]), float(t[-1]))

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0025
    win = max(5, int(round(0.1 / dt)))
    F_s = pd.Series(np.abs(F)).rolling(win, center=True, min_periods=1).mean().to_numpy()

    base_mask = t <= (t.min() + 0.5)
    baseline = float(np.nanmean(F_s[base_mask])) if base_mask.any() else 0.0
    std_b    = float(np.nanstd(F_s[base_mask]))  if base_mask.any() else 0.0
    thr = max(baseline + 5.0 * std_b, 5.0)

    segments, in_seg, start = [], False, None
    for ti, mi in zip(t, F_s > thr):
        if mi and not in_seg:
            in_seg, start = True, float(ti)
        elif not mi and in_seg:
            segments.append((start, float(ti)))
            in_seg = False
    if in_seg:
        segments.append((start, float(t[-1])))
    segments = [(a, b) for a, b in segments if (b - a) >= 0.2]

    def _w(seg):
        return (max(t_full[0], seg[0] - margin), min(t_full[1], seg[1] + margin))

    if len(segments) == 0:
        return t_full, t_full
    if len(segments) == 1:
        w = _w(segments[0])
        return w, w
    return _w(segments[-2]), _w(segments[-1])


def window_metrics_single(dfw: pd.DataFrame, col: str) -> dict:
    F = dfw[col].dropna().to_numpy()
    if len(F) == 0:
        return {"peak": float("nan"), "mean": float("nan")}
    return {"peak": float(np.nanmax(F)), "mean": float(np.nanmean(F))}


def asymmetry(v_left: float, v_right: float) -> float:
    denom = max(abs(v_left), abs(v_right), 1e-9)
    return 100.0 * (v_right - v_left) / denom


def filter_window(df: pd.DataFrame, time_col: str, t0: float, t1: float) -> pd.DataFrame:
    return df[(df[time_col] >= t0) & (df[time_col] <= t1)].copy()


def load_file_data(uploaded):
    """Lê CSV ForceFrame → (df, time_col, col_left_outer, col_right_outer, il, ir, parsed)."""
    nome = getattr(uploaded, "name", "arquivo.csv")
    uploaded.seek(0)
    df = pd.read_csv(uploaded, encoding="utf-8-sig", sep=None, engine="python")
    time_col, il, ir, ol, or_ = detect_forceframe_columns(df)
    for c in [c2 for c2 in [time_col, il, ir, ol, or_] if c2 is not None]:
        df[c] = to_numeric(df[c])
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    parsed = parse_filename(nome)
    return df, time_col, il, ir, ol, or_, parsed


# ===========================================================================
# Figuras
# ===========================================================================
_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(30,42,58,0.6)",
    font=dict(color="#e8eaed", size=12),
    xaxis=dict(gridcolor="rgba(45,61,79,0.8)"),
    yaxis=dict(gridcolor="rgba(45,61,79,0.8)"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def make_overview_figure(df, time_col, channels: dict, height=300) -> go.Figure:
    fig = go.Figure()
    labels = {"inner_left": "Inner Esq.", "inner_right": "Inner Dir.",
               "outer_left": "Outer Esq.", "outer_right": "Outer Dir."}
    widths = {"inner_left": 1.5, "inner_right": 1.5, "outer_left": 2.5, "outer_right": 2.5}
    for key, col in channels.items():
        if col is None:
            continue
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[col], mode="lines",
            name=labels.get(key, key),
            line=dict(color=CORES[key], width=widths.get(key, 2)),
        ))
    fig.update_layout(title="Visão geral – todos os canais",
                      xaxis_title="T (s)", yaxis_title="Força (N)",
                      legend_title="Canal", height=height, **_LAYOUT_BASE)
    return fig


def make_channel_figure(df, time_col, col, t0, t1, title, cor, height=380) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df[col], mode="lines", name="Força",
        line=dict(color=cor, width=2),
    ))
    fig.add_vrect(x0=t0, x1=t1, fillcolor="rgba(120,120,120,0.15)", line_width=0)
    fig.update_layout(title=title, xaxis_title="T (s)", yaxis_title="Força (N)",
                      showlegend=False, height=height, **_LAYOUT_BASE)
    return fig


def make_channel_figure_cropped(df, time_col, col, t0, t1, title, cor, height=280) -> go.Figure:
    fig = make_channel_figure(df, time_col, col, t0, t1, title, cor, height)
    fig.update_layout(xaxis_range=[t0, t1], margin=dict(t=30, b=24, l=40, r=14))
    return fig


# ===========================================================================
# PDF – helpers internos
# ===========================================================================
def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


_PDF_C = {
    "header_bg":   "#1a2332", "header_sub": "#8ab4f8", "accent":    "#4a7ac4",
    "body_bg":     "#ffffff", "text":       "#1c2738", "muted":     "#5a6677",
    "rule":        "#c8d0db", "title_bg":   "#eef2f7", "sum_hdr":   "#1a2332",
    "col_L":       "#dce8ff", "col_R":      "#fff3dc", "col_A":     "#f0f2f5",
    "col_delta":   "#fdf0e8", "alert":      "#c0392b",
    "row_even":    "#f7f9fc", "row_odd":    "#ffffff",
}


def _pdf_fill(cv, key):
    cv.setFillColorRGB(*_hex_rgb(_PDF_C[key]))


def _pdf_stroke(cv, key):
    cv.setStrokeColorRGB(*_hex_rgb(_PDF_C[key]))


def _draw_header(cv, w, h, mrg, cw, title: str, sublines: list) -> float:
    """Desenha cabeçalho; retorna body_top (y abaixo do cabeçalho)."""
    hdr_h   = 1.05 * cm + len(sublines) * 0.36 * cm + 0.30 * cm
    hdr_top = h - mrg
    body_top = hdr_top - hdr_h

    cv.setFillColorRGB(*_hex_rgb(_PDF_C["header_bg"]))
    cv.rect(mrg, body_top + 2, cw, hdr_h - 2, stroke=0, fill=1)
    cv.setFillColorRGB(*_hex_rgb(_PDF_C["accent"]))
    cv.rect(mrg, body_top, cw, 2, stroke=0, fill=1)

    cv.setFillColorRGB(1, 1, 1)
    cv.setFont("Helvetica-Bold", 12.5)
    cv.drawString(mrg + 0.35 * cm, hdr_top - 0.44 * cm, title)

    cv.setFillColorRGB(*_hex_rgb(_PDF_C["header_sub"]))
    cv.setFont("Helvetica", 8.0)
    sy = hdr_top - 0.44 * cm - 0.37 * cm
    for line in sublines:
        cv.drawString(mrg + 0.35 * cm, sy, line[:130])
        sy -= 0.35 * cm

    cv.setFillColorRGB(1, 1, 1)
    cv.rect(mrg, mrg, cw, body_top - mrg, stroke=0, fill=1)
    return body_top


def _draw_chart_cell(cv, fig, titulo: str, x0, cell_top, cell_w, cell_h, metrics: dict):
    """Desenha um bloco (título + gráfico + box de métricas) dentro de uma célula."""
    inner_pad = 0.12 * cm
    bcw = cell_w - 2 * inner_pad
    xi  = x0 + inner_pad

    title_h = 0.50 * cm
    mbox_h  = 1.35 * cm
    gap     = 0.10 * cm
    img_h   = cell_h - title_h - gap - mbox_h

    # Barra de título
    _pdf_fill(cv, "title_bg")
    _pdf_stroke(cv, "rule")
    cv.setLineWidth(0.5)
    cv.roundRect(xi + 0.08 * cm, cell_top - title_h, bcw - 0.16 * cm, title_h, 3, stroke=1, fill=1)
    _pdf_fill(cv, "text")
    cv.setFont("Helvetica-Bold", 8.5)
    cv.drawString(xi + 0.28 * cm, cell_top - title_h + 0.14 * cm,
                  titulo[:56] + ("…" if len(titulo) > 56 else ""))

    # Gráfico
    img_top = cell_top - title_h
    img_ok  = False
    dw = dh = 0.0
    img_buf = io.BytesIO()
    try:
        fe = go.Figure(fig.to_dict())
        fe.update_layout(
            template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#f5f7fa",
            font=dict(color="#1c2738", size=10),
            xaxis=dict(gridcolor="#d0d7e3", linecolor="#8a9ab5",
                       tickfont=dict(color="#1c2738"), title_font=dict(color="#1c2738")),
            yaxis=dict(gridcolor="#d0d7e3", linecolor="#8a9ab5",
                       tickfont=dict(color="#1c2738"), title_font=dict(color="#1c2738")),
            showlegend=False,
            height=max(280, int(img_h * 1.6)),
            margin=dict(t=20, b=24, l=42, r=12),
        )
        for sc in (1, 2):
            try:
                img_buf.seek(0); img_buf.truncate(0)
                fe.write_image(img_buf, format="png", scale=sc, engine="kaleido")
                img_buf.seek(0)
                ir = ImageReader(img_buf)
                iw, ih = ir.getSize()
                slot_w = bcw - 0.08 * cm
                sc2 = min(slot_w / iw, img_h / ih)
                dw, dh = iw * sc2, ih * sc2
                ix = xi + (slot_w - dw) / 2
                cv.drawImage(ir, ix, img_top - dh, width=dw, height=dh, mask="auto")
                img_ok = True
                break
            except Exception:
                continue
    except Exception:
        pass

    if not img_ok:
        _pdf_fill(cv, "muted")
        cv.setFont("Helvetica", 7.5)
        cv.drawString(xi, img_top - img_h * 0.5, "(Gráfico indisponível)")

    # Box de métricas
    box_y = (img_top - dh if img_ok else img_top - img_h) - gap - mbox_h
    half  = bcw / 2.0

    cv.setFillColorRGB(*_hex_rgb(_PDF_C["col_L"]))
    cv.rect(xi, box_y, half, mbox_h, stroke=0, fill=1)
    cv.setFillColorRGB(*_hex_rgb(_PDF_C["col_R"]))
    cv.rect(xi + half, box_y, half, mbox_h, stroke=0, fill=1)

    _pdf_stroke(cv, "rule")
    cv.setLineWidth(0.3)
    cv.line(xi + half, box_y, xi + half, box_y + mbox_h)
    cv.setLineWidth(0.7)
    cv.roundRect(xi, box_y, bcw, mbox_h, 3, stroke=1, fill=0)

    peak_v = float(metrics.get("peak", 0) or 0)
    mean_v = float(metrics.get("mean", 0) or 0)
    lbl_y  = box_y + mbox_h - 0.36 * cm
    val_y  = box_y + mbox_h - 0.86 * cm

    _pdf_fill(cv, "muted")
    cv.setFont("Helvetica", 7.0)
    cv.drawString(xi + 0.16 * cm, lbl_y, "Pico (N)")
    cv.drawString(xi + half + 0.16 * cm, lbl_y, "Média (N)")
    _pdf_fill(cv, "text")
    cv.setFont("Helvetica-Bold", 12)
    cv.drawString(xi + 0.16 * cm, val_y, f"{peak_v:.1f}")
    cv.drawString(xi + half + 0.16 * cm, val_y, f"{mean_v:.1f}")


def _draw_summary_bar_single(cv, mrg, cw, bar_h: float, data: dict):
    """Barra de resumo do PDF simples: 4 colunas (Curta Esq/Dir/Assim + Longa Esq/Dir/Assim)."""
    bar_top = mrg + bar_h
    cv.setFillColorRGB(*_hex_rgb(_PDF_C["sum_hdr"]))
    cv.rect(mrg, bar_top - 0.50 * cm, cw, 0.50 * cm, stroke=0, fill=1)
    cv.setFillColorRGB(1, 1, 1)
    cv.setFont("Helvetica-Bold", 8.5)
    cv.drawString(mrg + 0.3 * cm, bar_top - 0.34 * cm, "Resumo: Curta e Longa — Esquerda vs Direita")

    rows = [
        ("Pico Esq C (N)",  data.get("peak_esq_curta",  0), "col_L", 0, 0),
        ("Pico Dir C (N)",  data.get("peak_dir_curta",  0), "col_R", 0, 1),
        ("Assim. Curta",    data.get("asym_curta",      0), "col_A", 0, 2),
        ("Pico Esq L (N)",  data.get("peak_esq_longa",  0), "col_L", 1, 0),
        ("Pico Dir L (N)",  data.get("peak_dir_longa",  0), "col_R", 1, 1),
        ("Assim. Longa",    data.get("asym_longa",      0), "col_A", 1, 2),
    ]
    cell_area_h = bar_h - 0.50 * cm
    scell_h = cell_area_h / 2
    scell_w = cw / 3

    for lbl, val, bgk, row_s, col_s in rows:
        cx = mrg + col_s * scell_w
        cy = mrg + (1 - row_s) * scell_h
        cv.setFillColorRGB(*_hex_rgb(_PDF_C[bgk]))
        cv.rect(cx, cy, scell_w, scell_h, stroke=0, fill=1)
        _pdf_stroke(cv, "rule")
        cv.setLineWidth(0.3)
        if col_s > 0:
            cv.line(cx, cy, cx, cy + scell_h)
        if row_s > 0:
            cv.line(cx, cy + scell_h, cx + scell_w, cy + scell_h)
        lbl_y = cy + scell_h - 0.31 * cm
        val_y = cy + scell_h - 0.80 * cm
        _pdf_fill(cv, "muted")
        cv.setFont("Helvetica", 6.8)
        cv.drawString(cx + 0.16 * cm, lbl_y, lbl)
        fval = float(val or 0)
        is_asym = "Assim" in lbl
        val_str = f"{fval:.1f}%" if is_asym else f"{fval:.1f}"
        if is_asym and abs(fval) > 10:
            cv.setFillColorRGB(*_hex_rgb(_PDF_C["alert"]))
        else:
            _pdf_fill(cv, "text")
        cv.setFont("Helvetica-Bold", 12)
        cv.drawString(cx + 0.16 * cm, val_y, val_str)


def _draw_delta_mini(cv, mrg, cw, bottom_y, bar_h: float, lado: str,
                     d1: dict, d2: dict, label1: str, label2: str):
    """Tabela delta compacta (Curta + Longa) na base de uma página comparativa."""
    top = bottom_y + bar_h
    cv.setFillColorRGB(*_hex_rgb(_PDF_C["sum_hdr"]))
    cv.rect(mrg, top - 0.46 * cm, cw, 0.46 * cm, stroke=0, fill=1)
    cv.setFillColorRGB(1, 1, 1)
    cv.setFont("Helvetica-Bold", 8)
    cv.drawString(mrg + 0.3 * cm, top - 0.30 * cm,
                  f"Δ {lado}: {label2} (base) → {label1} (novo)")

    headers = ["Contração", f"{label1} Pico", f"{label2} Pico", "Δ Pico",
               f"{label1} Média", f"{label2} Média", "Δ Média"]
    col_pcts = [0.20, 0.115, 0.115, 0.10, 0.115, 0.115, 0.10]
    col_xs = [mrg]
    for pct in col_pcts[:-1]:
        col_xs.append(col_xs[-1] + pct * cw)
    row_h = (bar_h - 0.46 * cm) / 3   # header + 2 rows

    # Header row
    hy = top - 0.46 * cm - row_h
    cv.setFillColorRGB(*_hex_rgb("#1e2d3e"))
    cv.rect(mrg, hy, cw, row_h, stroke=0, fill=1)
    cv.setFillColorRGB(*_hex_rgb(_PDF_C["header_sub"]))
    cv.setFont("Helvetica-Bold", 6.5)
    for xi, hdr in zip(col_xs, headers):
        cv.drawString(xi + 0.08 * cm, hy + row_h * 0.3, hdr)

    rows_data = [("Curta", "curta"), ("Longa", "longa")]
    for i, (nome_c, key) in enumerate(rows_data):
        ry = hy - (i + 1) * row_h
        bg = _PDF_C["row_even"] if i % 2 == 0 else _PDF_C["row_odd"]
        cv.setFillColorRGB(*_hex_rgb(bg))
        cv.rect(mrg, ry, cw, row_h, stroke=0, fill=1)
        p1 = float(d1.get(f"peak_{key}",  0) or 0)  # Arq1 (novo)
        m1 = float(d1.get(f"mean_{key}",  0) or 0)
        p2 = float(d2.get(f"peak_{key}",  0) or 0)  # Arq2 (base)
        m2 = float(d2.get(f"mean_{key}",  0) or 0)
        dp = ((p1 - p2) / max(abs(p2), 1e-9)) * 100  # (Arq1-Arq2)/|Arq2|
        dm = ((m1 - m2) / max(abs(m2), 1e-9)) * 100
        vals = [nome_c, f"{p1:.1f}", f"{p2:.1f}", f"{dp:+.1f}%",
                f"{m1:.1f}", f"{m2:.1f}", f"{dm:+.1f}%"]
        vy = ry + row_h * 0.28
        for j, (xi, val) in enumerate(zip(col_xs, vals)):
            is_delta = j in (3, 6)
            dv = dp if j == 3 else dm
            if is_delta and abs(dv) > 0.5:
                cv.setFillColorRGB(*_hex_rgb(_PDF_C["alert"] if dv < 0 else "#2d7a3a"))
                cv.setFont("Helvetica-Bold", 7)
            else:
                _pdf_fill(cv, "text")
                cv.setFont("Helvetica", 7)
            cv.drawString(xi + 0.08 * cm, vy, val)

    _pdf_stroke(cv, "rule")
    cv.setLineWidth(0.4)
    cv.roundRect(mrg, bottom_y, cw, bar_h, 3, stroke=1, fill=0)


# ===========================================================================
# PDF – modo simples
# ===========================================================================
def build_pdf_single(parsed: dict, data: dict, nome_arquivo: str) -> bytes | None:
    """
    data = {
        "esq_curta": {"fig", "peak", "mean"},
        "esq_longa": {"fig", "peak", "mean"},
        "dir_curta": {"fig", "peak", "mean"},
        "dir_longa": {"fig", "peak", "mean"},
        "peak_esq_curta", "peak_dir_curta", "asym_curta",
        "peak_esq_longa", "peak_dir_longa", "asym_longa",
    }
    """
    if not HAS_REPORTLAB:
        return None

    buf = io.BytesIO()
    cv = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    mrg = 0.7 * cm
    cw  = w - 2 * mrg

    ap = parsed.get("aparelho_display") or "ForceFrame"
    te = parsed.get("teste_display") or ""
    if parsed.get("valid"):
        sublines = [f"Aparelho: {ap}   •   Teste: {te}   •   Atleta: {parsed.get('atleta','—')}   •   Data: {parsed.get('data','—')}"]
    else:
        sublines = [f"Arquivo: {parsed.get('filename', nome_arquivo)}"]

    body_top = _draw_header(cv, w, h, mrg, cw,
                             "ForceFrame – Relatório de Teste", sublines)
    content_h = body_top - mrg

    SUM_H   = 2.80 * cm
    SUM_GAP = 0.25 * cm
    avail_h = content_h - SUM_H - SUM_GAP

    ncols, nrows = 2, 2
    cell_w = cw / ncols
    cell_h = avail_h / nrows

    cells = [
        ("Esq. Curta",  "esq_curta", 0, 0),
        ("Dir. Curta",  "dir_curta", 0, 1),
        ("Esq. Longa",  "esq_longa", 1, 0),
        ("Dir. Longa",  "dir_longa", 1, 1),
    ]
    for nome_c, key, row, col in cells:
        cd = data.get(key, {})
        fig = cd.get("fig")
        if fig is None:
            continue
        x0       = mrg + col * cell_w
        cell_top = body_top - row * cell_h
        _draw_chart_cell(cv, fig, nome_c, x0, cell_top, cell_w, cell_h,
                         {"peak": cd.get("peak", 0), "mean": cd.get("mean", 0)})

    _draw_summary_bar_single(cv, mrg, cw, SUM_H, data)
    cv.save()
    buf.seek(0)
    return buf.read()


# ===========================================================================
# PDF – modo comparativo (3 páginas)
# ===========================================================================
def build_pdf_comparison(parsed1: dict, parsed2: dict,
                          data1: dict, data2: dict,
                          nome1: str, nome2: str) -> bytes | None:
    """
    Página 1: ESQUERDA (Arq1 C | Arq2 C / Arq1 L | Arq2 L) + Δ
    Página 2: DIREITA  (mesmo padrão)
    Página 3: tabela resumo completa
    """
    if not HAS_REPORTLAB:
        return None

    buf = io.BytesIO()
    cv = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    mrg = 0.7 * cm
    cw  = w - 2 * mrg

    def _atleta(p, nome_arq):
        if p.get("valid"):
            return f"{p.get('atleta','—')} ({p.get('data','—')})"
        return nome_arq

    arq1_lbl = _atleta(parsed1, nome1)
    arq2_lbl = _atleta(parsed2, nome2)

    # ── Páginas ESQUERDA e DIREITA ─────────────────────────────────────────
    sides = [
        ("ESQUERDA", "esq_curta", "esq_longa", COR_F1_ESQ, COR_F2_ESQ),
        ("DIREITA",  "dir_curta", "dir_longa", COR_F1_DIR, COR_F2_DIR),
    ]

    for lado, key_c, key_l, _cor1, _cor2 in sides:
        sublines = [
            f"Lado: {lado}   •   Arq 1: {arq1_lbl}",
            f"Arq 2: {arq2_lbl}",
        ]
        body_top = _draw_header(cv, w, h, mrg, cw,
                                 f"ForceFrame – Comparativo {lado}", sublines)
        content_h = body_top - mrg

        DELTA_H   = 2.20 * cm
        DELTA_GAP = 0.20 * cm
        avail_h = content_h - DELTA_H - DELTA_GAP

        cell_w = cw / 2
        cell_h = avail_h / 2

        quad = [
            (f"Arq1 – {lado[0]}. Curta",  data1.get(key_c, {}), 0, 0),
            (f"Arq2 – {lado[0]}. Curta",  data2.get(key_c, {}), 0, 1),
            (f"Arq1 – {lado[0]}. Longa",  data1.get(key_l, {}), 1, 0),
            (f"Arq2 – {lado[0]}. Longa",  data2.get(key_l, {}), 1, 1),
        ]
        for titulo_c, cd, row, col in quad:
            fig = cd.get("fig")
            if fig is None:
                continue
            x0       = mrg + col * cell_w
            cell_top = body_top - row * cell_h
            _draw_chart_cell(cv, fig, titulo_c, x0, cell_top, cell_w, cell_h,
                             {"peak": cd.get("peak", 0), "mean": cd.get("mean", 0)})

        # Tabela Δ no rodapé
        d1_side = {
            "peak_curta": data1.get(key_c, {}).get("peak", 0),
            "mean_curta": data1.get(key_c, {}).get("mean", 0),
            "peak_longa": data1.get(key_l, {}).get("peak", 0),
            "mean_longa": data1.get(key_l, {}).get("mean", 0),
        }
        d2_side = {
            "peak_curta": data2.get(key_c, {}).get("peak", 0),
            "mean_curta": data2.get(key_c, {}).get("mean", 0),
            "peak_longa": data2.get(key_l, {}).get("peak", 0),
            "mean_longa": data2.get(key_l, {}).get("mean", 0),
        }
        _draw_delta_mini(cv, mrg, cw, mrg, DELTA_H, lado,
                         d1_side, d2_side, arq1_lbl[:22], arq2_lbl[:22])
        cv.showPage()

    # ── Página 3: tabela resumo completa ──────────────────────────────────
    body_top = _draw_header(cv, w, h, mrg, cw,
                             "ForceFrame – Resumo Comparativo",
                             [f"Arq 1: {arq1_lbl}", f"Arq 2: {arq2_lbl}"])

    contrações = [
        ("Esq. Curta",  "esq_curta"),
        ("Esq. Longa",  "esq_longa"),
        ("Dir. Curta",  "dir_curta"),
        ("Dir. Longa",  "dir_longa"),
    ]
    col_headers = ["Contração",
                   "Arq1 Pico (N)", "Arq2 Pico (N)", "Δ Pico (%)",
                   "Arq1 Méd (N)",  "Arq2 Méd (N)",  "Δ Méd (%)"]
    col_pcts = [0.20, 0.115, 0.115, 0.11, 0.115, 0.115, 0.11]

    # Constrói xs
    col_xs = [mrg]
    for pct in col_pcts[:-1]:
        col_xs.append(col_xs[-1] + pct * cw)

    row_h   = 1.05 * cm
    hdr_h   = 0.60 * cm
    tbl_top = body_top - 0.4 * cm
    tbl_w   = sum(p * cw for p in col_pcts)

    # Linha de cabeçalho
    cv.setFillColorRGB(*_hex_rgb(_PDF_C["sum_hdr"]))
    cv.rect(mrg, tbl_top - hdr_h, tbl_w, hdr_h, stroke=0, fill=1)
    cv.setFillColorRGB(*_hex_rgb(_PDF_C["header_sub"]))
    cv.setFont("Helvetica-Bold", 7.5)
    for xi, hdr in zip(col_xs, col_headers):
        cv.drawString(xi + 0.1 * cm, tbl_top - hdr_h + 0.18 * cm, hdr)

    for i, (nome_c, key) in enumerate(contrações):
        ry = tbl_top - hdr_h - (i + 1) * row_h
        bg = _PDF_C["row_even"] if i % 2 == 0 else _PDF_C["row_odd"]
        cv.setFillColorRGB(*_hex_rgb(bg))
        cv.rect(mrg, ry, tbl_w, row_h, stroke=0, fill=1)

        cd1 = data1.get(key, {})
        cd2 = data2.get(key, {})
        p1 = float(cd1.get("peak", 0) or 0)  # Arq1 (novo)
        m1 = float(cd1.get("mean", 0) or 0)
        p2 = float(cd2.get("peak", 0) or 0)  # Arq2 (base)
        m2 = float(cd2.get("mean", 0) or 0)
        dp = ((p1 - p2) / max(abs(p2), 1e-9)) * 100  # (Arq1-Arq2)/|Arq2|
        dm = ((m1 - m2) / max(abs(m2), 1e-9)) * 100
        vals = [nome_c, f"{p1:.1f}", f"{p2:.1f}", f"{dp:+.1f}%",
                f"{m1:.1f}", f"{m2:.1f}", f"{dm:+.1f}%"]

        vy = ry + row_h * 0.32
        for j, (xi, val) in enumerate(zip(col_xs, vals)):
            is_delta = j in (3, 6)
            dv = dp if j == 3 else dm
            if is_delta and abs(dv) > 0.5:
                cv.setFillColorRGB(*_hex_rgb(_PDF_C["alert"] if dv < 0 else "#2d7a3a"))
                cv.setFont("Helvetica-Bold", 8.5)
            elif j == 0:
                _pdf_fill(cv, "text")
                cv.setFont("Helvetica-Bold", 8.5)
            else:
                _pdf_fill(cv, "text")
                cv.setFont("Helvetica", 8.5)
            cv.drawString(xi + 0.1 * cm, vy, val)

    # Borda da tabela
    _pdf_stroke(cv, "rule")
    cv.setLineWidth(0.6)
    tbl_h_total = hdr_h + len(contrações) * row_h
    cv.roundRect(mrg, tbl_top - tbl_h_total, tbl_w, tbl_h_total, 3, stroke=1, fill=0)

    # Legenda
    leg_y = tbl_top - tbl_h_total - 0.6 * cm
    cv.setFillColorRGB(*_hex_rgb(_PDF_C["muted"]))
    cv.setFont("Helvetica-Oblique", 7.5)
    cv.drawString(mrg, leg_y, "Δ = (Arq1 − Arq2) / |Arq2|. Positivo (verde) = Arq1 maior que Arq2 (evolução). Negativo (vermelho) = regressão.")

    cv.save()
    buf.seek(0)
    return buf.read()


# ===========================================================================
# UI helpers
# ===========================================================================
def render_file_info(parsed: dict, prefix: str = ""):
    if parsed.get("valid"):
        ap = parsed.get("aparelho_display", "")
        te = parsed.get("teste_display", "")
        label = f"{prefix} " if prefix else ""
        st.markdown(f"""
        <div class="info-card">
            <div>
                <div class="item"><h3>{label}Aparelho</h3><span class="valor">{ap}</span></div>
                <div class="item"><h3>Teste</h3><span class="valor">{te}</span></div>
                <div class="item"><h3>Atleta</h3><span class="valor">{parsed["atleta"]}</span></div>
                <div class="item"><h3>Data</h3><span class="valor">{parsed["data"]}</span></div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-card invalid">
            <h3>⚠️ Nome fora do padrão</h3>
            <p>Arquivo: <code>{parsed["filename"]}</code> — identificação automática indisponível.</p>
        </div>""", unsafe_allow_html=True)


def _window_block(df, time_col, col, cor, t_min, t_max, label, key_prefix,
                  suggested_curta, suggested_longa, height=360):
    """Renderiza os 2 sliders + gráficos + métricas de um canal (Curta e Longa)."""
    st.markdown(
        f'<span class="channel-badge" style="background:{cor};color:#0f1419;">'
        f'{label.upper()}</span> **{label}**', unsafe_allow_html=True)

    # — Curta —
    st.caption("Contração **Curta**")
    ec0, ec1 = float(np.clip(suggested_curta[0], t_min, t_max)), float(np.clip(suggested_curta[1], t_min, t_max))
    rng_c = st.slider(f"Janela Curta – {label} [s]", t_min, t_max, (ec0, ec1),
                       step=0.05, key=f"{key_prefix}_curta")
    t0_c, t1_c = min(rng_c), max(rng_c)
    st.caption(f"{t0_c:.2f}s → {t1_c:.2f}s  (Δ = {t1_c - t0_c:.2f}s)")
    fig_c = make_channel_figure(df, time_col, col, t0_c, t1_c, f"{label} – Curta", cor, height=height)
    st.plotly_chart(fig_c, use_container_width=True, key=f"chart_{key_prefix}_curta")
    dfw_c = filter_window(df, time_col, t0_c, t1_c)
    m_c   = window_metrics_single(dfw_c, col)
    st.markdown(f"""<div class="metrics-2">
        <div class="m-item"><div class="m-label">Pico (N)</div><div class="m-value">{m_c['peak']:.1f}</div></div>
        <div class="m-item"><div class="m-label">Média (N)</div><div class="m-value">{m_c['mean']:.1f}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    # — Longa —
    st.caption("Contração **Longa**")
    el0, el1 = float(np.clip(suggested_longa[0], t_min, t_max)), float(np.clip(suggested_longa[1], t_min, t_max))
    rng_l = st.slider(f"Janela Longa – {label} [s]", t_min, t_max, (el0, el1),
                       step=0.05, key=f"{key_prefix}_longa")
    t0_l, t1_l = min(rng_l), max(rng_l)
    st.caption(f"{t0_l:.2f}s → {t1_l:.2f}s  (Δ = {t1_l - t0_l:.2f}s)")
    fig_l = make_channel_figure(df, time_col, col, t0_l, t1_l, f"{label} – Longa", cor, height=height)
    st.plotly_chart(fig_l, use_container_width=True, key=f"chart_{key_prefix}_longa")
    dfw_l = filter_window(df, time_col, t0_l, t1_l)
    m_l   = window_metrics_single(dfw_l, col)
    st.markdown(f"""<div class="metrics-2">
        <div class="m-item"><div class="m-label">Pico (N)</div><div class="m-value">{m_l['peak']:.1f}</div></div>
        <div class="m-item"><div class="m-label">Média (N)</div><div class="m-value">{m_l['mean']:.1f}</div></div>
    </div>""", unsafe_allow_html=True)

    return (t0_c, t1_c, m_c), (t0_l, t1_l, m_l)


def _comparison_column(df, time_col, col, cor, t_min, t_max, label, key_prefix,
                        suggested_curta, suggested_longa, height=320):
    """Coluna de comparação: slider+gráfico para curta e longa de 1 arquivo."""
    # Curta
    st.caption(f"**Curta** – {label}")
    ec0 = float(np.clip(suggested_curta[0], t_min, t_max))
    ec1 = float(np.clip(suggested_curta[1], t_min, t_max))
    rng_c = st.slider(f"Janela Curta {label} [s]", t_min, t_max, (ec0, ec1),
                       step=0.05, key=f"{key_prefix}_curta")
    t0_c, t1_c = min(rng_c), max(rng_c)
    fig_c = make_channel_figure(df, time_col, col, t0_c, t1_c,
                                 f"{label} Curta", cor, height=height)
    st.plotly_chart(fig_c, use_container_width=True, key=f"chart_{key_prefix}_curta")
    dfw_c = filter_window(df, time_col, t0_c, t1_c)
    m_c   = window_metrics_single(dfw_c, col)
    st.markdown(f"""<div class="metrics-2">
        <div class="m-item"><div class="m-label">Pico (N)</div><div class="m-value">{m_c['peak']:.1f}</div></div>
        <div class="m-item"><div class="m-label">Média (N)</div><div class="m-value">{m_c['mean']:.1f}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Longa
    st.caption(f"**Longa** – {label}")
    el0 = float(np.clip(suggested_longa[0], t_min, t_max))
    el1 = float(np.clip(suggested_longa[1], t_min, t_max))
    rng_l = st.slider(f"Janela Longa {label} [s]", t_min, t_max, (el0, el1),
                       step=0.05, key=f"{key_prefix}_longa")
    t0_l, t1_l = min(rng_l), max(rng_l)
    fig_l = make_channel_figure(df, time_col, col, t0_l, t1_l,
                                 f"{label} Longa", cor, height=height)
    st.plotly_chart(fig_l, use_container_width=True, key=f"chart_{key_prefix}_longa")
    dfw_l = filter_window(df, time_col, t0_l, t1_l)
    m_l   = window_metrics_single(dfw_l, col)
    st.markdown(f"""<div class="metrics-2">
        <div class="m-item"><div class="m-label">Pico (N)</div><div class="m-value">{m_l['peak']:.1f}</div></div>
        <div class="m-item"><div class="m-label">Média (N)</div><div class="m-value">{m_l['mean']:.1f}</div></div>
    </div>""", unsafe_allow_html=True)

    return (t0_c, t1_c, m_c), (t0_l, t1_l, m_l)


# ===========================================================================
# INTERFACE PRINCIPAL
# ===========================================================================
st.markdown("# 💪 ForceFrame – Análise de Força")
st.markdown("Carregue o CSV trace exportado pelo ForceFrame (4 canais: Inner L/R + Outer L/R).")
st.markdown("---")

# ── Upload Arquivo 1 ─────────────────────────────────────────────────────────
uploaded1 = st.file_uploader(
    "📂 Arquivo 1 (obrigatório)",
    type=["csv"],
    help=f"Padrão: {EXEMPLO_NOME}",
    key="ff_upload1",
)

if uploaded1 is None:
    st.info("👆 Envie um arquivo CSV para começar.")
    st.caption(f"Padrão: `{EXEMPLO_NOME}`")
    st.stop()

# ── Upload Arquivo 2 (comparativo, na sidebar) ───────────────────────────────
st.sidebar.markdown("### Modo Comparativo")
st.sidebar.caption("Envie um segundo arquivo para comparar dois testes lado a lado.")
uploaded2 = st.sidebar.file_uploader(
    "📂 Arquivo 2 (opcional)",
    type=["csv"],
    key="ff_upload2",
)
MODO_COMP = uploaded2 is not None

# ── Grupo de canais ───────────────────────────────────────────────────────────
st.sidebar.markdown("### Canais")
grupo_sel = st.sidebar.radio(
    "Grupo de canais",
    ["Outer (principal)", "Inner (secundário)"],
    index=0,
    key="ff_grupo",
    help="Outer é o canal de força principal no ForceFrame.",
)
USE_OUTER = "Outer" in grupo_sel

# ── Carrega Arquivo 1 ─────────────────────────────────────────────────────────
df1, tc1, il1, ir1, ol1, or1, parsed1 = load_file_data(uploaded1)
nome1 = getattr(uploaded1, "name", "arquivo1.csv")

col_left1  = ol1 if USE_OUTER else il1
col_right1 = or1 if USE_OUTER else ir1
cor_left1  = CORES["outer_left"]  if USE_OUTER else CORES["inner_left"]
cor_right1 = CORES["outer_right"] if USE_OUTER else CORES["inner_right"]

t_min1 = float(df1[tc1].min())
t_max1 = float(df1[tc1].max())
t_arr1 = df1[tc1].to_numpy()

if col_left1 is None or col_left1 not in df1.columns:
    st.error("Canal esquerdo não encontrado no Arquivo 1.")
    st.stop()
if col_right1 is None or col_right1 not in df1.columns:
    st.error("Canal direito não encontrado no Arquivo 1.")
    st.stop()

curta1_esq, longa1_esq = suggest_two_windows(t_arr1, df1[col_left1].to_numpy())
curta1_dir, longa1_dir = suggest_two_windows(t_arr1, df1[col_right1].to_numpy())

# ── Restaurar janelas salvas quando o arquivo muda ───────────────────────────
if st.session_state.get("_ff1_loaded") != nome1:
    st.session_state["_ff1_loaded"] = nome1
    _sv_ff1 = windows_store.load_windows(nome1)
    if _sv_ff1:
        # Modo simples
        st.session_state["ff_esq_curta"] = (float(_sv_ff1.get("esq_curta_t0", curta1_esq[0])), float(_sv_ff1.get("esq_curta_t1", curta1_esq[1])))
        st.session_state["ff_esq_longa"] = (float(_sv_ff1.get("esq_longa_t0", longa1_esq[0])), float(_sv_ff1.get("esq_longa_t1", longa1_esq[1])))
        st.session_state["ff_dir_curta"] = (float(_sv_ff1.get("dir_curta_t0", curta1_dir[0])), float(_sv_ff1.get("dir_curta_t1", curta1_dir[1])))
        st.session_state["ff_dir_longa"] = (float(_sv_ff1.get("dir_longa_t0", longa1_dir[0])), float(_sv_ff1.get("dir_longa_t1", longa1_dir[1])))
        # Modo comparativo – mesmas janelas para as chaves de Arq1
        st.session_state["ff_cmp_a1e_curta"] = st.session_state["ff_esq_curta"]
        st.session_state["ff_cmp_a1e_longa"] = st.session_state["ff_esq_longa"]
        st.session_state["ff_cmp_a1d_curta"] = st.session_state["ff_dir_curta"]
        st.session_state["ff_cmp_a1d_longa"] = st.session_state["ff_dir_longa"]

render_file_info(parsed1, prefix="Arq 1" if MODO_COMP else "")

# ── Visão geral Arquivo 1 ─────────────────────────────────────────────────────
ch_map1 = {k: v for k, v in {
    "inner_left": il1, "inner_right": ir1,
    "outer_left": ol1, "outer_right": or1,
}.items() if v is not None and v in df1.columns}
fig_ov1 = make_overview_figure(df1, tc1, ch_map1, height=280)
st.plotly_chart(fig_ov1, use_container_width=True, key="ff_ov1")
st.caption(f"Duração: {t_max1 - t_min1:.1f}s  |  Amostras: {len(df1):,}")
st.markdown("---")

# ===========================================================================
# MODO SIMPLES
# ===========================================================================
if not MODO_COMP:
    st.markdown("### 📊 Análise por Contração")

    label_left  = "Outer Esq." if USE_OUTER else "Inner Esq."
    label_right = "Outer Dir." if USE_OUTER else "Inner Dir."

    col_e, col_d = st.columns(2)

    with col_e:
        (t0_ec, t1_ec, m_ec), (t0_el, t1_el, m_el) = _window_block(
            df1, tc1, col_left1, cor_left1, t_min1, t_max1,
            label_left, "ff_esq",
            curta1_esq, longa1_esq,
        )

    with col_d:
        (t0_dc, t1_dc, m_dc), (t0_dl, t1_dl, m_dl) = _window_block(
            df1, tc1, col_right1, cor_right1, t_min1, t_max1,
            label_right, "ff_dir",
            curta1_dir, longa1_dir,
        )

    # ── Assimetria ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚖️ Assimetria")

    asym_c_peak = asymmetry(m_ec["peak"], m_dc["peak"])
    asym_c_mean = asymmetry(m_ec["mean"], m_dc["mean"])
    asym_l_peak = asymmetry(m_el["peak"], m_dl["peak"])
    asym_l_mean = asymmetry(m_el["mean"], m_dl["mean"])

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.metric("Curta – Pico Esq (N)", f"{m_ec['peak']:.1f}")
    with ac2:
        st.metric("Curta – Pico Dir (N)", f"{m_dc['peak']:.1f}")
    with ac3:
        delta_c = "normal" if abs(asym_c_peak) <= 10 else "inverse"
        dom_c = "Dir" if asym_c_peak > 0 else "Esq"
        st.metric("Assim. Curta Pico", f"{asym_c_peak:.1f}%", delta=f"Dom: {dom_c}", delta_color=delta_c)
    with ac4:
        delta_l = "normal" if abs(asym_l_peak) <= 10 else "inverse"
        dom_l = "Dir" if asym_l_peak > 0 else "Esq"
        st.metric("Assim. Longa Pico", f"{asym_l_peak:.1f}%", delta=f"Dom: {dom_l}", delta_color=delta_l)

    for asym_val, label_asym in [(asym_c_peak, "Curta"), (asym_l_peak, "Longa")]:
        if abs(asym_val) > 15:
            st.warning(f"⚠️ Assim. **{label_asym}** elevada: **{asym_val:.1f}%** (limiar ≤ 15%).")
        elif abs(asym_val) > 10:
            st.info(f"ℹ️ Assim. **{label_asym}** moderada: **{asym_val:.1f}%** — monitorar.")
        else:
            st.success(f"✅ Assim. **{label_asym}** aceitável: **{asym_val:.1f}%**.")

    # ── Salvar janelas ────────────────────────────────────────────────────────
    st.markdown("---")
    _ff_sv = windows_store.load_windows(nome1)
    _fsc1, _fsc2, _fsc3 = st.columns([1.3, 1.3, 5])
    with _fsc1:
        if st.button("💾 Salvar janelas", key="ff_save_s", use_container_width=True):
            windows_store.save_windows(nome1, {
                "esq_curta_t0": st.session_state.get("ff_esq_curta", curta1_esq)[0],
                "esq_curta_t1": st.session_state.get("ff_esq_curta", curta1_esq)[1],
                "esq_longa_t0": st.session_state.get("ff_esq_longa", longa1_esq)[0],
                "esq_longa_t1": st.session_state.get("ff_esq_longa", longa1_esq)[1],
                "dir_curta_t0": st.session_state.get("ff_dir_curta", curta1_dir)[0],
                "dir_curta_t1": st.session_state.get("ff_dir_curta", curta1_dir)[1],
                "dir_longa_t0": st.session_state.get("ff_dir_longa", longa1_dir)[0],
                "dir_longa_t1": st.session_state.get("ff_dir_longa", longa1_dir)[1],
            })
            st.success("✅ Salvo!")
    with _fsc2:
        if _ff_sv and st.button("🗑️ Apagar save", key="ff_del_s", use_container_width=True):
            windows_store.delete_windows(nome1)
            st.info("Save apagado.")
    with _fsc3:
        if _ff_sv:
            st.caption(f"✅ Janelas salvas em {_ff_sv.get('_saved_at', '')} — carregadas automaticamente.")

    # ── Exportar PDF ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Exportar Relatório em PDF")

    if not HAS_REPORTLAB or not HAS_KALEIDO:
        st.warning("Para exportar em PDF instale: `pip install reportlab kaleido`")
    else:
        pdf_data = {
            "esq_curta": {
                "fig":  make_channel_figure_cropped(df1, tc1, col_left1,  t0_ec, t1_ec, f"{label_left} – Curta",  cor_left1),
                "peak": m_ec["peak"], "mean": m_ec["mean"],
            },
            "esq_longa": {
                "fig":  make_channel_figure_cropped(df1, tc1, col_left1,  t0_el, t1_el, f"{label_left} – Longa",  cor_left1),
                "peak": m_el["peak"], "mean": m_el["mean"],
            },
            "dir_curta": {
                "fig":  make_channel_figure_cropped(df1, tc1, col_right1, t0_dc, t1_dc, f"{label_right} – Curta", cor_right1),
                "peak": m_dc["peak"], "mean": m_dc["mean"],
            },
            "dir_longa": {
                "fig":  make_channel_figure_cropped(df1, tc1, col_right1, t0_dl, t1_dl, f"{label_right} – Longa", cor_right1),
                "peak": m_dl["peak"], "mean": m_dl["mean"],
            },
            "peak_esq_curta": m_ec["peak"], "peak_dir_curta": m_dc["peak"],
            "asym_curta": asym_c_peak,
            "peak_esq_longa": m_el["peak"], "peak_dir_longa": m_dl["peak"],
            "asym_longa": asym_l_peak,
        }
        with st.spinner("Gerando PDF…"):
            pdf_bytes = build_pdf_single(parsed1, pdf_data, nome1)

        if pdf_bytes:
            pdf_fn = nome1.replace(".csv", "_relatorio.pdf") if nome1.endswith(".csv") else nome1 + "_relatorio.pdf"
            st.download_button("⬇️ Baixar PDF", data=pdf_bytes, file_name=pdf_fn,
                                mime="application/pdf", use_container_width=False)
        else:
            st.error("Não foi possível gerar o PDF.")

# ===========================================================================
# MODO COMPARATIVO
# ===========================================================================
else:
    df2, tc2, il2, ir2, ol2, or2, parsed2 = load_file_data(uploaded2)
    nome2 = getattr(uploaded2, "name", "arquivo2.csv")

    col_left2  = ol2 if USE_OUTER else il2
    col_right2 = or2 if USE_OUTER else ir2

    t_min2 = float(df2[tc2].min())
    t_max2 = float(df2[tc2].max())
    t_arr2 = df2[tc2].to_numpy()

    if col_left2 is None or col_left2 not in df2.columns:
        st.error("Canal esquerdo não encontrado no Arquivo 2.")
        st.stop()
    if col_right2 is None or col_right2 not in df2.columns:
        st.error("Canal direito não encontrado no Arquivo 2.")
        st.stop()

    curta2_esq, longa2_esq = suggest_two_windows(t_arr2, df2[col_left2].to_numpy())
    curta2_dir, longa2_dir = suggest_two_windows(t_arr2, df2[col_right2].to_numpy())

    # ── Restaurar janelas salvas do Arq2 ──────────────────────────────────────
    if st.session_state.get("_ff2_loaded") != nome2:
        st.session_state["_ff2_loaded"] = nome2
        _sv_ff2 = windows_store.load_windows(nome2)
        if _sv_ff2:
            st.session_state["ff_cmp_a2e_curta"] = (float(_sv_ff2.get("esq_curta_t0", curta2_esq[0])), float(_sv_ff2.get("esq_curta_t1", curta2_esq[1])))
            st.session_state["ff_cmp_a2e_longa"] = (float(_sv_ff2.get("esq_longa_t0", longa2_esq[0])), float(_sv_ff2.get("esq_longa_t1", longa2_esq[1])))
            st.session_state["ff_cmp_a2d_curta"] = (float(_sv_ff2.get("dir_curta_t0", curta2_dir[0])), float(_sv_ff2.get("dir_curta_t1", curta2_dir[1])))
            st.session_state["ff_cmp_a2d_longa"] = (float(_sv_ff2.get("dir_longa_t0", longa2_dir[0])), float(_sv_ff2.get("dir_longa_t1", longa2_dir[1])))

    render_file_info(parsed2, prefix="Arq 2")

    ch_map2 = {k: v for k, v in {
        "inner_left": il2, "inner_right": ir2,
        "outer_left": ol2, "outer_right": or2,
    }.items() if v is not None and v in df2.columns}
    fig_ov2 = make_overview_figure(df2, tc2, ch_map2, height=240)
    with st.expander("📈 Visão geral – Arquivo 2", expanded=False):
        st.plotly_chart(fig_ov2, use_container_width=True, key="ff_ov2")
        st.caption(f"Duração: {t_max2 - t_min2:.1f}s  |  Amostras: {len(df2):,}")

    def _arq_label(p, nome):
        return p.get("atleta") + " (" + p.get("data") + ")" if p.get("valid") else nome

    arq1_lbl = _arq_label(parsed1, nome1)
    arq2_lbl = _arq_label(parsed2, nome2)

    # ── ESQUERDA ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""<div class="section-header"><h3>🔵 ESQUERDA</h3></div>""", unsafe_allow_html=True)
    col_a1e, col_a2e = st.columns(2)

    with col_a1e:
        st.markdown(f"**Arq 1** – {arq1_lbl}")
        (t0_1ec, t1_1ec, m_1ec), (t0_1el, t1_1el, m_1el) = _comparison_column(
            df1, tc1, col_left1, COR_F1_ESQ, t_min1, t_max1,
            "Esq Arq1", "ff_cmp_a1e",
            curta1_esq, longa1_esq,
        )
        # Save Arq1
        st.markdown("---")
        _ff_sv1 = windows_store.load_windows(nome1)
        _f1a, _f1b, _f1c = st.columns([1.3, 1.3, 3])
        with _f1a:
            if st.button("💾 Salvar Arq1", key="ff_cmp_save1", use_container_width=True):
                windows_store.save_windows(nome1, {
                    "esq_curta_t0": st.session_state.get("ff_cmp_a1e_curta", curta1_esq)[0],
                    "esq_curta_t1": st.session_state.get("ff_cmp_a1e_curta", curta1_esq)[1],
                    "esq_longa_t0": st.session_state.get("ff_cmp_a1e_longa", longa1_esq)[0],
                    "esq_longa_t1": st.session_state.get("ff_cmp_a1e_longa", longa1_esq)[1],
                    "dir_curta_t0": st.session_state.get("ff_cmp_a1d_curta", curta1_dir)[0],
                    "dir_curta_t1": st.session_state.get("ff_cmp_a1d_curta", curta1_dir)[1],
                    "dir_longa_t0": st.session_state.get("ff_cmp_a1d_longa", longa1_dir)[0],
                    "dir_longa_t1": st.session_state.get("ff_cmp_a1d_longa", longa1_dir)[1],
                })
                st.success("✅ Salvo!")
        with _f1b:
            if _ff_sv1 and st.button("🗑️ Apagar", key="ff_cmp_del1", use_container_width=True):
                windows_store.delete_windows(nome1)
                st.info("Apagado.")
        with _f1c:
            if _ff_sv1:
                st.caption(f"✅ {_ff_sv1.get('_saved_at', '')}")

    with col_a2e:
        st.markdown(f"**Arq 2** – {arq2_lbl}")
        (t0_2ec, t1_2ec, m_2ec), (t0_2el, t1_2el, m_2el) = _comparison_column(
            df2, tc2, col_left2, COR_F2_ESQ, t_min2, t_max2,
            "Esq Arq2", "ff_cmp_a2e",
            curta2_esq, longa2_esq,
        )
        # Save Arq2
        st.markdown("---")
        _ff_sv2 = windows_store.load_windows(nome2)
        _f2a, _f2b, _f2c = st.columns([1.3, 1.3, 3])
        with _f2a:
            if st.button("💾 Salvar Arq2", key="ff_cmp_save2", use_container_width=True):
                windows_store.save_windows(nome2, {
                    "esq_curta_t0": st.session_state.get("ff_cmp_a2e_curta", curta2_esq)[0],
                    "esq_curta_t1": st.session_state.get("ff_cmp_a2e_curta", curta2_esq)[1],
                    "esq_longa_t0": st.session_state.get("ff_cmp_a2e_longa", longa2_esq)[0],
                    "esq_longa_t1": st.session_state.get("ff_cmp_a2e_longa", longa2_esq)[1],
                    "dir_curta_t0": st.session_state.get("ff_cmp_a2d_curta", curta2_dir)[0],
                    "dir_curta_t1": st.session_state.get("ff_cmp_a2d_curta", curta2_dir)[1],
                    "dir_longa_t0": st.session_state.get("ff_cmp_a2d_longa", longa2_dir)[0],
                    "dir_longa_t1": st.session_state.get("ff_cmp_a2d_longa", longa2_dir)[1],
                })
                st.success("✅ Salvo!")
        with _f2b:
            if _ff_sv2 and st.button("🗑️ Apagar", key="ff_cmp_del2", use_container_width=True):
                windows_store.delete_windows(nome2)
                st.info("Apagado.")
        with _f2c:
            if _ff_sv2:
                st.caption(f"✅ {_ff_sv2.get('_saved_at', '')}")

    # Tabela Δ Esquerda
    # Δ = (Arq1 - Arq2) / |Arq2| × 100 → Arq1 é o mais recente (referência)
    dp_ec = ((m_1ec["peak"] - m_2ec["peak"]) / max(abs(m_2ec["peak"]), 1e-9)) * 100
    dm_ec = ((m_1ec["mean"] - m_2ec["mean"]) / max(abs(m_2ec["mean"]), 1e-9)) * 100
    dp_el = ((m_1el["peak"] - m_2el["peak"]) / max(abs(m_2el["peak"]), 1e-9)) * 100
    dm_el = ((m_1el["mean"] - m_2el["mean"]) / max(abs(m_2el["mean"]), 1e-9)) * 100

    def _delta_cls(v): return "delta-pos" if v > 0.5 else ("delta-neg" if v < -0.5 else "")

    st.markdown(f"""
    <table class="delta-table">
    <tr>
        <th>Contração</th>
        <th>Arq1 Pico (N)</th><th>Arq2 Pico (N)</th><th>Δ Pico</th>
        <th>Arq1 Méd (N)</th><th>Arq2 Méd (N)</th><th>Δ Média</th>
    </tr>
    <tr>
        <td><b>Curta</b></td>
        <td>{m_1ec['peak']:.1f}</td><td>{m_2ec['peak']:.1f}</td>
        <td class="{_delta_cls(dp_ec)}">{dp_ec:+.1f}%</td>
        <td>{m_1ec['mean']:.1f}</td><td>{m_2ec['mean']:.1f}</td>
        <td class="{_delta_cls(dm_ec)}">{dm_ec:+.1f}%</td>
    </tr>
    <tr>
        <td><b>Longa</b></td>
        <td>{m_1el['peak']:.1f}</td><td>{m_2el['peak']:.1f}</td>
        <td class="{_delta_cls(dp_el)}">{dp_el:+.1f}%</td>
        <td>{m_1el['mean']:.1f}</td><td>{m_2el['mean']:.1f}</td>
        <td class="{_delta_cls(dm_el)}">{dm_el:+.1f}%</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

    # ── DIREITA ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""<div class="section-header"><h3>🔴 DIREITA</h3></div>""", unsafe_allow_html=True)
    col_a1d, col_a2d = st.columns(2)

    with col_a1d:
        st.markdown(f"**Arq 1** – {arq1_lbl}")
        (t0_1dc, t1_1dc, m_1dc), (t0_1dl, t1_1dl, m_1dl) = _comparison_column(
            df1, tc1, col_right1, COR_F1_DIR, t_min1, t_max1,
            "Dir Arq1", "ff_cmp_a1d",
            curta1_dir, longa1_dir,
        )

    with col_a2d:
        st.markdown(f"**Arq 2** – {arq2_lbl}")
        (t0_2dc, t1_2dc, m_2dc), (t0_2dl, t1_2dl, m_2dl) = _comparison_column(
            df2, tc2, col_right2, COR_F2_DIR, t_min2, t_max2,
            "Dir Arq2", "ff_cmp_a2d",
            curta2_dir, longa2_dir,
        )

    dp_dc = ((m_1dc["peak"] - m_2dc["peak"]) / max(abs(m_2dc["peak"]), 1e-9)) * 100
    dm_dc = ((m_1dc["mean"] - m_2dc["mean"]) / max(abs(m_2dc["mean"]), 1e-9)) * 100
    dp_dl = ((m_1dl["peak"] - m_2dl["peak"]) / max(abs(m_2dl["peak"]), 1e-9)) * 100
    dm_dl = ((m_1dl["mean"] - m_2dl["mean"]) / max(abs(m_2dl["mean"]), 1e-9)) * 100

    st.markdown(f"""
    <table class="delta-table">
    <tr>
        <th>Contração</th>
        <th>Arq1 Pico (N)</th><th>Arq2 Pico (N)</th><th>Δ Pico</th>
        <th>Arq1 Méd (N)</th><th>Arq2 Méd (N)</th><th>Δ Média</th>
    </tr>
    <tr>
        <td><b>Curta</b></td>
        <td>{m_1dc['peak']:.1f}</td><td>{m_2dc['peak']:.1f}</td>
        <td class="{_delta_cls(dp_dc)}">{dp_dc:+.1f}%</td>
        <td>{m_1dc['mean']:.1f}</td><td>{m_2dc['mean']:.1f}</td>
        <td class="{_delta_cls(dm_dc)}">{dm_dc:+.1f}%</td>
    </tr>
    <tr>
        <td><b>Longa</b></td>
        <td>{m_1dl['peak']:.1f}</td><td>{m_2dl['peak']:.1f}</td>
        <td class="{_delta_cls(dp_dl)}">{dp_dl:+.1f}%</td>
        <td>{m_1dl['mean']:.1f}</td><td>{m_2dl['mean']:.1f}</td>
        <td class="{_delta_cls(dm_dl)}">{dm_dl:+.1f}%</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

    # ── Exportar PDF comparativo ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Exportar Relatório Comparativo em PDF")

    if not HAS_REPORTLAB or not HAS_KALEIDO:
        st.warning("Para exportar em PDF instale: `pip install reportlab kaleido`")
    else:
        def _crop(df, tc, col, t0, t1, lbl, cor):
            return make_channel_figure_cropped(df, tc, col, t0, t1, lbl, cor)

        pdf_data1 = {
            "esq_curta": {"fig": _crop(df1, tc1, col_left1,  t0_1ec, t1_1ec, "Esq Curta Arq1", COR_F1_ESQ),
                          "peak": m_1ec["peak"], "mean": m_1ec["mean"]},
            "esq_longa": {"fig": _crop(df1, tc1, col_left1,  t0_1el, t1_1el, "Esq Longa Arq1", COR_F1_ESQ),
                          "peak": m_1el["peak"], "mean": m_1el["mean"]},
            "dir_curta": {"fig": _crop(df1, tc1, col_right1, t0_1dc, t1_1dc, "Dir Curta Arq1", COR_F1_DIR),
                          "peak": m_1dc["peak"], "mean": m_1dc["mean"]},
            "dir_longa": {"fig": _crop(df1, tc1, col_right1, t0_1dl, t1_1dl, "Dir Longa Arq1", COR_F1_DIR),
                          "peak": m_1dl["peak"], "mean": m_1dl["mean"]},
        }
        pdf_data2 = {
            "esq_curta": {"fig": _crop(df2, tc2, col_left2,  t0_2ec, t1_2ec, "Esq Curta Arq2", COR_F2_ESQ),
                          "peak": m_2ec["peak"], "mean": m_2ec["mean"]},
            "esq_longa": {"fig": _crop(df2, tc2, col_left2,  t0_2el, t1_2el, "Esq Longa Arq2", COR_F2_ESQ),
                          "peak": m_2el["peak"], "mean": m_2el["mean"]},
            "dir_curta": {"fig": _crop(df2, tc2, col_right2, t0_2dc, t1_2dc, "Dir Curta Arq2", COR_F2_DIR),
                          "peak": m_2dc["peak"], "mean": m_2dc["mean"]},
            "dir_longa": {"fig": _crop(df2, tc2, col_right2, t0_2dl, t1_2dl, "Dir Longa Arq2", COR_F2_DIR),
                          "peak": m_2dl["peak"], "mean": m_2dl["mean"]},
        }

        with st.spinner("Gerando PDF comparativo (3 páginas)…"):
            pdf_bytes = build_pdf_comparison(parsed1, parsed2,
                                              pdf_data1, pdf_data2,
                                              nome1, nome2)
        if pdf_bytes:
            base1 = nome1.replace(".csv", "") if nome1.endswith(".csv") else nome1
            base2 = nome2.replace(".csv", "") if nome2.endswith(".csv") else nome2
            pdf_fn = f"comparativo_{base1}_vs_{base2}.pdf"
            st.download_button("⬇️ Baixar PDF Comparativo", data=pdf_bytes,
                                file_name=pdf_fn, mime="application/pdf",
                                use_container_width=False)
        else:
            st.error("Não foi possível gerar o PDF.")

# ── Dados brutos ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Dados brutos – Arquivo 1 (amostra)"):
    st.dataframe(df1.head(200))
if MODO_COMP:
    with st.expander("📋 Dados brutos – Arquivo 2 (amostra)"):
        st.dataframe(df2.head(200))
