"""
Página inicial do Dashboard VALD – Testes Neuromusculares.
Acesso ao Dashboard pelo menu lateral ou pelo botão abaixo.
"""
import streamlit as st

st.set_page_config(
    page_title="VALD – Testes Neuromusculares",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0f1419 0%, #1a2332 50%, #0f1419 100%); }
    .main .block-container { padding: 2rem 3rem; max-width: 900px; }
    h1, h2, h3 { font-family: 'Segoe UI', system-ui, sans-serif; color: #e8eaed !important; }
    p, span, li { color: #b8bcc4 !important; }
    .hero { font-size: 1.35rem; color: #8ab4f8; margin-bottom: 1.5rem; }
    .card-home {
        background: linear-gradient(145deg, #1c2738 0%, #232f3f 100%);
        border: 1px solid rgba(74, 158, 255, 0.25);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .card-home h3 { color: #8ab4f8 !important; margin-top: 0 !important; }
    .btn-dash { background: linear-gradient(135deg, #4a7ac4 0%, #5a8fd4 100%) !important; color: white !important; padding: 0.75rem 2rem !important; border-radius: 12px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏋️ Dashboard VALD – Testes Neuromusculares")
st.markdown('<p class="hero">Visualize testes de força (NordBord, ForceFrame e outros), analise contrações curta e longa, métricas de assimetria e exporte relatórios em PDF.</p>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class="card-home">
<h3>📌 O que é este app?</h3>
<p>Ferramenta para carregar arquivos CSV exportados dos equipamentos VALD, visualizar os sinais de força (membro esquerdo vs direito), definir janelas de análise com sliders (início e fim) e obter métricas de pico, média e assimetria. Inclui modo bilateral (2 gráficos) e modo unilateral (4 gráficos, uma perna por vez), além de exportação do relatório em PDF em uma única página.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-home">
<h3>🚀 Como usar</h3>
<ol>
<li>No menu à esquerda, clique em <strong>Dashboard</strong> (ou use o botão abaixo).</li>
<li>Envie o arquivo CSV do teste (exportado pelo equipamento VALD).</li>
<li>Ajuste os sliders de <strong>início e fim</strong> para cada janela (contração curta e longa).</li>
<li>Veja as métricas abaixo de cada gráfico e, se quiser, exporte o relatório em PDF.</li>
</ol>
<p><strong>Dica:</strong> Para identificação automática do atleta e do teste, use o nome no formato: <code>aparelho-teste-nome-sobrenome-export-data.csv</code> (ex.: nordbord-isoprone-Bernardo-Germano-export-19_02_2026.csv).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Acessar o Dashboard")

if st.button("📊 Abrir Dashboard", type="primary", use_container_width=False):
    try:
        st.switch_page("pages/2_Dashboard.py")
    except Exception:
        st.info("Use o menu lateral e clique em **Dashboard** para acessar a análise.")

st.caption("Você também pode clicar em **Dashboard** no menu lateral para ir direto à análise.")
