"""
Página inicial do Dashboard VALD – Testes Neuromusculares.
Acesso às páginas NordBord e ForceFrame pelo menu lateral.
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
st.markdown('<p class="hero">Visualize testes de força do NordBord e ForceFrame, analise contrações com detecção automática de janelas, métricas de assimetria e exporte relatórios em PDF.</p>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class="card-home">
<h3>📌 O que é este app?</h3>
<p>Ferramenta para carregar arquivos CSV exportados dos equipamentos VALD, visualizar os sinais de força, definir janelas de análise com sliders e obter métricas de pico, média e assimetria. Inclui exportação de relatório em PDF.</p>
<p>Equipamentos suportados:</p>
<ul>
<li>🔵 <strong>NordBord</strong> — 2 canais bilaterais simultâneos (Left Force / Right Force). Página: <em>NordBord</em>.</li>
<li>🟢 <strong>ForceFrame</strong> — 4 canais de força (Inner L/R + Outer L/R), teste sequencial por lado. Página: <em>ForceFrame</em>.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-home">
<h3>🚀 Como usar</h3>
<ol>
<li>No menu à esquerda, escolha a página do seu equipamento: <strong>NordBord</strong> ou <strong>ForceFrame</strong>.</li>
<li>Envie o arquivo CSV do teste (exportado pelo equipamento VALD).</li>
<li>Ajuste os sliders de <strong>início e fim</strong> para delimitar a janela de melhor esforço.</li>
<li>Veja as métricas e assimetria abaixo de cada gráfico e exporte o relatório em PDF.</li>
</ol>
<p><strong>Dica:</strong> Para identificação automática do atleta e do teste, use o nome no formato: <code>aparelho-teste-nome-sobrenome-export-data.csv</code> (ex.: nordbord-isoprone-Bernardo-Germano-export-19_02_2026.csv).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Acesso rápido")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🏋️ Abrir NordBord", type="primary", use_container_width=True):
        try:
            st.switch_page("pages/2_NordBord.py")
        except Exception:
            st.info("Use o menu lateral e clique em **NordBord**.")
with col_btn2:
    if st.button("💪 Abrir ForceFrame", type="secondary", use_container_width=True):
        try:
            st.switch_page("pages/4_ForceFrame.py")
        except Exception:
            st.info("Use o menu lateral e clique em **ForceFrame**.")

st.caption("Ou clique diretamente nas páginas no menu lateral.")
