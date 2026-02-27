# Renomear arquivo – padronizar nome do CSV
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Renomear arquivo CSV", layout="wide", initial_sidebar_state="expanded")

PADRAO = "aparelho-teste-nome-sobrenome-export-data.csv"

APARELHOS = [
    ("nordbord", "Nordbord"),
    ("forceframe", "ForceFrame"),
]

TESTES = [
    ("isoprone", "Isoprone"),
    ("nordic", "Nordic"),
    ("iso_30", "ISO 30"),
    ("knee_extension", "Knee Extension"),
    ("ankle_plantar_flexion", "Ankle Plantar Flexion"),
    ("hip_flexion", "Hip Flexion"),
]

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0f1419 0%, #1a2332 50%, #0f1419 100%); }
    .main .block-container { padding: 2rem 2.5rem; max-width: 800px; }
    h1, h2, h3 { color: #e8eaed !important; }
    p, label { color: #b8bcc4 !important; }
    .card-rename {
        background: linear-gradient(145deg, #1c2738 0%, #232f3f 100%);
        border: 1px solid rgba(74, 158, 255, 0.25);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📁 Renomear arquivo para o padrão")
st.markdown("Envie o CSV **cru** (qualquer nome), escolha **Aparelho**, **Teste** e **Atleta** e baixe o mesmo arquivo com o nome padronizado.")
st.markdown("---")

uploaded = st.file_uploader("Envie o arquivo CSV", type=["csv"], key="rename_upload")
if uploaded is None:
    st.info("👆 Envie um CSV para continuar.")
    st.caption(f"Padrão: `{PADRAO}`")
    st.stop()

aparelho_key = st.selectbox(
    "Aparelho",
    options=[a[0] for a in APARELHOS],
    format_func=lambda x: next((a[1] for a in APARELHOS if a[0] == x), x),
    key="sel_aparelho"
)
teste_key = st.selectbox(
    "Teste",
    options=[t[0] for t in TESTES],
    format_func=lambda x: next((t[1] for t in TESTES if t[0] == x), x),
    key="sel_teste"
)
atleta_nome = st.text_input("Nome do atleta (ex.: Bernardo Germano)", placeholder="Nome Sobrenome", key="nome_atleta")
if atleta_nome and atleta_nome.strip():
    atleta_slug = "-".join(atleta_nome.strip().split())
else:
    atleta_slug = "atleta"

data_hoje = datetime.now().strftime("%d_%m_%Y")
data_input = st.text_input("Data (formato DD_MM_AAAA)", value=datetime.now().strftime("%d/%m/%Y"), key="data_rename")
try:
    from datetime import datetime as dt
    d = dt.strptime(data_input.strip(), "%d/%m/%Y")
    data_slug = d.strftime("%d_%m_%Y")
except Exception:
    data_slug = data_hoje

novo_nome = f"{aparelho_key}-{teste_key}-{atleta_slug}-export-{data_slug}.csv"

st.markdown("---")
st.markdown("### Nome gerado")
st.markdown(f"""<div class="card-rename"><code style="font-size: 1rem; color: #81c995;">{novo_nome}</code></div>""", unsafe_allow_html=True)

csv_bytes = uploaded.getvalue()
st.download_button(
    "⬇️ Baixar CSV com nome padronizado",
    data=csv_bytes,
    file_name=novo_nome,
    mime="text/csv",
    type="primary",
    use_container_width=False,
    key="dl_renamed"
)
st.caption("O conteúdo do arquivo é o mesmo; apenas o nome do download segue o padrão.")
