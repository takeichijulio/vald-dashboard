# Dashboard VALD – Testes Neuromusculares

Aplicação web para visualização e análise de testes de força (VALD / NordBord e outros equipamentos), com gráficos de contração curta e longa, métricas de assimetria e exportação de relatório em PDF.

---

## Funcionalidades

- **Upload de CSV** exportado pelos equipamentos VALD (NordBord, ForceFrame, etc.).
- **Identificação automática** do arquivo pelo nome: aparelho, tipo de teste, atleta e data (padrão: `aparelho-teste-nome-sobrenome-export-data.csv`).
- **Gráficos de força** esquerda vs direita com janelas ajustáveis (sliders de início e fim) para contração curta e longa.
- **Modo unilateral**: opção para visualizar cada perna separadamente (4 gráficos).
- **Métricas**: pico e média por membro, assimetria (pico e média) em cada janela.
- **Exportação em PDF**: relatório em uma página com gráficos no intervalo selecionado e métricas formatadas.

---

## Requisitos

- Python 3.9+
- Dependências listadas em `requirements.txt`

---

## Instalação e execução local

```bash
# Clone ou acesse a pasta do projeto
cd testes_periodicos

# Crie um ambiente virtual (recomendado)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/macOS

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação (recomendado: com página Home + Dashboard)
streamlit run Home.py
```

A aplicação abrirá no navegador em `http://localhost:8501`. A **Home** é a página inicial com a introdução; use o menu lateral ou o botão **Abrir Dashboard** para acessar o upload e os gráficos.

Para abrir direto no Dashboard (sem Home), use:
```bash
streamlit run testes.py
```

---

## Deploy na nuvem (Streamlit Community Cloud)

1. **Envie o projeto para um repositório Git** (GitHub, GitLab ou Bitbucket).

2. Acesse [share.streamlit.io](https://share.streamlit.io), faça login e clique em **New app**.

3. **Configure o app**:
   - **Repository**: seu repositório
   - **Branch**: `main` (ou a branch que usar)
   - **Main file path**: `Home.py`
   - **App URL**: opcional (ex.: `vald-testes-periodicos`)

4. Clique em **Deploy**. O Streamlit instalará as dependências a partir de `requirements.txt` e colocará o app no ar.

5. Após o deploy, a URL será algo como:  
   `https://vald-testes-periodicos-xxxxx.streamlit.app`

### Estrutura esperada no repositório

```
testes_periodicos/
├── Home.py              # Página inicial (entry point)
├── pages/
│   └── 2_Dashboard.py   # Dashboard de testes
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml      # Tema e título (opcional)
```

---

## Padrão do nome do arquivo CSV

Para que o app identifique automaticamente **aparelho**, **teste** e **atleta**, use o formato:

```
aparelho-teste-nome-sobrenome-export-data.csv
```

**Exemplo:**  
`nordbord-isoprone-Bernardo-Germano-export-19_02_2026.csv`

- Aparelho e teste são exibidos em **MAIÚSCULAS**, com `_` trocado por espaço (ex.: `ISO 30`).
- Arquivos com outro nome ainda funcionam; o app exibirá um aviso e pedirá o ajuste do nome para a identificação automática.

---

## Licença e uso

Projeto para uso interno em avaliações de testes neuromusculares (ex.: SAF Botafogo / Dados Fisiologia). Ajuste e redistribuição conforme necessidade da equipe.
