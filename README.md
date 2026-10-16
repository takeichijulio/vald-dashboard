# Dashboard VALD — Testes Neuromusculares

Aplicação em Streamlit para visualizar arquivos exportados de equipamentos VALD (incluindo NordBord e ForceFrame), analisar curvas de força e assimetria e exportar relatórios em PDF.

Este projeto foi desenvolvido por mim para uso nas avaliações de fisiologia. A referência aos equipamentos descreve o formato de arquivo suportado e não afirma adoção ou uso institucional.

## Funcionalidades

- Carregamento local de CSVs exportados de equipamentos compatíveis.
- Identificação do equipamento, teste, atleta e data a partir do nome do arquivo.
- Visualização bilateral e unilateral das curvas de força.
- Métricas de pico, média e assimetria para janelas selecionadas.
- Exportação de relatório em PDF.

## Requisitos

- Python 3.9+
- Dependências de `requirements.txt`

## Executar localmente

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```

O app abre em `http://localhost:8501`. Use o menu lateral para navegar entre as páginas.

## Estrutura do repositório

```text
Home.py
pages/
  2_NordBord.py
  3_Renomear_arquivo.py
windows_store.py
requirements.txt
```

## Dados de demonstração

Os nomes de atleta e as datas usados nos exemplos de nome de arquivo são fictícios. O repositório não inclui dados de atletas. Para testar localmente, use arquivos de demonstração sintéticos; não adicione arquivos CSV ou XLSX ao Git.

Formato esperado para identificação automática:

```text
aparelho-teste-nome-sobrenome-export-data.csv
```

Exemplo fictício:

```text
nordbord-isoprone-Atleta-Exemplo-export-08_03_2025.csv
```

## Licença e uso

O código é apresentado para fins de portfólio. Os arquivos de dados carregados localmente não fazem parte do repositório.
