import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import gradio as gr


RATIO_DESCRIPTIONS = {
    "ROA": (
        "Return on Assets (ROA) eval\u00faa la eficiencia con la que la empresa "
        "utiliza sus activos para generar utilidades. Un valor m\u00e1s alto "
        "indica un mejor uso de los activos."
    ),
}


def _to_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(' ', '').replace('%', '')
    if s.count(',') > 1 and '.' in s:
        s = s.replace(',', '')
    elif s.count('.') > 1 and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        if ',' in s and '.' not in s:
            s = s.replace(',', '.')
        s = s.replace(',', '')
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_multi_year_sheet(path, sheet_name):
    """Extrae cuentas y montos anuales de una hoja Excel."""
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    hdr = None
    for r in range(min(50, df.shape[0])):
        if str(df.iat[r, 0]).strip().upper() == 'CONCEPTOS':
            hdr = r
            break
    if hdr is None:
        raise ValueError(f"No se encontró 'CONCEPTOS' en la hoja {sheet_name}")
    date_row = hdr + 1

    def _get_year_from(cell):
        dt = pd.to_datetime(cell, errors='coerce')
        if pd.isna(dt):
            return None
        return int(dt.year)

    years = []
    cols = []
    for c in range(1, df.shape[1]):
        cell = df.iat[hdr, c]
        if pd.isna(cell):
            continue
        year = _get_year_from(cell)
        if year is None:
            cell2 = df.iat[date_row, c]
            year = _get_year_from(cell2)
        if year is None:
            continue
        years.append(year)
        cols.append(c)
    data = {'Cuenta': df.iloc[hdr + 2:, 0].astype(str).str.strip()}
    for year, c in zip(years, cols):
        data[year] = df.iloc[hdr + 2:, c].apply(_to_number)
    df_wide = pd.DataFrame(data)
    return df_wide, years


def pick_value(df, patterns, year, prefer_max=True):
    """Obtiene el valor que coincide con cualquiera de los patrones dados."""
    mask = pd.Series(False, index=df.index)
    for pat in patterns:
        looks_regex = bool(re.search(r"[.\^$*+?{}\[\]|()\\]", pat))
        if looks_regex:
            pat_nc = re.sub(r"\((?!\?)", "(?:", pat)
            m = df['Cuenta'].str.contains(pat_nc, case=False, regex=True, na=False)
        else:
            m = df['Cuenta'].str.contains(pat, case=False, regex=False, na=False)
        mask = mask | m
    vals = df.loc[mask, year]
    if vals.empty:
        return np.nan
    return vals.max(skipna=True) if prefer_max else vals.sum(skipna=True)


def sdiv(n, d):
    return np.nan if (d in [0, None] or pd.isna(d) or pd.isna(n) or d == 0) else n / d


def load_company_data(file_obj, company_name):
    """Carga los datos de una empresa usando un archivo Excel."""
    balance_wide, bal_years = parse_multi_year_sheet(file_obj, 'ESTRUCTURA FINANCIERA')
    er_wide, er_years = parse_multi_year_sheet(file_obj, 'ESTRUCTURA ECONOMICA')
    return {
        company_name: {
            'balance': balance_wide,
            'eres': er_wide,
            'balance_years': bal_years,
            'er_years': er_years,
        }
    }


def compute_ratios(data_dict):
    """Calcula ratios básicos para cada empresa y año.

    El DataFrame resultante incluye en ``attrs['descriptions']`` una
    explicación de cada ratio calculado, lo que facilita su interpretación.
    """
    rows = []
    for company, info in data_dict.items():
        years = sorted(set(info['balance_years']) & set(info['er_years']))
        for y in years:
            assets = pick_value(info['balance'], [r'^TOTAL\s+ACTIVOS?$'], y, prefer_max=True)
            net_income = pick_value(info['eres'], [r'UTILIDAD\s+N[IÍ]ETA'], y, prefer_max=True)
            roa = sdiv(net_income, assets)
            rows.append({
                'company': company,
                'year': y,
                'Total Activos': assets,
                'Utilidad Neta': net_income,
                'ROA': roa,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        df.attrs['descriptions'] = RATIO_DESCRIPTIONS
        return df
    df = df.set_index(['company', 'year']).sort_index()
    df.attrs['descriptions'] = RATIO_DESCRIPTIONS
    return df


def plot_kpi(ratios_df, kpi='ROA', years=5):
    """Genera un gráfico comparativo para el KPI indicado.

    Solo se muestran los últimos ``years`` años disponibles.
    """
    if ratios_df.empty or kpi not in ratios_df.columns:
        fig, ax = plt.subplots()
        ax.set_title('Sin datos')
        return fig
    pivot = ratios_df[kpi].unstack(0).sort_index().tail(years)
    fig, ax = plt.subplots()
    pivot.plot(ax=ax, marker='o')
    ax.set_ylabel(kpi)
    ax.set_xlabel('Año')
    ax.set_title(f'{kpi} por Empresa')
    ax.grid(True)
    return fig


def process_files(files, names_text, statements, ratio):
    """Procesa los archivos y genera las vistas solicitadas."""
    names = [n.strip() for n in names_text.split(',') if n.strip()]
    if len(files) != len(names):
        raise ValueError('El número de archivos y nombres debe coincidir')

    # Cargar datos por empresa
    data_dict = {}
    for f, name in zip(files, names):
        data_dict.update(load_company_data(f.name, name))

    # Preparar tablas de estados financieros si se solicitaron
    balance_df = pd.DataFrame()
    er_df = pd.DataFrame()
    if 'Balance' in statements:
        frames = []
        for company, info in data_dict.items():
            wide = info['balance']
            long = wide.melt('Cuenta', var_name='year', value_name='valor')
            long['company'] = company
            frames.append(long)
        if frames:
            balance_df = pd.concat(frames, ignore_index=True)

    if 'Estado de Resultados' in statements:
        frames = []
        for company, info in data_dict.items():
            wide = info['eres']
            long = wide.melt('Cuenta', var_name='year', value_name='valor')
            long['company'] = company
            frames.append(long)
        if frames:
            er_df = pd.concat(frames, ignore_index=True)

    # Calcular ratios
    ratios_df = compute_ratios(data_dict)



demo = gr.Interface(
    fn=process_files,
    inputs=[
        gr.File(label='Estados financieros', file_count='multiple'),
        gr.Textbox(label='Nombres de empresas (orden, separados por coma)'),
        gr.CheckboxGroup(
            label='Vistas de estados',
            choices=['Balance', 'Estado de Resultados'],
            value=['Balance'],
        ),
        gr.Dropdown(
            label='Ratio',
            choices=['ROA', 'Total Activos', 'Utilidad Neta'],
            value='ROA',
        ),
    ],
    outputs=[

    ],
    title='Dashboard financiero multicompañía',
)

if __name__ == '__main__':
    demo.launch()
