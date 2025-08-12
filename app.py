import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import gradio as gr
from fpdf import FPDF
from tempfile import NamedTemporaryFile


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
    """Calcula ratios básicos para cada empresa y año."""
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
        return pd.DataFrame()
    return df.set_index(['company', 'year']).sort_index()


def plot_kpi(ratios_df, kpi='ROA'):
    """Genera un gráfico comparativo para el KPI indicado."""
    if ratios_df.empty or kpi not in ratios_df.columns:
        fig, ax = plt.subplots()
        ax.set_title('Sin datos')
        return fig
    pivot = ratios_df[kpi].unstack(0)
    fig, ax = plt.subplots()
    pivot.plot(ax=ax, marker='o')
    ax.set_ylabel(kpi)
    ax.set_xlabel('Año')
    ax.set_title(f'{kpi} por Empresa')
    ax.grid(True)
    return fig


def run_monte_carlo(params, n_iter):
    """Simula KPIs usando distribución normal con media y desviación."""
    results = {}
    for kpi, dist in params.items():
        mean = dist.get('mean', 0)
        std = max(dist.get('std', 0), 0)
        sims = np.random.normal(mean, std, int(n_iter))
        results[kpi] = sims
    return pd.DataFrame(results)


def run_deterministic(params):
    """Escenarios determinísticos: pesimista, base y optimista."""
    rows = []
    for kpi, dist in params.items():
        mean = dist.get('mean', 0)
        std = dist.get('std', 0)
        rows.append(
            {
                'KPI': kpi,
                'Pesimista': mean - std,
                'Base': mean,
                'Optimista': mean + std,
            }
        )
    if not rows:
        return pd.DataFrame(columns=['Pesimista', 'Base', 'Optimista'])
    return pd.DataFrame(rows).set_index('KPI')


def plot_histograms(df):
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(6, 4 * len(df.columns)))
    if len(df.columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, df.columns):
        ax.hist(df[col], bins=30, alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
    fig.tight_layout()
    return fig


def run_simulation(param_rows, n_iter):
    params = {}
    for row in param_rows:
        if not row or not row[0]:
            continue
        kpi, mean, std = row[0], row[1], row[2]
        params[kpi] = {'mean': float(mean), 'std': float(std)}
    sims_df = run_monte_carlo(params, int(n_iter))
    percentiles_df = sims_df.quantile([0.05, 0.5, 0.95]).T
    percentiles_df.columns = ['p5', 'p50', 'p95']
    fig = plot_histograms(sims_df)
    deterministic_df = run_deterministic(params)
    return percentiles_df, fig, deterministic_df, sims_df, percentiles_df


def export_simulation_excel(sims_df):
    if sims_df is None or sims_df.empty:
        raise ValueError('No hay resultados para exportar')
    tmp = NamedTemporaryFile(delete=False, suffix='.xlsx')
    sims_df.to_excel(tmp.name, index=False)
    return tmp.name


def export_simulation_pdf(percentiles_df):
    if percentiles_df is None or percentiles_df.empty:
        raise ValueError('No hay resultados para exportar')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 10, 'Resumen percentiles', ln=True)
    for kpi, row in percentiles_df.iterrows():
        pdf.cell(
            0,
            10,
            f"{kpi}: P5={row['p5']:.2f} P50={row['p50']:.2f} P95={row['p95']:.2f}",
            ln=True,
        )
    tmp = NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(tmp.name)
    return tmp.name


def process_files(files, names_text):
    names = [n.strip() for n in names_text.split(',') if n.strip()]
    if len(files) != len(names):
        raise ValueError('El número de archivos y nombres debe coincidir')
    data_dict = {}
    for f, name in zip(files, names):
        data_dict.update(load_company_data(f.name, name))
    ratios_df = compute_ratios(data_dict)
    fig = plot_kpi(ratios_df, 'ROA')
    return ratios_df.reset_index(), fig
    

with gr.Blocks(title='Dashboard financiero multicompañía') as demo:
    with gr.Tab('Ratios'):
        file_input = gr.File(label='Estados financieros', file_count='multiple')
        names_input = gr.Textbox(label='Nombres de empresas (orden, separados por coma)')
        run_btn = gr.Button('Procesar')
        ratios_out = gr.Dataframe(label='Ratios comparativos')
        plot_out = gr.Plot(label='ROA comparativo')
        run_btn.click(process_files, inputs=[file_input, names_input], outputs=[ratios_out, plot_out])

    with gr.Tab('Simulación'):
        params_df = gr.Dataframe(
            headers=['kpi', 'mean', 'std'],
            datatype=['str', 'number', 'number'],
            row_count=3,
            col_count=3,
            label='Parámetros',
        )
        n_iter_input = gr.Number(value=1000, label='Iteraciones')
        sim_btn = gr.Button('Simular')
        percentiles_out = gr.Dataframe(label='Percentiles (P5, P50, P95)')
        hist_out = gr.Plot(label='Histogramas')
        deterministic_out = gr.Dataframe(label='Escenarios determinísticos')
        sims_state = gr.State()
        perc_state = gr.State()
        sim_btn.click(
            run_simulation,
            inputs=[params_df, n_iter_input],
            outputs=[percentiles_out, hist_out, deterministic_out, sims_state, perc_state],
        )
        export_excel_btn = gr.Button('Exportar Excel')
        export_pdf_btn = gr.Button('Resumen PDF')
        excel_file = gr.File(label='Simulación Excel')
        pdf_file = gr.File(label='Resumen PDF')
        export_excel_btn.click(export_simulation_excel, inputs=sims_state, outputs=excel_file)
        export_pdf_btn.click(export_simulation_pdf, inputs=perc_state, outputs=pdf_file)


if __name__ == '__main__':
    demo.launch()
