import pandas as pd
import numpy as np 
import statsmodels.api as sm
from scipy import stats
import os
import datetime
import sys
import traceback
from itertools import product
import re

# --- CONFIGURACIÓN PRINCIPAL: PANEL 3 (PORTAFOLIO) ---

# 1. ARCHIVOS DE ENTRADA
FILE_PANEL_3 = 'panel_3_portafolio.xlsx'
FILE_PREDICTORES = 'dataset_final_interacciones.xlsx'
FILE_CONTROLES = 'variables_control_final.xlsx'

# 2. NOMBRES DE COLUMNAS
COL_FECHA = 'Fecha'
COL_FONDO = 'Fondo'
COL_EMISOR = 'Emisor_Origen'
COL_SECTOR = 'Sector'
VAR_Y_DEPENDIENTE = 'Stock_%'

# 3. VARIABLE DE EXCLUSIÓN
COL_DUMMY_EXCLUSION = 'Dummy_Inicio_Fondo0'

# Variables IV/Moderadora
VARS_IV_MOD = [
    'PC1_Global_c',
    'PC1_Sistematico_c',
    'D_COVID',
    'Int_Global_COVID',
    'Int_Sistematico_COVID'
]

# Variables de Control
VARS_CONTROL = [
    'Tasa_Referencia_BCRP',
    'Inflacion_t_1',
    'PBI_Crecimiento_Interanual',
    'Tipo_Cambio'
]

def sanitize_filename(name):
    """Limpia nombres de archivo/carpeta para Windows."""
    # Reemplazar caracteres problemáticos
    name = re.sub(r'[<>:"/\\|?*()]', '_', name)
    # Remover espacios múltiples
    name = re.sub(r'\s+', '_', name)
    # Truncar si es muy largo (límite Windows ~260 chars total)
    # Dejamos max 60 chars por componente para evitar rutas largas
    if len(name) > 60:
        name = name[:60]
    return name

class Logger(object):
    """Redirige la salida 'print' a un archivo y a la consola."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def obtener_combinaciones_ecuaciones(file_panel, col_fondo, col_emisor, col_sector, col_exclusion):
    """Lee el archivo y obtiene todas las combinaciones únicas de Fondo, Emisor y Sector."""
    print("\n" + "="*60)
    print("--- IDENTIFICANDO TODAS LAS ECUACIONES POSIBLES ---")
    print("="*60)
    
    try:
        df = pd.read_excel(file_panel)
        print(f"✓ Archivo '{file_panel}' cargado. Dimensiones: {df.shape}")
        
        # Filtrar por dummy de exclusión si existe
        if col_exclusion and col_exclusion in df.columns:
            df = df[df[col_exclusion] == 0].copy()
            print(f"✓ Filtrado por '{col_exclusion}' = 0")
        
        # Obtener valores únicos
        fondos = sorted(df[col_fondo].unique())
        emisores = sorted(df[col_emisor].unique())
        sectores = sorted(df[col_sector].unique())
        
        print(f"\n📊 Valores únicos encontrados:")
        print(f"   • Fondos ({len(fondos)}): {fondos}")
        print(f"   • Emisores ({len(emisores)}): {emisores}")
        print(f"   • Sectores ({len(sectores)}): {sectores}")
        
        # Generar todas las combinaciones
        combinaciones = []
        for fondo, emisor, sector in product(fondos, emisores, sectores):
            filtro = {
                col_fondo: fondo,
                col_emisor: emisor,
                col_sector: sector
            }
            # Verificar que existan datos para esta combinación
            df_temp = df.copy()
            for col, val in filtro.items():
                df_temp = df_temp[df_temp[col] == val]
            
            if len(df_temp) > 0:
                combinaciones.append(filtro)
        
        print(f"\n✓ Total de ecuaciones válidas identificadas: {len(combinaciones)}")
        print("\n📋 Lista de ecuaciones:")
        for i, combo in enumerate(combinaciones, 1):
            print(f"   {i:2d}. Fondo={combo[col_fondo]}, Emisor={combo[col_emisor]}, Sector={combo[col_sector]}")
        
        return combinaciones
        
    except Exception as e:
        print(f"\n¡ERROR al identificar combinaciones! {e}")
        traceback.print_exc()
        return []

def load_and_prep_data(file_panel, file_ivs, file_ctrls, col_fecha, filtros_eq, var_y, col_exclusion):
    """Carga, filtra (para 1 ecuación), fusiona, y prepara datos OLS."""
    try:
        df_panel = pd.read_excel(file_panel)
        
        # --- FILTRO 1: DUMMY DE EXCLUSIÓN ---
        if col_exclusion and col_exclusion in df_panel.columns:
            df_panel = df_panel[df_panel[col_exclusion] == 0].copy()
        
        # --- FILTRO 2: FILTRAR PARA 1 ECUACIÓN ---
        df_eq = df_panel.copy()
        for col, valor in filtros_eq.items():
            df_eq = df_eq[df_eq[col] == valor]
        
        if df_eq.empty:
            return None, None, None, None
        
        if var_y not in df_eq.columns:
            return None, None, None, None

        df_ivs = pd.read_excel(file_ivs)
        df_ctrls = pd.read_excel(file_ctrls)
        
        for df in [df_eq, df_ivs, df_ctrls]:
            df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=False) 
            
    except Exception as e:
        print(f"\n¡ERROR durante la carga de datos! {e}")
        return None, None, None, None
    
    # Fusionar datos
    df = pd.merge(df_eq, df_ivs, on=col_fecha, how='inner')
    df = pd.merge(df, df_ctrls, on=col_fecha, how='inner')
    
    # Crear dummies de mes
    df['month'] = df[col_fecha].dt.month
    month_dummies = pd.get_dummies(df['month'], prefix='Mes', drop_first=True, dtype=int)
    df = pd.concat([df, month_dummies], axis=1)
    month_dummy_names = month_dummies.columns.tolist()
    
    # Centrar variables de control
    controls_c = []
    for col in VARS_CONTROL:
        col_c = f"{col}_c"
        df[col_c] = df[col] - df[col].mean()
        controls_c.append(col_c)
    
    # Crear interacciones de control con D_COVID
    control_interactions = []
    for col_c in controls_c:
        int_name = f"Int_{col_c}_COVID"
        df[int_name] = df[col_c] * df['D_COVID']
        control_interactions.append(int_name)
    
    all_vars_needed = [var_y] + VARS_IV_MOD + controls_c + control_interactions + month_dummy_names + [col_fecha]
    df = df.dropna(subset=all_vars_needed)
    
    # Índice de 1 nivel (Fecha) para OLS
    df = df.set_index(col_fecha)
    
    return df, controls_c, control_interactions, month_dummy_names

def run_control_break_test_ols(df, Y, IV_MOD, CONTROLS_C, CONTROL_INTS, MONTH_DUMMIES):
    """Ejecuta el Test F para la ruptura estructural en los CONTROLES usando OLS."""
    
    X_restringido_vars = VARS_IV_MOD + CONTROLS_C + MONTH_DUMMIES
    X_restringido = sm.add_constant(df[X_restringido_vars])
    
    X_no_restringido_vars = VARS_IV_MOD + CONTROLS_C + CONTROL_INTS + MONTH_DUMMIES
    X_no_restringido = sm.add_constant(df[X_no_restringido_vars])
    
    Y_var = df[Y]
    
    # Estimando modelos
    model_restringido = sm.OLS(Y_var, X_restringido).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    model_no_restringido = sm.OLS(Y_var, X_no_restringido).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    
    # Test F
    hipotesis_list = [f"{interaccion} = 0" for interaccion in CONTROL_INTS]
    
    try:
        f_test_result = model_no_restringido.f_test(hipotesis_list)
        
        f_stat = f_test_result.fvalue
        p_value = f_test_result.pvalue
        
        # CORRECCIÓN: usar df_num y df_denom en lugar de df_den
        df1 = f_test_result.df_num
        df2 = f_test_result.df_denom
        
        # Corrección para extraer el valor F si es una matriz
        if isinstance(f_stat, (np.ndarray, list)): 
            f_stat = f_stat[0][0] if f_stat.size > 0 else f_stat.item()
        if isinstance(p_value, (np.ndarray, list)): 
            p_value = p_value[0][0] if hasattr(p_value, 'size') and p_value.size > 0 else p_value.item() if hasattr(p_value, 'item') else p_value
        
        # Verificar si los valores son NaN (pueden ocurrir con datos muy limitados)
        if np.isnan(f_stat) or np.isnan(p_value):
            print(f"\n⚠️ ADVERTENCIA: Test F produjo valores NaN. Probablemente datos insuficientes o singularidad.")
            # En caso de NaN, por defecto usar modelo restringido (más conservador)
            resultado = "MODELO_RESTRINGIDO_NAN"
            return model_restringido, model_no_restringido, f_test_result, resultado, f_stat, p_value, df1, df2
        
        alpha = 0.05
        if p_value > alpha:
            resultado = "MODELO_RESTRINGIDO"
        else:
            resultado = "MODELO_NO_RESTRINGIDO"
        
        return model_restringido, model_no_restringido, f_test_result, resultado, f_stat, p_value, df1, df2
            
    except Exception as e:
        print(f"\n¡ERROR al ejecutar el F-test! {e}")
        traceback.print_exc()
        return model_restringido, model_no_restringido, None, None, None, None, None, None

def procesar_ecuacion(filtros_eq, ecuacion_num, total_ecuaciones, output_base_dir):
    """Procesa una ecuación individual y guarda sus resultados."""
    
    # Crear ID único para la ecuación (sanitizado para Windows)
    # Usar nombres más cortos para evitar rutas largas
    fondo_clean = sanitize_filename(str(filtros_eq[COL_FONDO]).replace('Tipo ', 'T'))
    emisor_clean = sanitize_filename(str(filtros_eq[COL_EMISOR]).replace('Emisores ', ''))
    sector_clean = sanitize_filename(str(filtros_eq[COL_SECTOR]))
    
    # ID corto para nombres de archivo
    ecuacion_id = f"F{fondo_clean}_{emisor_clean}_{sector_clean}"
    # Si aún es muy largo, usar hash
    if len(ecuacion_id) > 80:
        import hashlib
        hash_suffix = hashlib.md5(ecuacion_id.encode()).hexdigest()[:8]
        ecuacion_id = f"Eq{ecuacion_num:02d}_{hash_suffix}"
    
    print("\n" + "="*80)
    print(f"  PROCESANDO ECUACIÓN {ecuacion_num}/{total_ecuaciones}: {ecuacion_id}")
    print("="*80)
    print(f"  Fondo: {filtros_eq[COL_FONDO]}")
    print(f"  Emisor: {filtros_eq[COL_EMISOR]}")
    print(f"  Sector: {filtros_eq[COL_SECTOR]}")
    print("="*80)
    
    # Crear directorio para esta ecuación
    output_dir = os.path.join(output_base_dir, f"ecuacion_{ecuacion_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Archivo de log para esta ecuación
    log_file = os.path.join(output_dir, f'reporte_{ecuacion_id}.txt')
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            
            print("="*60)
            print(f" ECUACIÓN: {ecuacion_id}")
            print(f" Fecha: {datetime.datetime.now()}")
            print("="*60 + "\n")
            
            # Cargar y preparar datos
            df_panel, controls_c, control_ints, month_dummies = load_and_prep_data(
                FILE_PANEL_3, FILE_PREDICTORES, FILE_CONTROLES,
                COL_FECHA, filtros_eq, VAR_Y_DEPENDIENTE, COL_DUMMY_EXCLUSION
            )
            
            if df_panel is None or len(df_panel) < 30:
                print(f"⚠️ ADVERTENCIA: Datos insuficientes para esta ecuación (n={len(df_panel) if df_panel is not None else 0})")
                sys.stdout = original_stdout
                return {
                    'ecuacion_id': ecuacion_id,
                    'fondo': filtros_eq[COL_FONDO],
                    'emisor': filtros_eq[COL_EMISOR],
                    'sector': filtros_eq[COL_SECTOR],
                    'n_obs': len(df_panel) if df_panel is not None else 0,
                    'resultado': 'DATOS_INSUFICIENTES',
                    'f_stat': None,
                    'p_value': None
                }
            
            # Ejecutar test
            result = run_control_break_test_ols(
                df_panel, VAR_Y_DEPENDIENTE, VARS_IV_MOD, controls_c, 
                control_ints, month_dummies
            )
            
            # Desempaquetar resultado
            if len(result) == 8:
                m_res, m_no_res, test, resultado, f_stat, p_value, df1, df2 = result
            else:
                m_res, m_no_res, test, resultado = result
                f_stat, p_value, df1, df2 = None, None, None, None
            
            sys.stdout = original_stdout
            
            # Guardar resúmenes
            if m_res:
                with open(os.path.join(output_dir, f'summary_restringido.txt'), 'w', encoding='utf-8') as f_res:
                    f_res.write(m_res.summary().as_text())
                with open(os.path.join(output_dir, f'summary_no_restringido.txt'), 'w', encoding='utf-8') as f_nores:
                    f_nores.write(m_no_res.summary().as_text())
                
                if test is not None and f_stat is not None:
                    with open(os.path.join(output_dir, 'RESULTADO.txt'), 'w', encoding='utf-8') as f_rec:
                        f_rec.write("="*60 + "\n")
                        f_rec.write(f"ECUACIÓN: {ecuacion_id}\n")
                        f_rec.write(f"Fondo: {filtros_eq[COL_FONDO]}\n")
                        f_rec.write(f"Emisor: {filtros_eq[COL_EMISOR]}\n")
                        f_rec.write(f"Sector: {filtros_eq[COL_SECTOR]}\n")
                        f_rec.write("="*60 + "\n\n")
                        f_rec.write(f"N observaciones: {len(df_panel)}\n")
                        f_rec.write(f"Estadístico F: {f_stat:.4f}\n")
                        f_rec.write(f"p-valor: {p_value:.4f}\n")
                        f_rec.write(f"Grados de libertad: ({int(df1)}, {int(df2)})\n\n")
                        f_rec.write(f"MODELO RECOMENDADO: {resultado}\n")
                    
                    print(f"  ✓ Ecuación {ecuacion_num} completada: {resultado} (F={f_stat:.2f}, p={p_value:.4f})")
                    
                    return {
                        'ecuacion_id': ecuacion_id,
                        'fondo': filtros_eq[COL_FONDO],
                        'emisor': filtros_eq[COL_EMISOR],
                        'sector': filtros_eq[COL_SECTOR],
                        'n_obs': len(df_panel),
                        'resultado': resultado,
                        'f_stat': f_stat,
                        'p_value': p_value
                    }
                else:
                    print(f"  ⚠️ Ecuación {ecuacion_num}: Test fallido")
                    return {
                        'ecuacion_id': ecuacion_id,
                        'fondo': filtros_eq[COL_FONDO],
                        'emisor': filtros_eq[COL_EMISOR],
                        'sector': filtros_eq[COL_SECTOR],
                        'n_obs': len(df_panel),
                        'resultado': 'TEST_FALLIDO',
                        'f_stat': None,
                        'p_value': None
                    }
            
            return None
            
    except Exception as e:
        print(f"  ✗ ERROR en ecuación {ecuacion_num}: {e}")
        traceback.print_exc()
        return None

def main():
    """Función principal que ejecuta todas las ecuaciones."""
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_base_dir = f"resultados_panel3_completo_{timestamp}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("  ANÁLISIS AUTOMATIZADO - PANEL 3 (PORTAFOLIO)")
    print(f"  Fecha de inicio: {datetime.datetime.now()}")
    print("="*80)
    
    # Obtener todas las combinaciones
    combinaciones = obtener_combinaciones_ecuaciones(
        FILE_PANEL_3, COL_FONDO, COL_EMISOR, COL_SECTOR, COL_DUMMY_EXCLUSION
    )
    
    if not combinaciones:
        print("\n¡ERROR! No se encontraron combinaciones válidas.")
        return
    
    print(f"\n🚀 Iniciando procesamiento de {len(combinaciones)} ecuaciones...\n")
    
    # Procesar cada ecuación
    resultados = []
    for i, filtros in enumerate(combinaciones, 1):
        resultado = procesar_ecuacion(filtros, i, len(combinaciones), output_base_dir)
        if resultado:
            resultados.append(resultado)
    
    # Crear resumen consolidado
    print("\n" + "="*80)
    print("  GENERANDO RESUMEN CONSOLIDADO")
    print("="*80)
    
    if len(resultados) == 0:
        print("\n⚠️ No se procesaron ecuaciones exitosamente.")
        return
    
    df_resumen = pd.DataFrame(resultados)
    
    # Guardar resumen en Excel
    resumen_file = os.path.join(output_base_dir, 'RESUMEN_TODAS_ECUACIONES.xlsx')
    df_resumen.to_excel(resumen_file, index=False)
    
    # Guardar resumen en texto
    resumen_txt = os.path.join(output_base_dir, 'RESUMEN_TODAS_ECUACIONES.txt')
    with open(resumen_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN DE TODAS LAS ECUACIONES - PANEL 3\n")
        f.write(f"Fecha: {datetime.datetime.now()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total de ecuaciones procesadas: {len(resultados)}\n\n")
        
        # Conteo por resultado
        conteo = df_resumen['resultado'].value_counts()
        f.write("DISTRIBUCIÓN DE RESULTADOS:\n")
        f.write("-" * 40 + "\n")
        for modelo, count in conteo.items():
            pct = (count / len(resultados)) * 100
            f.write(f"  {modelo}: {count} ({pct:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETALLE POR ECUACIÓN:\n")
        f.write("="*80 + "\n\n")
        
        for _, row in df_resumen.iterrows():
            f.write(f"Ecuación: {row['ecuacion_id']}\n")
            f.write(f"  Fondo: {row['fondo']}\n")
            f.write(f"  Emisor: {row['emisor']}\n")
            f.write(f"  Sector: {row['sector']}\n")
            f.write(f"  N obs: {row['n_obs']}\n")
            f.write(f"  Resultado: {row['resultado']}\n")
            if pd.notna(row['f_stat']):
                f.write(f"  F-stat: {row['f_stat']:.4f}, p-value: {row['p_value']:.4f}\n")
            f.write("\n")
    
    print(f"\n✅ ¡PROCESO COMPLETADO!")
    print(f"\n📁 Resultados guardados en: {os.path.abspath(output_base_dir)}")
    print(f"\n📊 Resumen:")
    print(f"   • Total ecuaciones procesadas: {len(resultados)}")
    print(f"\n   Distribución de modelos recomendados:")
    conteo = df_resumen['resultado'].value_counts()
    for modelo, count in conteo.items():
        print(f"      - {modelo}: {count}")
    
    print(f"\n📄 Archivos principales:")
    print(f"   • RESUMEN_TODAS_ECUACIONES.xlsx")
    print(f"   • RESUMEN_TODAS_ECUACIONES.txt")
    print(f"   • ecuacion_[ID]/ (una carpeta por cada ecuación)")

if __name__ == "__main__":
    main()