import pandas as pd
import numpy as np 
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import os
import datetime
import sys
from scipy import stats

# --- CONFIGURACIÓN PRINCIPAL: PANEL 2 (REASIGNACIÓN) ---

# 1. ARCHIVOS DE ENTRADA
FILE_PANEL_2 = 'panel_2_reasignacion.xlsx'
FILE_PREDICTORES = 'dataset_final_interacciones.xlsx'
FILE_CONTROLES = 'variables_control_final.xlsx'

# 2. NOMBRES DE COLUMNAS
COL_FECHA = 'Fecha' 
COL_AFP = 'AFP'
COL_FONDO = 'TipodeFondo'
VAR_Y_DEPENDIENTE = 'Variacion_Neta_Afiliados'

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
# ---------------------------------

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

def load_and_prep_data(file_panel, file_ivs, file_ctrls, col_fecha, col_afp, col_fondo, var_y, col_exclusion):
    """Carga, filtra, fusiona, crea dummies de mes, centra controles y crea interacciones."""
    print("="*60)
    print("--- 1. Carga y Preparación de Datos (Panel 2) ---")
    print("="*60)
    try:
        df_panel = pd.read_excel(file_panel)
        print(f"✓ Archivo de panel '{file_panel}' cargado.")
        print(f"  Dimensiones iniciales: {df_panel.shape}")
        
        # SOLUCIÓN: Crear entidad compuesta AFP_Fondo
        print(f"\n📌 Creando entidad compuesta: {col_afp} + {col_fondo}...")
        df_panel['Entidad_Compuesta'] = (
            df_panel[col_afp].astype(str) + '_F' + df_panel[col_fondo].astype(str)
        )
        print(f"✓ Entidad compuesta creada.")
        print(f"  Ejemplo de entidades: {df_panel['Entidad_Compuesta'].unique()[:5].tolist()}")
        print(f"  Total de entidades únicas: {df_panel['Entidad_Compuesta'].nunique()}")
        
        if col_exclusion:
            if col_exclusion in df_panel.columns:
                obs_iniciales = len(df_panel)
                df_panel = df_panel[df_panel[col_exclusion] == 0].copy()
                obs_filtradas = obs_iniciales - len(df_panel)
                print(f"\n✓ FILTRO APLICADO: {obs_filtradas} observaciones atípicas eliminadas usando '{col_exclusion}'.")
                print(f"  Observaciones restantes: {len(df_panel)}")
            else:
                print(f"\nADVERTENCIA: Columna de exclusión '{col_exclusion}' no encontrada. Continuando sin filtrar.")

        if var_y not in df_panel.columns:
            print(f"\n¡ERROR! Variable dependiente '{var_y}' no se encuentra en '{file_panel}'.")
            return None, None, None, None

        df_ivs = pd.read_excel(file_ivs)
        df_ctrls = pd.read_excel(file_ctrls)
        
        for df in [df_panel, df_ivs, df_ctrls]:
            df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=False) 
            
    except FileNotFoundError as e:
        print(f"\n¡ERROR! No se encontró el archivo: {e.filename}")
        return None, None, None, None
    except Exception as e:
        print(f"\n¡ERROR durante la carga de datos! {e}")
        return None, None, None, None
    
    df_tiempo = pd.merge(df_ivs, df_ctrls, on=col_fecha, how='inner')
    df = pd.merge(df_panel, df_tiempo, on=col_fecha, how='inner')
    
    print(f"\n✓ Datos cargados y fusionados. Total observaciones: {len(df)}")

    print("\n📅 Creando dummies de mes (estacionalidad)...")
    df['month'] = df[col_fecha].dt.month
    month_dummies = pd.get_dummies(df['month'], prefix='Mes', drop_first=True, dtype=int)
    df = pd.concat([df, month_dummies], axis=1)
    month_dummy_names = month_dummies.columns.tolist() 
    print(f"✓ Dummies de estacionalidad creadas: {len(month_dummy_names)} variables")
    
    print("\n📊 Centrando variables de control...")
    controls_c = []
    for col in VARS_CONTROL:
        col_c = f"{col}_c"
        df[col_c] = df[col] - df[col].mean()
        controls_c.append(col_c)
    print(f"✓ {len(controls_c)} variables centradas")
        
    print("\n🔗 Creando interacciones de control con D_COVID...")
    control_interactions = []
    for col_c in controls_c:
        int_name = f"Int_{col_c}_COVID"
        df[int_name] = df[col_c] * df['D_COVID']
        control_interactions.append(int_name)
    print(f"✓ {len(control_interactions)} interacciones creadas")
        
    # Usar la entidad compuesta para el índice de panel
    all_vars_needed = [var_y] + VARS_IV_MOD + controls_c + control_interactions + month_dummy_names + ['Entidad_Compuesta', col_fecha]
    df = df.dropna(subset=all_vars_needed)
    
    print(f"\n🧹 Eliminando valores faltantes...")
    print(f"✓ Observaciones finales: {len(df)}")
    
    # CRÍTICO: Índice de 2 niveles (Entidad_Compuesta, Fecha)
    df = df.set_index(['Entidad_Compuesta', col_fecha])
    
    print(f"\n✅ Datos listos para PanelOLS:")
    print(f"   • Observaciones totales: {len(df)}")
    print(f"   • Entidades únicas: {df.index.get_level_values(0).nunique()}")
    print(f"   • Períodos temporales: {df.index.get_level_values(1).nunique()}")
    print(f"   • MultiIndex niveles: {df.index.nlevels} (✓ Compatible con PanelOLS)")
    
    return df, controls_c, control_interactions, month_dummy_names

def run_control_break_test(df, Y, IV_MOD, CONTROLS_C, CONTROL_INTS, MONTH_DUMMIES):
    """Ejecuta el Test F para la ruptura estructural en los CONTROLES."""
    print("\n" + "="*60)
    print("--- 2. Test F de Ruptura Estructural (Controles) ---")
    print("="*60)
    
    X_restringido_vars = VARS_IV_MOD + CONTROLS_C + MONTH_DUMMIES
    X_restringido = sm.add_constant(df[X_restringido_vars])
    
    X_no_restringido_vars = VARS_IV_MOD + CONTROLS_C + CONTROL_INTS + MONTH_DUMMIES
    X_no_restringido = sm.add_constant(df[X_no_restringido_vars])
    
    Y_var = df[Y]
    
    print("\n🔄 Estimando Modelo RESTRINGIDO (H0)...")
    model_restringido = PanelOLS(Y_var, X_restringido, 
                                 entity_effects=True
                                 ).fit(cov_type='robust')
    print("   ✓ Modelo restringido estimado")
    
    print("\n🔄 Estimando Modelo NO RESTRINGIDO (HA)...")
    model_no_restringido = PanelOLS(Y_var, X_no_restringido, 
                                    entity_effects=True
                                    ).fit(cov_type='robust')
    print("   ✓ Modelo no restringido estimado")
    
    print("\n" + "="*60)
    print("RESUMEN DEL MODELO NO RESTRINGIDO (COMPLETO)")
    print("="*60)
    print(model_no_restringido.summary)

    print("\n" + "="*60)
    print("--- 3. Prueba de Wald (Test F) de Significancia Conjunta ---")
    print("="*60)
    
    hipotesis_list = [f"{interaccion} = 0" for interaccion in CONTROL_INTS]
    print(f"\nHipótesis Nula (H0): {', '.join(hipotesis_list)}")
    print(f"\n📋 Variables siendo probadas simultáneamente:")
    for i, var in enumerate(CONTROL_INTS, 1):
        print(f"   {i}. {var}")
    
    try:
        # Crear matriz de restricciones
        param_names = list(model_no_restringido.params.index)
        num_params = len(param_names)
        num_restrictions = len(CONTROL_INTS)
        
        R = np.zeros((num_restrictions, num_params))
        for i, var_name in enumerate(CONTROL_INTS):
            if var_name in param_names:
                j = param_names.index(var_name)
                R[i, j] = 1.0
        
        print(f"\n✓ Matriz de restricciones creada correctamente.")
        print(f"  Dimensión: {R.shape} ({num_restrictions} restricciones × {num_params} parámetros)")
        
        wald_result = model_no_restringido.wald_test(R)
        
        f_stat = wald_result.stat
        p_value = wald_result.pval
        df1 = wald_result.df
        df2 = wald_result.df_denom
        
        print("\n" + "="*60)
        print("--- 4. RESULTADOS DEL TEST DE WALD ---")
        print("="*60)
        print(f"  Estadístico F (Wald):  {f_stat:.4f}")
        print(f"  Grados de libertad:    ({df1}, {df2})")
        print(f"  p-valor:               {p_value:.4f}")
        print("="*60)
        
        print("\n" + "="*60)
        print("--- 5. VEREDICTO METODOLÓGICO (Panel 2: Reasignación) ---")
        print("="*60)
        
        alpha = 0.05
        if p_value > alpha:
            print(f"\n✓ RESULTADO: NO SE RECHAZA H0 (p-valor = {p_value:.4f} > α = {alpha})")
            print("\n📊 INTERPRETACIÓN:")
            print("  • Las interacciones de los controles con COVID NO son")
            print("    conjuntamente significativas.")
            print("  • NO hay evidencia estadística de ruptura estructural")
            print("    en las variables de control durante el periodo COVID.")
            print("\n💡 RECOMENDACIÓN METODOLÓGICA:")
            print("  ➜ Usar el **MODELO RESTRINGIDO (Simple)** para el Panel 2.")
            print("  ➜ Este modelo es más parsimonioso y eficiente.")
            print("\n📈 COMPARACIÓN DE MODELOS:")
            print(f"  • Modelo Restringido:    R² Within = {model_restringido.rsquared_within:.4f}, {len(X_restringido_vars)} variables")
            print(f"  • Modelo No Restringido: R² Within = {model_no_restringido.rsquared_within:.4f}, {len(X_no_restringido_vars)} variables")
            print(f"  • Mejora en R²: {(model_no_restringido.rsquared_within - model_restringido.rsquared_within):.4f} (no significativa)")
            resultado = "MODELO_RESTRINGIDO"
        else:
            print(f"\n✗ RESULTADO: SE RECHAZA H0 (p-valor = {p_value:.4f} ≤ α = {alpha})")
            print("\n📊 INTERPRETACIÓN:")
            print("  • Las interacciones de los controles con COVID SÍ son")
            print("    conjuntamente significativas.")
            print("  • HAY evidencia estadística de ruptura estructural")
            print("    en las variables de control durante el periodo COVID.")
            print("\n💡 RECOMENDACIÓN METODOLÓGICA:")
            print("  ➜ Usar el **MODELO NO RESTRINGIDO (Completo)** para el Panel 2.")
            print("  ➜ Este modelo captura mejor la heterogeneidad estructural.")
            print("\n📈 COMPARACIÓN DE MODELOS:")
            print(f"  • Modelo Restringido:    R² Within = {model_restringido.rsquared_within:.4f}, {len(X_restringido_vars)} variables")
            print(f"  • Modelo No Restringido: R² Within = {model_no_restringido.rsquared_within:.4f}, {len(X_no_restringido_vars)} variables")
            print(f"  • Mejora en R²: {(model_no_restringido.rsquared_within - model_restringido.rsquared_within):.4f} (significativa)")
            resultado = "MODELO_NO_RESTRINGIDO"
        
        print("="*60)
        
        print("\n" + "="*60)
        print("--- 6. COEFICIENTES INDIVIDUALES DE LAS INTERACCIONES ---")
        print("="*60)
        for var in CONTROL_INTS:
            coef = model_no_restringido.params[var]
            se = model_no_restringido.std_errors[var]
            tstat = model_no_restringido.tstats[var]
            pval = model_no_restringido.pvalues[var]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            print(f"  {var:45s}: β={coef:7.4f} (SE={se:.4f}, t={tstat:6.3f}, p={pval:.4f}) {sig}")
        print("\n  Nota: *** p<0.01, ** p<0.05, * p<0.10")
        print("="*60)
        
        return model_restringido, model_no_restringido, wald_result, resultado
            
    except Exception as e:
        print(f"\n¡ERROR al ejecutar el Wald-test!")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje: {e}")
        import traceback
        print("\nTraceback completo:")
        print(traceback.format_exc())
        return model_restringido, model_no_restringido, None, None

def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = f"test_panel_2_reasignacion_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    output_log_file = os.path.join(output_dir, 'reporte_test_F_panel_2.txt')
    
    original_stdout = sys.stdout
    with open(output_log_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        print("="*60)
        print(" INICIO DEL TEST DE RUPTURA ESTRUCTURAL")
        print(" PANEL 2: REASIGNACIÓN (AFP × TIPO DE FONDO)")
        print(f" Fecha: {datetime.datetime.now()}")
        print("="*60 + "\n")
        
        df_panel, controls_c, control_ints, month_dummies = load_and_prep_data(
            FILE_PANEL_2, FILE_PREDICTORES, FILE_CONTROLES,
            COL_FECHA, COL_AFP, COL_FONDO, VAR_Y_DEPENDIENTE, COL_DUMMY_EXCLUSION
        )
        
        if df_panel is not None:
            m_res, m_no_res, test, resultado = run_control_break_test(
                df_panel, VAR_Y_DEPENDIENTE, VARS_IV_MOD, controls_c, 
                control_ints, month_dummies
            )
            
            if m_res:
                # Guardar resúmenes de modelos
                with open(os.path.join(output_dir, 'summary_restringido_P2.txt'), 'w', encoding='utf-8') as f_res:
                    f_res.write(m_res.summary.as_text())
                with open(os.path.join(output_dir, 'summary_no_restringido_P2.txt'), 'w', encoding='utf-8') as f_nores:
                    f_nores.write(m_no_res.summary.as_text())
                
                # Guardar resultado del test
                if test is not None:
                    with open(os.path.join(output_dir, 'resultado_Wald_test_P2.txt'), 'w', encoding='utf-8') as f_test_out:
                        f_test_out.write("="*60 + "\n")
                        f_test_out.write("RESULTADO DEL TEST DE WALD (RUPTURA ESTRUCTURAL)\n")
                        f_test_out.write("Panel 2: Reasignación (AFP × Tipo de Fondo)\n")
                        f_test_out.write("="*60 + "\n\n")
                        f_test_out.write(f"Estadístico F: {test.stat:.4f}\n")
                        f_test_out.write(f"p-valor: {test.pval:.4f}\n")
                        f_test_out.write(f"Grados de libertad: ({test.df}, {test.df_denom})\n\n")
                        f_test_out.write(f"Recomendación: {resultado}\n")
                        f_test_out.write("\n" + str(test))
                    
                    # Guardar recomendación en archivo separado
                    with open(os.path.join(output_dir, 'RECOMENDACION_MODELO.txt'), 'w', encoding='utf-8') as f_rec:
                        f_rec.write("="*60 + "\n")
                        f_rec.write("RECOMENDACIÓN FINAL PARA PANEL 2 (REASIGNACIÓN)\n")
                        f_rec.write("="*60 + "\n\n")
                        f_rec.write(f"MODELO RECOMENDADO: {resultado}\n\n")
                        f_rec.write(f"Estadístico F del Test de Wald: {test.stat:.4f}\n")
                        f_rec.write(f"p-valor: {test.pval:.4f}\n")
                        f_rec.write(f"\nInterpretación: {'Las interacciones NO son significativas' if test.pval > 0.05 else 'Las interacciones SÍ son significativas'}\n")
        else:
            print("La preparación de datos falló. El análisis se detuvo.")

        print("\n" + "="*60)
        print(" ANÁLISIS COMPLETADO")
        print(f" Resultados guardados en: {os.path.abspath(output_dir)}")
        print("="*60)
    
    sys.stdout = original_stdout
    print(f"\n✅ ¡Análisis del Panel 2 completado exitosamente!")
    print(f"📁 Revisa la carpeta: {output_dir}")
    print(f"\n📄 Archivos generados:")
    print(f"   • reporte_test_F_panel_2.txt (log completo)")
    print(f"   • summary_restringido_P2.txt")
    print(f"   • summary_no_restringido_P2.txt")
    print(f"   • resultado_Wald_test_P2.txt")
    print(f"   • RECOMENDACION_MODELO.txt")

if __name__ == "__main__":
    main()