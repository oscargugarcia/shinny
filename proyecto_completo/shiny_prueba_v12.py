# Importar librerías 
from ipyleaflet import Map, Marker, LayerGroup, Popup, AwesomeIcon, GeoData
from shiny.express import ui, input, render 
from shiny import reactive, session
from shinywidgets import render_widget
import ipywidgets as widgets
import pandas as pd
import geopandas as gpd 
import numpy as np
import regex as re 
import geopy
from shapely.geometry import Polygon, MultiPolygon, Point 
import shapely
import json
from chatlas import ChatOpenAI
from smolagents import CodeAgent, InferenceClientModel, tool
from smolagents.models import OpenAIModel
import pyarrow as pa
import pyarrow.parquet as pq
import h3
import os

# Funciones
from Funcs import load_admin_shapefiles, extract_data, get_hex_pols, group_results, iso_wrapper, crear_equidistancia, crear_isocronas, swap_coordinates

# Globales
MAIN_DIR = "" # Directorio a la carpeta proyecto_completo
INPUT_PARAMS_DIR = os.path.join(MAIN_DIR, "input/parametros")
INPUT_DATA_DIR = os.path.join(MAIN_DIR, "input/datos")
MUNIP_SHP_DIR = os.path.join(INPUT_DATA_DIR, "municipios/MGN_ANM_MPIOS.shp")
OUTPUT_MAP_DIR = os.path.join(MAIN_DIR, "output/mapas")
OUTPUT_DATA_DIR = os.path.join(MAIN_DIR, "output/datos")
MAIN_DICT_RAW = os.path.join(INPUT_PARAMS_DIR, "dic_variables_dimensiones_todas.xlsx")
PARQUET_FILE = "parquet_all_vars.parquet"
HEX_RES = 10

# Diccionarios de indicadores y dimensiones
main_dict_df = pd.read_excel(MAIN_DICT_RAW)
main_dict_content = main_dict_df.to_string(index=False)

with open(os.path.join(INPUT_PARAMS_DIR, "dimension_variables.json")) as f:
    RENAME_DICT = json.load(f)

with open(os.path.join(INPUT_PARAMS_DIR, "nombres_variables.json")) as f:
    INDICATORS_DICT = json.load(f)


# Llaves de API
API_MB = ""
API_ORS = ""
API_OPENAI = ""
API_D = {"isocrona": API_MB, "equidistancia": API_ORS}

# Diccionarios --- Estos deberían ser generados y cargados
muni_code_dict = {"Cartagena": "13001", "Medellín": "05001", "Bogotá": "11001"}
inds_code_dict = RENAME_DICT
 
# Directorios
os.chdir(MAIN_DIR)

# Cargar datos municipales
munip_shp = load_admin_shapefiles(admin_shp_path=MUNIP_SHP_DIR)


# ==================================================  Inicializar eventos de clickeo en el mapa ======================================
clicked_coords = reactive.Value(None)

# Inicializar marcador de direcciones candidatas
cand_dir_check = reactive.Value(None) 

# Inicializar indicador de procesamiento de isocronas o equidistancias
run_iso_bool = reactive.Value(False)

# Inicializar indicador de mapeo de iscronas o equidistancias
plot_iso_bool = reactive.Value(False)

# Inicializar valores relacionados con el agente
ai_processing = reactive.Value(False)
ai_selected_indicators = reactive.Value([])
ai_summary_ready = reactive.Value(False)
final_ai_summary = reactive.Value("")
chat_messages = reactive.Value("")

# ============================================== Interfaz de usuario =========================================================
ui.page_opts(title="Prueba concepto", fillable=True, resizable=False)
# Replace your CSS line with both CSS and JavaScript
ui.tags.style("""
.shiny-file-input * {
    color: transparent !important;
}
.shiny-file-input *:not(.btn):not(.form-control) {
    display: none !important;
}
.shiny-file-input .btn {
    color: #333 !important;
}
.shiny-file-input label {
    color: #333 !important;
    display: block !important;
}
""")

with ui.sidebar(resizable=False):
    # Keep only the input controls in the sidebar
    # Municipios
    ui.input_select(id="muni_selector", label="Seleccionar municipio", choices=list(muni_code_dict.keys()))
    
    # Replace your current file upload section (lines ~561-595) with this:

    ui.card_header("📁 Cargar Archivos")
    with ui.div(style="margin-bottom: 15px;"):
        # Single file input that accepts multiple files
        ui.input_file(
            id="files_upload", 
            label="Seleccionar archivos (puedes elegir múltiples):", 
            accept=[".csv", ".xlsx", ".xls", ".parquet", ".json"],
            multiple=True,  # This allows multiple file selection
            width="100%"
        )

        @render.ui
        def file_status_display():
            files = input.files_upload()
            
            if not files:
                return ui.div(
                    ui.p("📂 No hay archivos seleccionados", 
                        style="font-size: 12px; color: #666; margin: 5px 0;"),
                    ui.p("💡 Selecciona los archivos que necesites para el análisis", 
                        style="font-size: 10px; color: #888; font-style: italic; margin: 5px 0;")
                )
            
            # Display file list with validation
            file_list = ""
            valid_files = 0
            for i, file_info in enumerate(files, 1):
                file_name = file_info['name']
                file_size = round(file_info['size'] / 1024, 1)  # Size in KB
                file_ext = file_name.split('.')[-1].lower()
                
                # Check if file type is valid
                if file_ext in ['csv', 'xlsx', 'xls', 'parquet', 'json']:
                    file_list += f"✅ Archivo {i}: {file_name} ({file_size} KB)\n"
                    valid_files += 1
                else:
                    file_list += f"❌ Archivo {i}: {file_name} (tipo no soportado)\n"
            
            button_disabled = valid_files == 0
            button_style = (
                "width: 100%; background-color: #6c757d; color: white; margin-top: 5px;" 
                if button_disabled 
                else "width: 100%; background-color: #28a745; color: white; margin-top: 5px;"
            )
            
            return ui.div(
                ui.pre(file_list, style="font-size: 11px; margin: 5px 0; color: #333; background-color: #f8f9fa; padding: 8px; border-radius: 4px;"),
                # ui.input_action_button(
                #     "confirm_files", 
                #     f"Confirmar {valid_files} archivo{'s' if valid_files != 1 else ''} válido{'s' if valid_files != 1 else ''}", 
                #     style=button_style,
                #     disabled=button_disabled
                # ) if files else None

            )


    # # Dirección
    # ui.input_text(id="address_input",label="Ingrese una dirección o seleccione una coordenada en el mapa")

    # # Botón de geocodificación
    # ui.input_action_button("geo_button", "Geocodificar")
    # Conditional address input/coordinate display
    @render.ui
    def conditional_address_input():
        # Check if user has clicked on the map
        clicked = clicked_coords.get()
        if clicked:
            # If user clicked on map, show message instead of address input
            lat, lon = clicked[0], clicked[1]
            return ui.div(
                ui.p(f"📍 Ubicación seleccionada: ({lat:.4f}, {lon:.4f})", 
                    style="margin: 10px 0; padding: 8px; background-color: #e8f5e9; border-radius: 4px; font-weight: bold;"),
                ui.input_action_button("clear_coords", "Cambiar ubicación", 
                    style="margin-top: 5px; background-color: #6c757d; color: white; width: 100%;")
            )
        else:
            # Show normal address input and geocoding button
            return ui.div(
                ui.input_text(id="address_input", label="Ingrese una dirección o seleccione una coordenada en el mapa"),
                ui.input_action_button("geo_button", "Geocodificar", style="margin-top: 5px; width: 100%;")
            )

    # Coordenadas candidatas
    @render.ui
    def conditional_candidate_selector():
        candidates = dir_to_coords()
        if candidates:
            return ui.input_select(
                id="candidate_selector", 
                label="Selecciona una dirección candidata:", 
                choices=list(candidates.keys()), 
                selected=list(candidates.keys())[0] if candidates else "None"
            )
        else:
            return None

    # Área
    ui.input_select(id="area_selector", label="¿Cómo quiere delimitar el área de interés?", choices=["", "Isocrona", "Equidistancia"], selected="")
    
    # Conditional transport selector - only show for Isocrona
    @render.ui
    def conditional_transport_selector():
        area_type = input.area_selector()
        if area_type == "Isocrona":
            return ui.input_select(
                id="trans_selector", 
                label="Seleccionar modo de transporte", 
                choices=["Vehículo", "A pie"], 
                selected="Vehículo"
            )
        else:
            return None

    # Dynamic interval selector with changing label
    @render.ui
    def dynamic_interval_selector():
        area_type = input.area_selector()
        if area_type == "Isocrona":
            label_text = "Ingrese el tiempo en minutos"
            return ui.input_numeric(
            id="interval_selector", 
            label=label_text, 
            value=0
            )

        elif area_type == "Equidistancia":
            label_text = "Ingrese la distancia en metros"
            return ui.input_numeric(
            id="interval_selector", 
            label=label_text, 
            value=0
            )
        else:
           return None

    # Modo de transporte
    #ui.input_select(id="trans_selector", label="Seleccionar modo de transporte", choices=["Vehículo", "A pie"], selected="None")
    
    # Intervalo
    #ui.input_numeric(id="interval_selector", label="Seleccione el tiempo (minutos) o la distancia (metros)", value=0)

    # Botón para fijar el área
    # Conditional "Crear área" button - only show when all required inputs are ready
    @render.ui
    def conditional_create_area_button():
        area_type = input.area_selector()
        intervalo = input.interval_selector()
        if (area_type == "Isocrona" or area_type == "Equidistancia") and (intervalo and intervalo > 0):
            # If all conditions are met, show the button
            return ui.input_action_button(
                "create_area_button", 
                "Crear área", 
                style="width: 100%; margin-top: 10px; background-color: #28a745; color: white;"
            )
        else:
            return None

    # @render.ui
    # def conditional_candidate_selector():
    #     # Check if user has clicked on the map
    #     clicked = clicked_coords.get()
    #     if clicked:
    #         # If user clicked on map, show coordinates instead of selector
    #         lat, lon = clicked[0], clicked[1]
    #         return ui.div(
    #             ui.p(f"📍 Coordenadas seleccionadas: ({lat:.4f}, {lon:.4f})", 
    #                 style="margin: 10px 0; padding: 8px; background-color: #e8f5e9; border-radius: 4px; font-weight: bold;")
    #         )
    
    #     # Otherwise, show candidate selector if we have geocoded candidates
    #     candidates = dir_to_coords()
    #     if candidates:
    #         return ui.input_select(
    #             id="candidate_selector", 
    #             label="Selecciona una dirección candidata:", 
    #             choices=list(candidates.keys()), 
    #             selected=list(candidates.keys())[0] if candidates else "None"
    #         )
    #     else:
    #         return None

# Main content area with side-by-side layout
with ui.layout_columns(col_widths=[6, 6], height="65vh"):  # Split 50-50, 65% of viewport height
    # Left panel - AI Response Display
    with ui.card(fill=True):
        ui.card_header("📊 Resultados del Análisis")
        
        # AI status and summary display with better styling
        with ui.div(style="padding: 15px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;"):
            @render.text
            def ai_status():
                if ai_processing.get():
                    return "🤖 Procesando consulta..."
    
                # Check if we have a final summary ready
                summary = final_ai_summary.get()
                if summary and summary != "":
                    return summary
    
                elif plot_iso_bool.get() and not run_iso_bool.get():
                    # Area created but no query sent yet
                    return "📍 Área creada. Envía una consulta para obtener información sobre esta zona."
    
                elif run_iso_bool.get():
                    geometry_ready = create_isochrone_geometry()
                    geometry_success = isinstance(geometry_ready, gpd.GeoDataFrame)
    
                    if geometry_success:
                        return "📊 Área lista para consultas. Envía tu pregunta para obtener información detallada."
                    else:
                        return "📍 Creando área de interés..."
                else:
                    return "💡 Crea una zona de interés y envía una consulta para obtener información sobre ella."


    # Right panel - Map
    with ui.card(fill=True):
        ui.card_header("🗺️ Mapa")
        @render_widget  
        def map():
            m = create_map()
            return m

# Chat input section below the main panels
with ui.card(height="30vh", fill=True):
    ui.card_header("🤖 Asistente de Consultas")
    
    # Chat input and button
    with ui.layout_columns(col_widths=[8, 4]):
        ui.input_text_area(
            "chat_input", 
            "Escribe tu consulta:", 
            placeholder="Ej: Quiero información demográfica de esta área", 
            width="100%", 
            height="80px"
        )
        with ui.div(style="margin-top: 30px;"):
            ui.input_action_button(
                "send_query", 
                "Enviar consulta", 
                width="100%", 
                style="height: 80px;"
                )
    
    # Chat history (conversation log)
    with ui.div():        
        @render.text
        def chat_history_display():
            return chat_messages.get()
    
# ====================================================== Procesos ===============================================================
# ------------------------------------------------- Funciones estáticas ---------------------------------------------------------
# 1) Función para extraer los datos de los hexágonos
def filtered_hexagons():

    # Obtener el código del municipio
    filter_muni_name = input.muni_selector()
    try:
        filter_muni_code = muni_code_dict.get(filter_muni_name)
    except Exception as e:
        print(f"Problema al extraer el código municipal: {e}")
        return
    
    # Extraer los indicadores de la respuesta del agente de etiquetado
    inds_ai = ai_selected_indicators.get()
    inds = [ind for ind in inds_ai if ind != "" and ind!= "None" and ind is not None]
    #inds = inds_code_dict.get(inds_name)
    print(f"Indicadores en filtered_hexagons: {inds}")
    if not inds: 
        return 
    else: 
        inds = list(set(inds))

    # Obtener datos del parquet y agrupar a nivel de hexágono
    # Esquema del parquet
    pq_scheme = pq.ParquetFile(os.path.join(INPUT_DATA_DIR, PARQUET_FILE)).schema.names
    pq_inds = [ind for ind in inds if ind in pq_scheme]
    print(pq_inds)

    try: 
        print("Cargando datos dentro de filtered_hexagons")
        main_data = extract_data(file_dir=INPUT_DATA_DIR, file_name=PARQUET_FILE, munip_filter = filter_muni_code, indicadores = pq_inds)
        print("Datos cargados. Columnas:")
        print(main_data.columns)

        print("Agrupando datos dentro de filtered hexagons")
        main_data_gr = group_results(main_data=main_data, group_cols=["MPIO_CDPMP", "hexagon_id"], weight_col="weight", target_cols=pq_inds)

        # Recuperar la geometría de los hexágonos
        hex_geoms = get_hex_pols(main_data_gr["hexagon_id"].tolist())
        hex_geoms['pols_geom'] = hex_geoms['pols_geom'].apply(swap_coordinates)

        main_data_gr_geo = gpd.GeoDataFrame(
            pd.merge(main_data_gr, hex_geoms, on="hexagon_id"), 
            geometry="pols_geom", 
            crs="EPSG:4326"
            )

        return main_data_gr_geo

    except Exception as e:
        print(f"Error al cargar o agregar el archivo parquet: {e}") 
        return 


# 2) Función para generar el resumen del agente intérprete
def generate_area_summary(aggregated_df, user_query, area_info, main_dict_content, client_data_content):
    """
    Generate AI summary using the interpretation agent
    
    Args:
        aggregated_df: DataFrame with aggregated results
        user_query: Original user query
        area_info: Dictionary with area information (tipo_area, intervalo, transporte)
    
    Returns:
        String with the natural language summary
    """
    
    # Create the interpretation agent
    interpretation_agent = create_interpretation_agent(main_dict_content, client_data_content)
    
    # Prepare the data for the interpretation agent
    # df_summary = aggregated_df.drop('geometry', axis=1, errors='ignore').to_dict('records')[0] if len(aggregated_df) > 0 else {}
    # df_summary = aggregated_df.to_dict('records')[0] if len(aggregated_df) > 0 else {}
    df_summary = aggregated_df
    
    # Create the prompt for the interpretation agent
    interpretation_prompt = f"""
    Consulta original del usuario: "{user_query}"
    
    Información del área:
    - Tipo de área: {area_info.get('tipo_area', 'N/A')}
    - Intervalo: {area_info.get('intervalo', 'N/A')} {'minutos' if area_info.get('tipo_area') == 'Isocrona' else 'metros'}
    - Transporte: {area_info.get('transporte', 'N/A')} (si aplica)
    
    Diccionario con los datos de los indicadores:
    {df_summary}
    
    Por favor, genera un resumen natural y comprensible de esta área basado en los datos proporcionados.
    Incluye los valores específicos y contextualiza su significado.
    """
    
    try:
        # Generate the summary
        response = interpretation_agent.chat(interpretation_prompt)
        return response.content
        
    except Exception as e:
        return f"Error al generar el resumen: {str(e)}"

# 4) Función para preparar los datos de la isocrona
def prepare_iso_with_data():

    iso_geometry = create_isochrone_geometry()
    selected_indicators = ai_selected_indicators.get()
    
    if isinstance(iso_geometry, bool):
        return False
        
    try:
        print("Agregando datos de indicadores a la isocrona")
        
        # Indicadores
        inds = [ind for ind in selected_indicators if ind and ind != "None"]
        if not inds:
            return False
        else: 
            inds = list(set(inds))
            
        # Obtener hexágonos con indicadores
        print("hex")
        hexagon_data = filtered_hexagons()
        print("hex final") 

        # Esquema del parquet
        pq_scheme = pq.ParquetFile(os.path.join(INPUT_DATA_DIR, PARQUET_FILE)).schema.names
        pq_inds = [ind for ind in inds if ind in pq_scheme]
        if not pq_inds:
            return False

        # Spatial join
        print("Join espacial")
        main_data_int = gpd.sjoin(iso_geometry, hexagon_data)
        if "hexagon_id" in pq_inds:
            pq_inds.remove("hexagon_id")

        sel_cols_n = ["Intervalo", "hexagon_id"] + pq_inds
        
        main_data_int_f = main_data_int[sel_cols_n]

        #int_gr = aggregate_data(main_data_int)

        print("Datos intersectados con la isocrona")
        return main_data_int_f
        
    except Exception as e:
        print(f"Error agregando datos: {e}")
        return False

def prepare_iso_with_client_data(client_data_filtered):

    iso_geometry = create_isochrone_geometry()
    selected_indicators = ai_selected_indicators.get()
    
    if isinstance(iso_geometry, bool):
        return False
        
    try:
        print("Agregando datos de indicadores a la isocrona")
        
        # Indicadores
        inds = [ind for ind in selected_indicators if ind and ind != "None" and ind in client_data_filtered.columns]
        if not inds:
            return False
        else: 
            inds = list(set(inds))
            
        # Obtener hexágonos con indicadores
        print("hex")
        hexagon_data = filtered_hexagons_client_data(client_data_filtered, inds)
        print("hex final") 

        # Spatial join
        print("Join espacial")
        main_data_int = gpd.sjoin(iso_geometry, hexagon_data)
        if "hexagon_id" in inds:
            inds.remove("hexagon_id")
        sel_cols_n = ["Intervalo", "hexagon_id"] + inds
        print(sel_cols_n)
        
        main_data_int_f = main_data_int[sel_cols_n]

        #int_gr = aggregate_data(main_data_int)

        print("Datos intersectados la isocrona")
        return main_data_int_f
        
    except Exception as e:
        print(f"Error agregando datos: {e}")
        return False

# Modified function to use client data instead of parquet files
def filtered_hexagons_client_data(client_data_filtered, selected_indicators):
    """
    Filter hexagons using client data instead of parquet files
    
    Args:
        client_data_filtered: Processed client dataframe
        selected_indicators: List of indicators to include
        muni_code: Municipality code filter (optional)
    
    Returns:
        GeoDataFrame with hexagon data and indicators
    """
    
    # Filter indicators
    inds = [ind for ind in selected_indicators if ind != "" and ind != "None" and ind is not None]
    print(f"Indicadores en filtered_hexagons_client_data: {inds}")
    
    if not inds: 
        return False
    else: 
        inds = list(set(inds))
    

    try: 
        print("Procesando datos del cliente en filtered_hexagons_client_data")
        
        # Use client data instead of loading from parquet
        main_data = client_data_filtered.copy()
        #main_data.drop(["geometry"], axis=1, inplace=True)
               
        # Select only available indicators that exist in the data
        available_inds = [ind for ind in inds if ind in main_data.columns]
        if not available_inds:
            print("No hay indicadores disponibles en los datos del cliente")
            return False
            
        print("Datos del cliente cargados. Columnas:")
        print(main_data.columns)

        # Get hexagon geometries (this function should work the same)
        hex_geoms = get_hex_pols(main_data["hexagon_id"].tolist())
        hex_geoms['pols_geom'] = hex_geoms['pols_geom'].apply(swap_coordinates)

        main_data_gr_geo = gpd.GeoDataFrame(
            pd.merge(main_data, hex_geoms, on="hexagon_id"), 
            geometry="pols_geom", 
            crs="EPSG:4326"
        )

        return main_data_gr_geo

    except Exception as e:
        print(f"Error al procesar datos del cliente: {e}") 
        return None


# 5) Función para llamar al agente de agregación de datos
def aggregate_iso_with_data(main_data, subqueries):

    iso_geometry = create_isochrone_geometry()
    if not isinstance(iso_geometry, gpd.GeoDataFrame):
        print("Failed to create isochrone geometry")
        return None  
    
    try:
        # Convertir el dataframe a string para dar contexto al agente
        data_info = f"""
        DataFrame shape: {main_data.shape}
        Columns: {list(main_data.columns)}
        Sample data (first 10 rows): {main_data.head(10).to_string()}
        Data types: {main_data.dtypes.to_string()}
        """
      
        
        # Preparar query
        combined_query = f"""
        Tengo el siguiente DataFrame con datos de indicadores por intervalo:
        
        {data_info}
        
        Necesito que respondas las siguientes preguntas agregando los datos por la columna 'Intervalo':
        {' y '.join(subqueries)}
        
        Por favor, realiza las agregaciones correspondientes y devuelve el resultado.
        """
        
        # LLamar al agente
        print("Calling code agent with dataframe and subqueries...")
        code_agent = create_code_agent_client()
        agent_result = code_agent.run(combined_query, additional_args={"df": main_data})
        print(f"Code agent result: {agent_result}")
        int_gr = agent_result
        res_dict = int_gr.to_dict(orient="list")
        print("Todo bien")
        clean_dict = {}
        for k, v in res_dict.items():
            clean_dict[k] = v[0]
        
        print(clean_dict)
        
    except Exception as e:
        print(f"Error in aggregate_iso_with_data: {e}")
        # Fallback 
        print("Devolviendo el promedio")
        indicator_columns = [col for col in main_data.columns if col != 'Intervalo']
        int_gr = main_data.groupby(["Intervalo"])[indicator_columns].mean().reset_index()
        res_dict = int_gr.to_dict(orient="list")
        clean_dict = {}
        for k, v in res_dict.items():
            clean_dict[k] = v[0]
        
    # all_iso_geo = gpd.GeoDataFrame(
    #     pd.merge(int_gr, iso_geometry, on="Intervalo"), 
    #     geometry="geometry", 
    #     crs="EPSG:4326"
    #     )
        
    return res_dict

# 7) Función de integración del flujo de verificación
def process_ai_query_new_architecture(user_message, main_dict_content, indicators_dict, client_data_context = "[]"):
    """
    New 4-agent architecture workflow with PROPER verification and feedback loop
    
    Returns:
        dict: {
            "success": bool,
            "inds": list,
            "subqueries": list,
            "planning_instructions": dict,
            "verification_results": list,
            "final_attempt": int
        }
    """
    
    if not user_message:
        return {
            "success": False,
            "inds": [],
            "subqueries": [],
            "error": "Empty query"
        }
    
    try:
        # 1 --- Analizar consulta y generar instrucciones
        planning_agent = create_planning_agent(client_data_context, main_dict_content, indicators_dict)
        planning_response = planning_agent.chat(user_message)
        
        # Parsear instrucciones
        import json
        try:
            planning_instructions = json.loads(planning_response.content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', planning_response.content, re.DOTALL)
            if json_match:
                planning_instructions = json.loads(json_match.group())
            else:
                raise ValueError("No fue posible parsear las instrucciones como un JSON")
        
        print(f"Instrucciones: {planning_instructions}")
        
        # 2 --- Llamar al agente de subpreguntas e inicializar iteraciones de mejora
        max_attempts = 2
        current_attempt = 0
        verification_results = []
        feedback_for_next_attempt = ""  #Inicializar feedback
        
        while current_attempt <= max_attempts:
            print(f"🔄 Paso 2 -- intento {current_attempt}:")
            
            # Crear agente
            interpreter_agent = create_modified_query_interpreter_agent(client_data_context, main_dict_content)
            if current_attempt == 0:
                interpreter_input = json.dumps(planning_instructions, ensure_ascii=False, indent=2)
                full_prompt = f"Genera las subpreguntas siguiendo estas instrucciones exactas:\n{interpreter_input}"
            else:
                interpreter_input = json.dumps(planning_instructions, ensure_ascii=False, indent=2)
                full_prompt = f"""
                INTENTO #{current_attempt} - CORRECCIÓN REQUERIDA

                FEEDBACK DEL AGENTE VERIFICADOR:
                {feedback_for_next_attempt}

                INSTRUCCIONES ORIGINALES:
                {interpreter_input}

            Por favor, corrige los errores identificados y genera las subpreguntas correctas siguiendo exactamente las instrucciones y el feedback proporcionado.
            """
            
            print(f"Prompt para intérprete (intento {current_attempt}):")
            print(f"{full_prompt[:200]}..." if len(full_prompt) > 200 else full_prompt)
            
            # Enviar prompt
            interpreter_response = interpreter_agent.chat(full_prompt)
            
            # Extraer subpreguntas
            subqueries_text = interpreter_response.content
            
            # Parsear preguntas
            try:
                import ast
                subqueries = ast.literal_eval(subqueries_text)
                if not isinstance(subqueries, list):
                    raise ValueError("Response is not a list")
            except:
                # Fallback con expresiones regulares
                import re
                subqueries = re.findall(r'¿[^?]+\?', subqueries_text)
            
            print(f"Preguntas generadas (intento {current_attempt}): {subqueries}")
            
            # 3 --- Agente de verificación
            print(f"🔍 Paso 3 --- Verificación intento {current_attempt}:")
            verifying_agent = create_verifying_agent(client_data_context, main_dict_content)
            
            verification_input = {
                "instrucciones": planning_instructions["instrucciones"],
                "preguntas_generadas": subqueries
            }
            
            verification_response = verifying_agent.chat(
                f"Verifica si estas preguntas cumplen con las instrucciones:\n{json.dumps(verification_input, ensure_ascii=False, indent=2)}"
            )
            
            # Parsear resultado de verificación
            try:
                verification_result = json.loads(verification_response.content)
            except:
                # Fallback
                verification_result = {
                    "valido": "true" in verification_response.content.lower(),
                    "feedback": verification_response.content
                }
            
            verification_results.append({
                "attempt": current_attempt,
                "result": verification_result
            })
            
            print(f"✅ Resultado verificación (intento {current_attempt}): {verification_result.get('valido', False)}")
            
            # Verificación correcta
            if verification_result.get("valido", False):
                print("🎉 ¡Preguntas correctas! Workflow completado.")
                
                # Extraer indicadores
                import re
                indicators = re.findall(r'\(([^)]+)\)', ' '.join(subqueries))
                indicators = [ind for ind in indicators if ind and ind != "None"]
                
                return {
                    "success": True,
                    "inds": indicators,
                    "subqueries": subqueries,
                    "planning_instructions": planning_instructions,
                    "verification_results": verification_results,
                    "final_attempt": current_attempt
                }
            
            # Verificación inválida
            elif current_attempt <= max_attempts:
                print(f"Verificación inválida. Preparando reintento {current_attempt + 1}...")
                feedback_for_next_attempt = verification_result.get("feedback", "Please follow the instructions more carefully.")
                print(f"📋 Feedback para próximo intento: {feedback_for_next_attempt}")
                
                current_attempt += 1
                # ✅ Loop continues and will use feedback_for_next_attempt in next iteration

            # Máximo de intentos alcanzados    
            else:
              
                # Usar indicadores el último intento
                import re
                indicators = re.findall(r'\(([^)]+)\)', ' '.join(subqueries))
                indicators = [ind for ind in indicators if ind and ind != "None"]
                
                return {
                    "success": False,
                    "inds": indicators,
                    "subqueries": subqueries,
                    "planning_instructions": planning_instructions,
                    "verification_results": verification_results,
                    "final_attempt": current_attempt,
                    "error": "Validation failed after maximum attempts"
                }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "success": False,
            "inds": [],
            "subqueries": [],
            "error": str(e)
        }


# 7) Función de orquestración
# def process_ai_query_new_architecture_client(user_message, client_data_context, client_indicators_dict=None):
#     """
#     New 4-agent architecture workflow using client data instead of hardcoded data
    
#     Args:
#         user_message: User's query
#         client_data_context: String representation of client's data dictionary
#         client_indicators_dict: Optional indicators dictionary from client
    
#     Returns:
#         dict: Results of the workflow
#     """
    
#     if not user_message:
#         return {
#             "success": False,
#             "inds": [],
#             "subqueries": [],
#             "error": "Empty query"
#         }
    
#     try:
#         # 1 --- Planning agent with client data
#         planning_agent = create_planning_agent(client_data_context, client_indicators_dict)
#         planning_response = planning_agent.chat(user_message)
        
#         # Parse instructions
#         import json
#         try:
#             planning_instructions = json.loads(planning_response.content)
#         except json.JSONDecodeError:
#             import re
#             json_match = re.search(r'\{.*\}', planning_response.content, re.DOTALL)
#             if json_match:
#                 planning_instructions = json.loads(json_match.group())
#             else:
#                 raise ValueError("No fue posible parsear las instrucciones como un JSON")
        
#         print(f"Instrucciones con datos del cliente: {planning_instructions}")
        
#         # 2 --- Questions agent with client data
#         max_attempts = 2
#         current_attempt = 0
#         verification_results = []
#         feedback_for_next_attempt = ""
        
#         while current_attempt <= max_attempts:
#             print(f"🔄 Paso 2 -- intento {current_attempt} con datos del cliente:")
            
#             # Create interpreter agent with client data
#             interpreter_agent = create_modified_query_interpreter_agent(client_data_context)
#             if current_attempt == 0:
#                 interpreter_input = json.dumps(planning_instructions, ensure_ascii=False, indent=2)
#                 full_prompt = f"Genera las subpreguntas siguiendo estas instrucciones exactas:\n{interpreter_input}"
#             else:
#                 interpreter_input = json.dumps(planning_instructions, ensure_ascii=False, indent=2)
#                 full_prompt = f"""
#                 INTENTO #{current_attempt} - CORRECCIÓN REQUERIDA

#                 FEEDBACK DEL AGENTE VERIFICADOR:
#                 {feedback_for_next_attempt}

#                 INSTRUCCIONES ORIGINALES:
#                 {interpreter_input}

#             Por favor, corrige los errores identificados y genera las subpreguntas correctas siguiendo exactamente las instrucciones y el feedback proporcionado.
#             """
            
#             print(f"Prompt para intérprete (intento {current_attempt}):")
#             print(f"{full_prompt[:200]}..." if len(full_prompt) > 200 else full_prompt)
            
#             # Send prompt
#             interpreter_response = interpreter_agent.chat(full_prompt)
            
#             # Extract subqueries
#             subqueries_text = interpreter_response.content
            
#             # Parse questions
#             try:
#                 import ast
#                 subqueries = ast.literal_eval(subqueries_text)
#                 if not isinstance(subqueries, list):
#                     raise ValueError("Response is not a list")
#             except:
#                 # Fallback with regex
#                 import re
#                 subqueries = re.findall(r'¿[^?]+\?', subqueries_text)
            
#             print(f"Preguntas generadas con datos del cliente (intento {current_attempt}): {subqueries}")
            
#             # 3 --- Verifying agent with client data
#             print(f"🔍 Paso 3 --- Verificación intento {current_attempt} con datos del cliente:")
#             verifying_agent = create_verifying_agent(client_data_context)
            
#             verification_input = {
#                 "instrucciones": planning_instructions["instrucciones"],
#                 "preguntas_generadas": subqueries
#             }
            
#             verification_response = verifying_agent.chat(
#                 f"Verifica si estas preguntas cumplen con las instrucciones:\n{json.dumps(verification_input, ensure_ascii=False, indent=2)}"
#             )
            
#             # Parse verification result
#             try:
#                 verification_result = json.loads(verification_response.content)
#             except:
#                 # Fallback
#                 verification_result = {
#                     "valido": "true" in verification_response.content.lower(),
#                     "feedback": verification_response.content
#                 }
            
#             verification_results.append({
#                 "attempt": current_attempt,
#                 "result": verification_result
#             })
            
#             print(f"✅ Resultado verificación con datos del cliente (intento {current_attempt}): {verification_result.get('valido', False)}")
            
#             # Successful verification
#             if verification_result.get("valido", False):
#                 print("🎉 ¡Preguntas correctas con datos del cliente! Workflow completado.")
                
#                 # Extract indicators
#                 import re
#                 indicators = re.findall(r'\(([^)]+)\)', ' '.join(subqueries))
#                 indicators = [ind for ind in indicators if ind and ind != "None"]
                
#                 return {
#                     "success": True,
#                     "inds": indicators,
#                     "subqueries": subqueries,
#                     "planning_instructions": planning_instructions,
#                     "verification_results": verification_results,
#                     "final_attempt": current_attempt
#                 }
            
#             # Invalid verification - prepare retry
#             elif current_attempt < max_attempts:
#                 print(f"Verificación inválida. Preparando reintento {current_attempt + 1}...")
#                 feedback_for_next_attempt = verification_result.get("feedback", "Please follow the instructions more carefully.")
#                 print(f"📋 Feedback para próximo intento: {feedback_for_next_attempt}")
                
#                 current_attempt += 1

#             # Max attempts reached    
#             else:
#                 # Use indicators from last attempt
#                 import re
#                 indicators = re.findall(r'\(([^)]+)\)', ' '.join(subqueries))
#                 indicators = [ind for ind in indicators if ind and ind != "None"]
                
#                 return {
#                     "success": False,
#                     "inds": indicators,
#                     "subqueries": subqueries,
#                     "planning_instructions": planning_instructions,
#                     "verification_results": verification_results,
#                     "final_attempt": current_attempt,
#                     "error": "Validation failed after maximum attempts"
#                 }
        
#     except Exception as e:
#         print(f"❌ Error en workflow con datos del cliente: {e}")
#         return {
#             "success": False,
#             "inds": [],
#             "subqueries": [],
#             "error": str(e)
#         }



# 8) Función para procesar datos del cliente
# def prepare_client_data(df, selected_analysis_cols, geo_ref_cols, cat_cols, count_cols, id_cols, res=HEX_RES):

#     # Seleccionar columnas relevantes
#     df_f = df.copy()
#     keep_cols = selected_analysis_cols + geo_ref_cols + id_cols + cat_cols + count_cols
#     df_f = df[keep_cols]
#     print(f"cols: {df_f.columns}")

#     # Convertir variables categóricas a dummies
#     df_f = pd.concat(
#         [df_f, pd.get_dummies(df_f[cat_cols])],
#         axis=1, 
#         join="inner"
#     )

#     # Convertir NAs en 0
#     df_f[selected_analysis_cols] = df_f[selected_analysis_cols].fillna(0)

#     df_f.drop(cat_cols, axis=1, inplace=True)

#     # Convertir latitud y longitud a coordenada
#     lat_col_name = None
#     lon_col_name = None
    
#     for col in geo_ref_cols:
#         if re.search(r'lat', col, re.IGNORECASE):
#             lat_col_name = col
#         elif re.search(r'lon', col, re.IGNORECASE):
#             lon_col_name = col
    
#     df_f['geometry'] = df_f.apply(
#         lambda row: Point(row[lon_col_name], row[lat_col_name]), 
#         axis=1
#     )
    
#     # Convertir a Geopandas
#     geo_df = gpd.GeoDataFrame(
#         data=df_f, 
#         geometry="geometry", 
#         crs="epsg:4326"
#     ) 

#     # Cruzar con hexágonos
#     def point_to_h3(point, resolution):
#         return h3.latlng_to_cell(point.y, point.x, resolution)

#     geo_df["hexagon_id"] = geo_df.geometry.apply(
#         lambda x: point_to_h3(x, res)
#     )

#     geo_df.drop([lat_col_name, lon_col_name], axis=1, inplace=True)   

#     # Convertir columnas a mayúsculas
#     #geo_df.columns = [col.upper() if col != "geometry" else col for col in geo_df.columns]

#     # Preparar aggregación
#     exc_vars = count_cols + id_cols + ["geometry", "hexagon_id"]

#     # Conteos únicos
#     group_dict = {k: ["nunique"] for k in count_cols}
#     group_dict.update({k: ["count"] for k in id_cols}) # Conteo total

#     # Sumas
#     group_dict.update({k: ["sum"] for k in geo_df.columns if k not in exc_vars})
#     print(f"Llaves: {group_dict.keys()}")

#     geo_df.drop(["geometry"], axis=1, inplace=True)
#     geo_df_g = geo_df.groupby(["hexagon_id"]).agg(group_dict).reset_index()
#     #geo_df_g.columns = geo_df_g.columns.droplevel(1)
#     geo_df_g.columns = [f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in geo_df_g.columns]

#     return geo_df_g
def prepare_client_data(df, selected_analysis_cols, geo_ref_cols, cat_cols, count_cols, id_cols, res=HEX_RES):

    # Seleccionar columnas relevantes
    df_f = df.copy()
    keep_cols = selected_analysis_cols + geo_ref_cols + id_cols + cat_cols + count_cols
    df_f = df[keep_cols]

    # Convertir variables categóricas a dummies
    dummy_df = pd.get_dummies(df_f[cat_cols], prefix_sep='_')
    df_f = pd.concat([df_f, dummy_df], axis=1, join="inner")
    print("COLUMAS CONCAT:")
    print(df_f.columns)

    # Convertir NAs en 0
    df_f[selected_analysis_cols] = df_f[selected_analysis_cols].fillna(0)

    # Drop original categorical columns AFTER creating dummies
    df_f.drop(cat_cols, axis=1, inplace=True)

    # Convertir latitud y longitud a coordenada
    lat_col_name = None
    lon_col_name = None
    
    for col in geo_ref_cols:
        if re.search(r'lat', col, re.IGNORECASE):
            lat_col_name = col
        elif re.search(r'lon', col, re.IGNORECASE):
            lon_col_name = col
    
    df_f['geometry'] = df_f.apply(
        lambda row: Point(row[lon_col_name], row[lat_col_name]), 
        axis=1
    )
    
    # Convertir a Geopandas
    geo_df = gpd.GeoDataFrame(
        data=df_f, 
        geometry="geometry", 
        crs="epsg:4326"
    ) 

    # Cruzar con hexágonos
    def point_to_h3(point, resolution):
        return h3.latlng_to_cell(point.y, point.x, resolution)

    geo_df["hexagon_id"] = geo_df.geometry.apply(
        lambda x: point_to_h3(x, res)
    )

    geo_df.drop([lat_col_name, lon_col_name], axis=1, inplace=True)   

    # Preparar aggregación
    exc_vars = count_cols + id_cols + ["geometry", "hexagon_id"]

    # Conteos únicos
    group_dict = {k: ["nunique"] for k in count_cols}
    group_dict.update({k: ["count"] for k in id_cols}) # Conteo total

    # Get dummy column names (they should include the categorical dummies)
    dummy_cols = [col for col in geo_df.columns if any(cat_col in col for cat_col in cat_cols)]
    
    # Sumas - explicitly include analysis cols and dummy cols
    analysis_and_dummy_cols = selected_analysis_cols + dummy_cols
    for k in analysis_and_dummy_cols:
        if k in geo_df.columns and k not in exc_vars:
            group_dict[k] = ["sum"]
    
    print(f"Columns before groupby: {geo_df.columns.tolist()}")
    print(f"Aggregation dictionary: {group_dict}")

    geo_df.drop(["geometry"], axis=1, inplace=True)
    geo_df_g = geo_df.groupby(["hexagon_id"]).agg(group_dict).reset_index()
    geo_df_g.columns = [f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in geo_df_g.columns]

    print(f"Final columns after aggregation: {geo_df_g.columns.tolist()}")
    
    return geo_df_g


# 9) Función para generar un nuevo mapeo de variables con los datos del cliente
def update_dict(processed_df, data_dict, cat_cols):

    # Patrón de búsqueda
    pattern = r"^(.*?)_(sum|count|nunique)$"

    # Diccionario
    data_dict = data_dict.to_dict("list")
    new_dict = {
        "Nombre": [], 
        "Descripción": [], 
        "Tipo": [],
        "Llave": []
    }

    for col in processed_df.columns:
        if not re.findall("hexagon_id", col):
            indicator_name = re.search(pattern, col).group(1)
            operation_name = re.search(pattern, col).group(2)
            if re.search(rf"^({'|'.join(cat_cols)})", indicator_name):
                indicator_name_full = indicator_name
                indicator_name = indicator_name.split("_")[0]

            print(f"{indicator_name} - {operation_name}")
            try:
                indicator_index = data_dict.get("Nombre").index(indicator_name)
                original_description = data_dict.get("Descripción")[indicator_index]
                var_type = data_dict.get("Tipo")[indicator_index]
                llave_type = data_dict.get("Llave")[indicator_index]
            except: 
                print(f"El indicador {indicator_name} no está en el diccionario de datos")
                var_type = None
                original_description = ""
            
            new_description = original_description
            if operation_name == "sum" and var_type != "Categórica" and var_type:
                new_description = "Suma del " + original_description.lower()
            elif operation_name == "sum" and var_type == "Categórica": 
                new_description = "Suma del " + original_description.lower() + " cuando la categoría es " + indicator_name_full.split("_")[1]
            elif operation_name == "nunique" or operation_name == "count":
                print(original_description)
                if llave_type.strip() == "Identificación grupo" and re.search(r"familiar|grupo", original_description):
                    new_description = "Número de familias"
                elif llave_type.strip() == "Identificación usuario":
                    new_description = "Número de afiliados"
                else: 
                    new_description = ""
            else:
                new_description = original_description
        
            new_dict["Nombre"].append(col)
            new_dict["Descripción"].append(new_description)
            new_dict["Tipo"].append(var_type) 
            new_dict["Llave"].append(llave_type)

    new_dict_df = pd.DataFrame(new_dict)
    new_dict_df["Dimensión"] = "Afiliados"
    new_dict_content = new_dict_df.to_string(index=False)

    return new_dict_content

# 10) Función para extraer archivos cargados por el cliente
def get_uploaded_file_data(file_index=0, file_type="data"): 
    """
    Access uploaded file data
    
    Args:
        file_index (int): Index of the file to access (0 for first file, 1 for second, etc.)
        file_type (str): Type of file expected - "data" for analysis files, "dict" for data dictionaries
    
    Returns:
        pandas.DataFrame or dict: The file content, or None if file not found/invalid
    """
    try:
        # Get processed files
        files = process_uploaded_files()
        
        if not files or len(files) <= file_index:
            print(f"No file found at index {file_index}")
            return None
        
        file_info = files[file_index]
        
        # Check if file had processing errors
        if 'error' in file_info:
            print(f"Error in file {file_info['name']}: {file_info['error']}")
            return None
        
        file_content = file_info['content']
        file_dict = file_info["dict"]
        file_name = file_info['name']
        file_ext = file_info['extension']
        
        print(f"Accessing file: {file_name} ({file_ext})")
        
        # Return content based on file type
        if file_type == "data":
            # For data files, ensure it's a DataFrame
            if isinstance(file_content, pd.DataFrame):
                return {"data": file_content, "dict": file_dict}
            else:
                print(f"File {file_name} is not a valid data file (not a DataFrame)")
                return None
                
        elif file_type == "dict":
            # For dictionary files (JSON or Excel with dictionary structure)
            if file_ext == 'json':
                return file_content
            elif isinstance(file_content, pd.DataFrame):
                # Convert DataFrame to dict if needed
                return file_content
            else:
                print(f"File {file_name} is not a valid dictionary file")
                return None
        
        else:
            # Return raw content
            return file_content
            
    except Exception as e:
        print(f"Error accessing uploaded file: {e}")
        return None

# 11) Función wrapper de extracción de archivos
def get_all_uploaded_files():
    """
    Get all uploaded files with their metadata
    
    Returns:
        list: List of dictionaries with file information
    """
    try:
        files = process_uploaded_files()
        if not files:
            return []
        
        file_summary = []
        for i, file_info in enumerate(files):
            summary = {
                'index': i,
                'name': file_info['name'],
                'extension': file_info['extension'],
                'size_kb': round(file_info['size'] / 1024, 1),
                'has_error': 'error' in file_info,
                'is_dataframe': isinstance(file_info.get('content'), pd.DataFrame),
                'shape': file_info['content'].shape if isinstance(file_info.get('content'), pd.DataFrame) else None,
                'columns': list(file_info['content'].columns) if isinstance(file_info.get('content'), pd.DataFrame) else None
            }
            file_summary.append(summary)
        
        return file_summary
        
    except Exception as e:
        print(f"Error getting file summary: {e}")
        return []

# 12) Función para procesar archivos del cliente
def process_uploaded_files():
    """Process and return information about uploaded files"""
    files = input.files_upload()
    
    if not files:
        return None
    
    processed_files = []
    
    for i, file_info in enumerate(files):
        try:
            # File metadata
            file_name = file_info['name']
            file_size = file_info['size']
            file_ext = file_name.split('.')[-1].lower()
            
            # Read file content based on extension
            file_content = None
            if file_ext == 'csv':
                file_content = pd.read_csv(file_info['datapath'], sheet_name=0)
                file_dict = pd.read_csv(file_info['datapath'], sheet_name=1)
            elif file_ext in ['xlsx', 'xls']:
                file_content = pd.read_excel(file_info['datapath'], sheet_name=0)
                file_dict = pd.read_excel(file_info['datapath'], sheet_name=1)
            
            processed_files.append({
                'index': i,
                'name': file_name,
                'extension': file_ext,
                'size': file_size,
                'datapath': file_info['datapath'],
                'content': file_content, 
                'dict': file_dict, 
                
            })
            
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            processed_files.append({
                'index': i,
                'name': file_name,
                'extension': file_ext,
                'size': file_size,
                'error': str(e)
            })
    
    return processed_files

# 13) Función para obtener la categoría de las variables
def extract_col_categories(data_dict):
    cat_dict = {
        "analysis": data_dict[data_dict["Tipo"] == "Numérica"]["Nombre"].tolist(), 
        "categorical": data_dict[data_dict["Tipo"] == "Categórica"]["Nombre"].tolist(), 
        "geo_ref": data_dict[data_dict["Tipo"] == "Coordenada"]["Nombre"].tolist(), 
        "grouping": data_dict[(data_dict["Llave"] == "Identificación grupo") & (data_dict["Nombre"] != "ID_AFILIADO")]["Nombre"].tolist(),
        "id": data_dict[(data_dict["Llave"] == "Identificación usuario") & (data_dict["Nombre"] == "ID_AFILIADO")]["Nombre"].tolist()
    }
    
    return cat_dict

# Modified prompt for the Query Interpreter (Questions Agent)
def create_agente_etiquetado_prompt(client_data_context, main_dict_content):
   return  f"""
    --- ROL ---
    Eres un agente intérprete de consultas que recibe instrucciones específicas del agente de planificación
    y genera subpreguntas estructuradas siguiendo esas instrucciones exactamente.
    
    --- CONTEXTO DE DATOS ---
    Archivo tabular de indicadores sobre información pública: {main_dict_content}
    Archivo tabular de indicadores sobre información de afiliados: {client_data_context}
    
    --- INPUT ---
    Recibirás instrucciones en formato JSON de un agente planificador con esta estructura:
    {{
        "dimensiones_relevantes": ["dimension1"],
        "tipo_consulta": "dimension" | "especifica" | "general",
        "instrucciones": [
            {{
                "dimension": "nombre_dimension",
                "num_preguntas": 3,
                "indicadores": ["ind1", "ind2", "ind3"],
                "agregaciones": ["suma", "promedio", "suma"]
            }}
        ]
    }}
    
    --- TAREA ---
    Generar exactamente las subpreguntas especificadas en las instrucciones:
    1. Usar EXACTAMENTE los indicadores especificados (códigos técnicos)
    2. Usar EXACTAMENTE las agregaciones especificadas
    3. Generar EXACTAMENTE el número de preguntas indicado
    4. Seguir el formato: ¿Cuál es |agregación| del indicador (código)?
    
    --- OUTPUT ---
    Retorna ÚNICAMENTE una lista de Python con las subpreguntas generadas.
    
    --- EJEMPLOS ---
    1)
    Input: {{
        "instrucciones": [{{
            "dimension": "demografica",
            "num_preguntas": 2,
            "indicadores": ["TP27_PERSO", "TP51SUPERI"],
            "agregaciones": ["suma", "promedio"]
        }}]
    }}
    
    Output: ["¿Cuál es |la suma| del indicador (TP27_PERSO)?", "¿Cuál es |el promedio| del indicador (TP51SUPERI)?"]

    2)
    Input: {{
        "instrucciones": [{{
            "dimension": "laboral",
            "num_preguntas": 2,
            "indicadores": ["TP19_EE_1", "TP19_EE_1"],
            "agregaciones": ["suma", "promedio"]
        }}]
    }}
    
    Output: ["¿Cuál es |la suma| del indicador (TP19_EE_1)?", "¿Cuál es |el promedio| del indicador (TP19_EE_1)?"]

    3)
    Input: {{
        "instrucciones": [{{
            "dimension": "afiliados",
            "num_preguntas": 1,
            "indicadores": ["ID_AFILIADO_count"],
            "agregaciones": ["suma"]
        }}]
    }}
    
    Output: ["¿Cuál es |la suma| del indicador (ID_AFILIADO_count)?"]
    """

# Modified prompt for the Planning Agent
def create_planning_agent_prompt(main_dict_content, INDICATORS_DICT, client_data_context="[]"):
    # Prompt agente de pleaneación
    return f"""
--- ROL ---
Eres un agente de planificación especializado en analizar consultas en lenguaje natural y crear instrucciones 
específicas para un agente intérprete de consultas. Tu tarea es entender la consulta del usuario, mapearla 
a las dimensiones e indicadores disponibles, y generar instrucciones claras y precisas.

--- CONTEXTO DE DATOS ---
Tienes acceso a la misma información que el agente intérprete:
1. Un archivo tabular de información pública con indicadores organizados por dimensiones
2. Un diccionario de dimensiones e indicadores sobre información pública
3. Un archivo tabular de información sobre los afiliados a una empresa de salud
 

El archivo tabular con la información pública es este: {main_dict_content}
El diccionario de dimensiones de la información pública es este: {json.dumps(INDICATORS_DICT, indent=2, ensure_ascii=False)}
El archivo tabular con la información de los afiliados a la empresa de salud es {client_data_context}. Si este archivo es 
una lista vacía, [], puedes ignorarlo. 

--- TAREA PRINCIPAL ---
Tu función es generar instrucciones específicas para el agente intérprete, no generar las subpreguntas directamente.

Proceso:
1. Analiza la consulta del usuario para entender el tema principal
2. Identifica las dimensiones e indicadores relevantes, tanto sobre la información pública como sobre la información de los afiliados
3. Determina cuántas preguntas se necesitan y sobre qué temas. Siempre que te pregunten sobre información de afiliados, regímenes de cotización a salud, 
o condiciones de salud de los afiliados, debes revisar el archivo de información de afiliados. El archivo de información pública no 
contiene información de afiliados. 
4. Genera instrucciones específicas en formato JSON

--- FORMATO DE INSTRUCCIONES ---
Debes retornar un JSON con esta estructura:
{{
    "dimensiones_relevantes": ["dimension1", "dimension2"],
    "indicadores_especificos": ["indicador1", "indicador2"],  // Si la consulta es específica
    "tipo_consulta": "especifica" o "dimension" o "general",
    "instrucciones": [
        {{
            "dimension": "nombre_dimension",
            "num_preguntas": 3,
            "indicadores": ["ind1", "ind2", "ind3"],  // Indicadores específicos por código técnico
            "agregaciones": ["suma", "promedio", "suma"]  // Agregación para cada indicador
        }}
    ]
}}

Puede haber casos en los que el usuario te pide más de una agregación para el mismo indicador. Por ejemplo, te puede pedir
el promedio y la suma de un indicador. En ese caso, aunque el indicador sea el mismo, asegúrate de incluirlo tantas veces
como agregaciones sobre el indicador sean pedidas. 
Solo debes retornar el JSON. No des descripciones ni comentarios. 

--- REGLAS ---
1. Si la consulta menciona indicadores específicos Y agregaciones: tipo_consulta = "especifica"
2. Si menciona solo dimensiones: tipo_consulta = "dimension" 
3. Si es muy general: tipo_consulta = "general"
4. Para consultas generales, sigues estas reglas: 
    - Si el archivo de información de afiliados es una lista vacía, [], omítelo. En caso contrario, 
    pregunta por la suma del indicador ID_AFILIADO_count. 
    - En todos los casos, haz preguntas de estos indicadores, siguiendo la columna de agregación en el archivo tabular
    de información pública: TP19_EE_E3, TP19_EE_1, TP19_ACU_1, TP19_GAS_1, TP27_PERSO, TP34_3_EDA, TP34_7_EDA, DENS_EMP_M2. 
5. Para consultas de una dimensión específica, sigue estas reglas. Debes seleccionar, según la dimensión o las dimensiones de la consulta, 
    los siguientes indicadores: 
    - Acceso a servicios: TP19_EE_1, TP19_ACU_1, TP19_GAS_1
    - Socioeconómica: TP19_EE_E1, TP19_EE_E3, TP19_EE_E5
    - Económica: DENS_EMP_M2
    - Demográfica: TP27_PERSO, TP32_1_SEX, TP32_2_SEX, TP34_3_EDA, TP34_5_EDA, TP34_7_EDA, TP34_9_EDA
    - Educación: TP51SUPERI, TP51POSTGR
    - Costos: AVALUO_PROM_IMP
    - Ambiental: RIESGO_PLUV_MOD
    - Accesibilidad: AVG_DIST, N_RUTAS

--- EJEMPLOS ---
Input: "Dame información demográfica de esta zona"
Output: {{
    "dimensiones_relevantes": ["demografica"],
    "indicadores_especificos": [],
    "tipo_consulta": "dimension",
    "instrucciones": [
        {{
            "dimension": "demografica",
            "num_preguntas": 5,
            "indicadores": ["TP27_PERSO", "TP32_1_SEX", "TP32_2_SEX", "TP34_3_EDA", "TP34_5_EDA", "TP34_7_EDA", "TP34_9_EDA"],
            "agregaciones": ["suma", "suma", "suma", "suma", "suma", "suma", "suma"]
        }}
    ]
}}

Input: "Cuántas personas viven en total"
Output: {{
    "dimensiones_relevantes": ["demografica"],
    "indicadores_especificos": ["TP27_PERSO"],
    "tipo_consulta": "especifica",
    "instrucciones": [
        {{
            "dimension": "demografica", 
            "num_preguntas": 1,
            "indicadores": ["TP27_PERSO"],
            "agregaciones": ["suma"]
        }}
    ]
}}

Input: "Cuál es el promedio, el máximo y el mínimo de los costos en esta zona"
Output: {{
    "dimensiones_relevantes": ["costos"],
    "indicadores_especificos": ["AVALUO_PROM_IMP"],
    "tipo_consulta": "especifica",
    "instrucciones": [
        {{
            "dimension": "económica", 
            "num_preguntas": 3,
            "indicadores": ["TP9_1_USO", "TP9_1_USO", "TP9_1_USO"],
            "agregaciones": ["promedio", "máximo", "mínimo"]
        }}
    ]
}}

Input: "Cuántos afiliados viven en esta zona" # Ejemplo de un caso en el que el archivo de afiliados no es []
Output: {{
    "dimensiones_relevantes": ["afiliados"],
    "indicadores_especificos": ["ID_AFILIADO_count"],
    "tipo_consulta": "especifica",
    "instrucciones": [
        {{
            "dimension": "afiliados", 
            "num_preguntas": 1,
            "indicadores": ["ID_AFILIADO_count"],
            "agregaciones": ["suma"]
        }}
    ]
}}

"""

# Modified prompt for the Verifying Agent
def create_verifying_agent_prompt(main_dict_content, client_data_context="[]"):
    return f"""
--- ROL ---
Eres un agente verificador que examina si el output del agente intérprete cumple con las instrucciones 
del agente de planificación. Tu tarea es validar que las preguntas generadas coincidan exactamente 
con lo solicitado.

--- CONTEXTO DE DATOS ---
Archivo tabular de indicadores de datos públicos: {main_dict_content}
Archivo tabular de datos sobre afiliados a una empresa de salud: {client_data_context}. Si es una lista vacía, [], 
puedes ignorarlo. 

--- INPUTS ---
Recibirás:
1. Las instrucciones originales del agente de planificación (formato JSON)
2. El output del agente de interpretación (lista de preguntas)

--- TAREA PRINCIPAL ---
Verificar que las preguntas cumplan con las instrucciones:

1. **Número de preguntas**: Verificar que haya exactamente el número especificado de preguntas por dimensión
2. **Indicadores correctos**: Verificar que se usen los códigos técnicos exactos especificados
3. **Formato correcto**: Verificar que sigan la estructura ¿Cuál es |agregación| del indicador (código)?
4. **Agregaciones correctas**: Verifica que las agregaciones de cada indicador concuerden con las agregaciones en las instrucciones
    del agente de planificación. Ten en cuenta que a veces las agregaciones en las preguntas pueden ser distintas a las que aparecen en el archivo tabular. 
    En esos casos, debes verificar que la agregación en la pregunta corresponda con la agregación indicada por el agente planificador; NO te bases únicamente
    en el archivo tabular y sé flexible con las agregaciones. 
    Algunas agregaciones permitidas para todos los indicadores son: 
    "máximo", "mínimo", "promedio", "mediana", "suma", "promedio ponderado", entre otras. Recuerda siempre que las agregaciones
    son operaciones estadísticas. Por tanto, aunque la agregación de la pregunta no concuerde con el archivo tabular, debes aceptarle SOLO si
    concuerda con las instrucciones del agente planificador. 

--- OUTPUT FORMAT ---
Retorna un JSON con esta estructura:
{{
    "valido": true/false,
    "errores": [
        {{
            "tipo": "numero_preguntas" | "indicador_incorrecto" | "agregacion_incorrecta" | "formato_incorrecto",
            "descripcion": "descripción específica del error",
            "esperado": "valor esperado",
            "encontrado": "valor encontrado"
        }}
    ],
    "feedback": "Feedback específico para corregir los errores encontrados"
}}

--- EJEMPLOS ---
Input - Instrucciones: {{
    "instrucciones": [{{
        "dimension": "demografica",
        "num_preguntas": 2, 
        "indicadores": ["TP27_PERSO", "TP51SUPERI"],
        "agregaciones": ["suma", "promedio"]
    }}]
}}

Input - Preguntas del Intérprete: [
    "¿Cuál es |la suma| del indicador (TP27_PERSO)?",
    "¿Cuál es |el promedio| del indicador (TP51SUPERI)?"
]

Output: {{
    "valido": true,
    "errores": [],
    "feedback": "Todas las preguntas cumplen con las instrucciones correctamente."
}}

Input - Caso con errores:
Preguntas del Intérprete: [
    "¿Cuál es |la suma| del indicador (TP27_PERSO)?",
    "¿Cuál es |la suma| del indicador (TP51SUPERI)?",  # Debería ser promedio
    "¿Cuál es |el máximo| del indicador (OTRO_COD)?"   # Pregunta extra no solicitada
]

Output: {{
    "valido": false,
    "errores": [
        {{
            "tipo": "agregacion_incorrecta",
            "descripcion": "Agregación incorrecta para TP51SUPERI",
            "esperado": "promedio", 
            "encontrado": "suma"
        }},
        {{
            "tipo": "numero_preguntas",
            "descripcion": "Número incorrecto de preguntas",
            "esperado": "2",
            "encontrado": "3"
        }}
    ],
    "feedback": "Corrección necesaria: 1) Cambiar la agregación de TP51SUPERI a 'promedio', 2) Generar exactamente 2 preguntas, no 3"
}}

Input - Caso con errores:
Preguntas del Intérprete: [
    "¿Cuál es |la suma| del indicador (TP27_PERSO)?"  # Debería haber más de una pregunta
]

Output: {{
    "valido": false,
    "errores": [
        {{
            "tipo": "numero_preguntas",
            "descripcion": "Número incorrecto de preguntas",
            "esperado": "3", 
            "encontrado": "1"
        }}
    ],
    "feedback": "Corrección necesaria: 1) Generar exactamente 3 preguntas relacionadas con la consulta, no 1"
}}


"""

# Modified prompt for the Interpretation Agent
def create_prompt_agente_interpretacion(main_dict_content, client_data_context):
    return f"""
--- ROL ---
Eres un analista de datos experto en tomar una base de datos y hacer un resumen claro, conciso y natural sobre ella. Tu tarea
consiste en recibir información de un área y realizar un resumen de sus indicadores de forma técnica y veraz.

--- CONTEXTO DE DATOS ---
Se te proporcionará 4 inputs distintos: 
1. La consulta en lenguaje natural del usuario original. 
2. Un archivo tabular con seis columnas: Dimensión, Label, Nombre, Referencia, Agregación y Tipo. Este archivo contiene
    información de indicadores pertenecientes a distintas dimensiones de INFORMACIÓN PÚBLICA, así como su valores de referencia, las operaciones estadísticas típicas
    que se hacen con ellos y el tipo de indicador que son. 
        2.1 La columna Dimensión indica la dimensión a la que pertenecen una serie de indicadores. 
            Ejemplos: ambiental (e.g.: si se trata de temas ambientales o ecológicos), demográfica (e.g.: temas poblacionales), 
            económica (e.g.: ingresos, tasas de empleo, industria), laboral (e.g: tasas de empleo). 
            La dimensión englobal la temática general de una serie de indicadores que le corresponden. 
        2.2 La columna Label es una descripción en lenguaje natural de un indicador. 
            Ejemplo: Total Personas, Tasa de Empleo, Calidad del aire, Densidad empresarial. 
            Label permite a un usuario entender en lenguaje natural la variable que mide un indicador específico. 
        2.3 La columna Nombre tiene el nombre técnico del indicador. 
            Ejemplo: TP27_PERSO.
            Nombre contiene, para cada indicador, el valor en el que está almacenado en una base de datos. Los usuarios no pueden 
            entender el tema del indicador a partir de Nombre. Sin embargo, es crucial para que nosotros podamos mapear la consulta del usuario
            en lenguaje natural a los indicadores que debemos utilizar en nuestras bases de datos. 
        2.4 La columna Referencia tiene un valor de referencia para cada indicador.
            Ejemplo: 100000. 
            Cada indicador tiene un valor en la base de datos. La columna Referencia permite entender si ese valor es mayor o menor
            que el valor de referencia. 
        2.5 La columna Agregación tiene la agregación u operación usual para cada indicador.
            Ejemplo: suma, promedio. 
            Algunas variables suelen ser tratadas con operaciones típicas. Por ejemplo, cuando se agregan tasas, se suelen utilizar promedios simples
            o ponderados. Cuando se trata de conteos, como el total de persona, típicamente interesa hacer una suma. Eso no significa que otras operaciones
            no sean adecuadas. Los valores en Agregación son únicamente operaciones típicas, de ningún modo excluyentes. 
        2.6 La columna Tipo indica si el indicador es una suma, porcentaje, entre otros. 
            Ejemplo: porcentaje, suma, índice. 
            Esta columna da mayor información para entender la naturaleza del indicador. 
    El archivo tabular es este: {main_dict_content}
3. Un diccionario con los resultados agregados por indicadores y agregación estadística.
    La estructura del diccionario es: 
    {
        {
        ("indicador", "agregación") : "valor"
        }
    }

    Las llaves son tuplas con la forma (nombre del indicador, agregación aplicada). Los valores son números. 
    Esto te ayudará a saber cuál es el valor y la operación para cada indicador. 

4. Información del área geográfica (tipo, intervalo, transporte).
5. Un archivo tabular con información sobre usuarios de una empresa de salud: {client_data_context}. Si su valor es [], 
puedes ignorarlo. Es equivalente al archivo tabular para información pública, pero tiene las siguientes columnas relevantes: 
    4.1 Nombre: nombre técnico del indicador. 
    4.2 Descripción: nombre en lenguaje natural del indicador. Es equivalente a la columna de label del archivo de información pública
    y debes usarlo para traducir los nombres técnicos de los indicadores de los afiliados de salus (si los hay) a lenguaje natural. 
    4.3 Tipo. No debes fijarte en esta columna.  
 

--- TAREA PRINCIPAL ---
1. Analizar los datos agregados para obtener los valores de los indicadores.
2. Crear un resumen detallado, conciso y articulado de la zona basado en los datos reales. Para ello, debes utilizar el archivo tabular.
Debes tomar las llaves del diccionario suministrado para determinar cuáles indicadores están en el diccionario. Una vez los identifiques, para poder saber cómo traducir esos indicadores a 
lenguaje natural, debes identificar los valores en las columnas Nombre y Label del archivo tabular que corresponden a cada indicador en el diccionario. 
3. Contextualiza los resultados de manera que sean comprensibles para el usuario. Para ello, debes tomar el valor de cada indicador en la base agregada y comprarlo con la columna de Referencia en el archivo tabular. 
Este valor te servirá para saber si el valor en la base agregada es mayor o menor que el valor de referencia nacional. 

--- REGLAS DE LÓGICA Y OUTPUT ---
1. Tu resumen debe ser conciso, informativo y articulado.
2. Redacta un párrafo articulado que sea claro, en un tono técnico y formal.
3. No des respuestas muy extensas, máximo 400 palabras.
4. Incluye los valores numéricos redondeados al número entero más cercano.
5. Al mencionar los indicadores, NO digas el nombre técnico (que está en la columna Nombre), sino solo los valores en lenguaje natural en la clumna Label
6. Siempre incluye al final un breve resumen que caracterice el área.

--- EJEMPLOS DEL DICCIONARIO Y CÓMO DEBES LEERLO---
El diccionario tendrá la siguiente estructura:
Por ejemplo: 
{
    {
        ("TP27_PERSO", "sum") : "12345", 
        ("TP27_PERSO", "mean") : "897542",
        ("TP51SUPERI", "min"): "0"
    }
}

Los indicadores son "TP27_PERSO" y "TPSUPERI51". 
Para "TP27_PERSO", se hicieron dos operaciones: 
    - "sum", es decir, la suma, con un valor de 12345. 
    - "mean", es decir, el promedio, con un valor de 987542. 

Para "TP51SUPERI" se hizo solo una operación:
    - "min", es decir, "minimo, con un valor de 0. 

--- EJEMPLOS DE CONSULTAS ---

Query del usuario: ¿Cuál es la suma de TP27_PERSO? y ¿Cuál es el promedio de TP51SUPERI? y ¿Cuál es el promedio de TP15_1_OCU? y ¿Cuál es el promedio de TP19_EE_E4? 
Diccionario: 
{
    {
        ("TP27_PERSO", "sum") : "100", 
        ("TP51SUPERI", "mean") : "90.53", 
        ("TP15_1_OCU", "mean"): "72",
        ("TP19_EE_E4", "mean"): "20"
    }
}

Razonamiento (sigue este razonamiento pero no lo incluyas en tu respuesta): 
(TP27_PERSO, "sum") --> Indicador (siempre primero en la tupla): TP27_PERSO. Agregación (siempre segundo en la tupla): "sum" ---> Nombre en archivo tabular: TP27_PERSO --> Label en archivo tabular: Total Personas
TP51SUPERI --> Nombre en archivo tabular: TP51SUPERI--> Label en archivo tabular: Educación superior
TP15_1_OCU --> Nombre en archivo tabular: TP15_1_OCU --> Label en archivo tabular: Edad promedio
TP19_EE_E4 --> Nombre en archivo tabular: TTP19_EE_E4--> Label en archivo tabular: Población migrante

Valores de referencia a partir de la columna Referencia en el archivo tabular: 
TP27_PERSO --> 500---> El valor de TP27_PERSO de 100 en el diccionario es menor que 500
TP51SUPERI --> 80 ---> El valor de TP51SUPERI de 90.53 en el diccionario es mayor
TP15_1_OCU --> 70 ---> El valor de TP15_1_OCU de 72 en el diccionario es muy similar
TP19_EE_E4 -- > 50 ---> El valor de TP19_EE_E4 de 20 en el diccionario mucho menor

Respuesta: 
En esta área, marcada por una isocrona con un intervalo de 10 minutos, el número total de personas es de 100, mucho menor que la referencia nacional. En promedio, 
la tasa de personas con educación superior es de 90.53, mucho mayor que el referente nacional de 80. Además, en promedio la edad es de 72, cercana al valor nacional. Por último, 
hay 20 personas migrantes, lo cual representa un valor muy bajo respecto al valor nacional de 50. 



    """

# Updated prompt for the coding agent that works with any client data
def create_prompt_agente_codigo_client():
    return """
--- ROL ---
Eres un analista de datos especializado en procesar dataframes en Python. 
Tu tarea es, dada una o más preguntas y un dataframe ya cargado, realizar las agregaciones 
correspondientes para responder a las preguntas.

--- INPUTS ---
Recibirás los siguientes inputs:
    1. Una lista de preguntas estructuradas del usuario
    2. Un dataframe llamado df que ya contiene todos los datos necesarios, incluyendo:
        - Una columna llamada "Intervalo" (por la cual siempre debes agrupar)
        - Las columnas de indicadores mencionadas en las preguntas
        - Los datos ya están intersectados con el área geográfica de interés

--- PROCESOS ---
Sigue estos pasos: 
    1. Analiza cada pregunta en la lista para extraer:
        1.1 El indicador: está SIEMPRE ubicado entre paréntesis ()
        1.2 La función de agregación: está SIEMPRE entre el separador ||
        
        Importante: La función de agregación siempre debe ser una operación matemática o estadística 
        (suma, promedio, mediana, máximo, mínimo, etc.). Si ves casos como "la temperatura máxima", 
        entiende que la función relevante es "máximo" (la función estadística).
    
    2. Aplica las agregaciones correspondientes al dataframe proporcionado:
        - SIEMPRE agrupa por la columna "Intervalo"
        - Usa siempre la función .agg() de pandas con un diccionario en donde las llaves son los indicadores y los valores
        son una lista con las agregaciones solicitadas para cada indicador. 
        - Usa las funciones de pandas apropiadas ("sum", "mean", "median", "max", "min", etc.)
        
        Ejemplos de agregaciones:
        - Una sola función y varios indicadores: df.groupby(["Intervalo"]).agg({'indicador1': ['sum'], 'indicador2': ['sum']})
        - Múltiples funciones y varios indicadores: df.groupby(["Intervalo"]).agg({'indicador1': ['sum'], 'indicador2': ['mean']})
        - Una función y un indicador: df.groupby(["Intervalo"]).agg({'indicador1': ['sum']})

    3. IMPORTANTE: Siempre retorna el resultado completo de la agregación como un DataFrame.
       NO retornes valores individuales, sumas, o números sueltos.
       El resultado DEBE ser un DataFrame con columnas multi-nivel que pueda procesarse posteriormente. 

--- EJEMPLOS ---
    1. Input: 
       - Preguntas: ["¿Cuál es |la suma| del indicador (TP27_PERSO)?"]
       - DataFrame: Ya cargado con columnas ["Intervalo", "TP27_PERSO", ...]
       
       Proceso:
       - Indicador: TP27_PERSO
       - Agregación: suma
       - Código: df.groupby(["Intervalo"]).agg({'TP27_PERSO': ['sum']})
       
    2. Input:
       - Preguntas: ["¿Cuál es |la suma| del indicador (cliente_ind_1)?", "¿Cuál es |el promedio| del indicador (cliente_ind_2)?"]
       - DataFrame: Ya cargado con columnas ["Intervalo", "cliente_ind_1", "cliente_ind_2", ...]
       
       Proceso:
       - Indicadores: cliente_ind_1 (suma), cliente_ind_2 (promedio)
       - Código: df.groupby(["Intervalo"]).agg({'cliente_ind_1': ['sum'], 'cliente_ind_2': ['mean']})

--- REGLAS IMPORTANTES ---
- NO intentes cargar datos de archivos externos
- El dataframe YA está disponible y listo para procesar
- SIEMPRE agrupa por "Intervalo"
- Retorna solo el resultado de la agregación, sin explicaciones adicionales
- Los nombres de indicadores pueden variar según los datos del cliente
- Adapta tu código a los nombres de columnas que encuentres en el dataframe
"""

# --------------------------------------------------------- Funciones reactivas --------------------------------------------------
# --- Selección de área 
@reactive.effect
@reactive.event(input.area_selector)
def handle_area_selection():
    area_type = input.area_selector()
    if area_type == "Equidistancia":
        pass  

# --- Selección de medio de transporte
@reactive.calc
def get_transport_mode():
    """Get the transport mode, handling the invisible setting for Equidistancia"""
    area_type = input.area_selector()
    if area_type == "Equidistancia":
        return "Vehículo"  
    else:
        try:
            return input.trans_selector()  
        except AttributeError:
            return "Vehículo"  

# --- Acción reactiva para obtener el filtro del municipio seleccionado
@reactive.calc
def muni_filter():
    filter_muni_name = input.muni_selector()
    filter_muni_code = muni_code_dict.get(filter_muni_name)
    filtered_shp = munip_shp[munip_shp["MPIO_CDPMP"].isin([filter_muni_code])]
    filtered_shp = gpd.GeoDataFrame(filtered_shp, geometry="geometry", crs="EPSG:4326") 
    return filtered_shp


# --- Acción reactiva para inicializar el mapa de la isocrona
@reactive.effect
@reactive.event(input.create_area_button)
def update_plot_iso():
    plot_iso_bool.set(True)

# --- Acción reactiva para obtener las coordenadas tras un click
@reactive.calc()
def get_coords():
    coords = clicked_coords.get()
    if coords: 
        return coords
    return False

@reactive.effect
@reactive.event(input.clear_coords)
def clear_clicked_coordinates():
    clicked_coords.set(None)
    ui.update_text("address_input", value="")

# --- Acción reactiva para geocodificar la dirección
@reactive.effect
@reactive.event(input.geo_button)
def update_coords():
    direccion = input.address_input()
    if len(direccion) == 0:
        cand_dir_check.set(None)
        return False   
    
    # ------  Añadir chequeo de direcciones en base de datos --------
    
    # Solo se activa si no hay coordenadas seleccionadas y si hay alguna dirección propuesta
    mp_geocoder = geopy.geocoders.MapBox(api_key=API_D.get("isocrona"))
    muni_shp_filtered = muni_filter()
    centroid_c = shapely.get_coordinates(muni_shp_filtered.to_crs("EPSG:4326").centroid)[0]
    try: 
        dir_coords = mp_geocoder.geocode(
                query=direccion,   
                exactly_one=False, 
                country="CO", 
                language="CO", 
                proximity=(centroid_c[1], centroid_c[0])
                )
        if dir_coords:
            coords_dict = {coords.address:(coords.point.latitude, coords.point.longitude) for coords in dir_coords}
            cand_dir_check.set(coords_dict)
    except:
        cand_dir_check.set(coords_dict)
    
    return False

# --- Acción reactiva para obtener la dirección seleccionada
@reactive.calc
def dir_to_coords():
    return cand_dir_check.get()

# --- Acción reactiva para seleccionar las coordenadas finales
@reactive.calc
def select_final_coords():
    # Priorizar selección manual
    coord_click = clicked_coords.get()
    if coord_click:
        return {"sel_coords": (coord_click[0], coord_click[1])}

    # Selección de direcciones candidatas
    try:
        candidate_address = input.candidate_selector()
        if candidate_address and candidate_address != "None":
            coords_dict = dir_to_coords()
            if coords_dict and coords_dict != "None":
                coords = coords_dict[candidate_address] 
                return {candidate_address: coords}
    except: 
        pass
    return None


# ---------------------------------------------------------- Agentes ---------------------------------------------------------------
# --- Agente planificador
def create_planning_agent(client_data_context, main_dict_content, indicators_dict):
    """Create the planning agent with client's data context"""
    
    prompt = create_planning_agent_prompt(client_data_context, main_dict_content, indicators_dict)
    
    planning_client = ChatOpenAI(
        api_key=API_OPENAI,
        model="gpt-4.1",
        system_prompt=prompt
    )
    return planning_client

# --- Agente verificador
def create_verifying_agent(client_data_context, main_dict_content):
    """Create the verifying agent with client's data context"""

    verifying_client = ChatOpenAI(
        api_key=API_OPENAI,
        model="gpt-4.1", 
        system_prompt=create_verifying_agent_prompt(client_data_context, main_dict_content)
    )
    return verifying_client

# --- Agente de interpretación final de los datos
def create_interpretation_agent(main_dict_content, client_data_context):
    """Create the interpretation agent with client's data context"""
    prompt = create_prompt_agente_interpretacion(main_dict_content, client_data_context)
    
    interpretation_client = ChatOpenAI(
        api_key=API_OPENAI,
        model="gpt-4o-mini", 
        system_prompt=prompt
    )
    
    return interpretation_client

# --- Agente de creación de preguntas
def create_modified_query_interpreter_agent(client_data_context, main_dict_content):
        
    modified_interpreter_client = ChatOpenAI(
        api_key=API_OPENAI,  
        model="gpt-4o-mini",
        system_prompt=create_agente_etiquetado_prompt(client_data_context, main_dict_content)
    )
    return modified_interpreter_client


# --- Agente de código
def create_code_agent_client():
    """Create the coding agent with updated prompt for client data"""
    
    # Create the model for the coding agent
    model = OpenAIModel(
        model_id="gpt-4o-mini",
        api_key=API_OPENAI
    )
    
    # Create the coding agent with updated prompt
    code_agent = CodeAgent(
        model=model,
        tools=[],
        instructions=create_prompt_agente_codigo_client(), 
        additional_authorized_imports=['pandas'],
        add_base_tools=True
    )
    
    return code_agent

# --- Función para referencias
# def prepare_reference_data(client_data_filtered_df):
#     df = client_data_filtered_df.copy()
#     df.drop(["hexagon_id"], axis=1, inplace=True)
#     df_ag = df.sum().reset_index()
#     df_ag.columns = ["Indicador", "Referencia"]
#     print("Resumen agregado:")
#     print(df_ag.head())
#     return df_ag
def prepare_reference_data(client_data_filtered_df, selected_indicators=None):
    """
    Prepare reference data for specific indicators or all indicators
    
    Args:
        client_data_filtered_df: DataFrame with client data
        selected_indicators: List of specific indicators to include, if None includes all
    
    Returns:
        DataFrame with indicator names and reference values
    """
    df = client_data_filtered_df.copy()
    df.drop(["hexagon_id"], axis=1, inplace=True)
    
    # Filter to only selected indicators if provided
    if selected_indicators:
        # Only keep columns that are in selected_indicators
        available_indicators = [col for col in selected_indicators if col in df.columns]
        if available_indicators:
            df = df[available_indicators]
        else:
            # Return empty DataFrame if no indicators match
            return pd.DataFrame(columns=["Indicador", "Referencia"])
    
    df_ag = df.sum().reset_index()
    df_ag.columns = ["Indicador", "Referencia"]
    print("Resumen agregado (indicadores específicos):")
    print(df_ag.head())
    return df_ag
    


# --- Función orquestradora
def handle_chat_input_new_architecture(main_dict_content, indicators_dict, client_data_context = "[]"):
    
    # Obtener mensaje del usuario
    user_message = input.chat_input()
    
    if not user_message.strip():
        return

    # Cargar los archivos del usuario
    client_res = get_uploaded_file_data()
    client_data_filtered = None
    if client_res: 
        client_data = client_res.get("data")
        client_dict = client_res.get("dict")

        if isinstance(client_data, pd.DataFrame) and isinstance(client_dict, pd.DataFrame): 
            
            # Procesar archivos del clientes

            # ANALYSIS_COLS = ["hta", "dm", "epoc"]
            # GEO_REF_COLS = ["latitude", "longitude"]
            # CAT_COLS = ["SEXO", "REGIMEN"]
            # COUNT_COLS = ["ID_GRUPO_FAMILIAR"]
            # ID_COLS = ["ID_AFILIADO"]

            var_categories = extract_col_categories(client_dict)
            print(var_categories)
            ANALYSIS_COLS = var_categories.get("analysis")
            GEO_REF_COLS = var_categories.get("geo_ref")
            CAT_COLS = var_categories.get("categorical")
            COUNT_COLS = var_categories.get("grouping")
            ID_COLS = var_categories.get("id")
            print(CAT_COLS)

            client_data_filtered = prepare_client_data(
                df=client_data, 
                selected_analysis_cols=ANALYSIS_COLS, 
                geo_ref_cols=GEO_REF_COLS, 
                cat_cols=CAT_COLS, 
                count_cols=COUNT_COLS, 
                id_cols=ID_COLS
            )
            
            print("Datos procesados:")
            print(client_data_filtered.head())

    
            client_data_context = update_dict(
                processed_df=client_data_filtered, 
                data_dict=client_dict, 
                cat_cols=CAT_COLS
            )

            print("Diccionario nuevo de datos:")
            print(client_data_context)
        else: 
            client_data_context = "[]"
            print(client_data_context)
    else: 
        client_data_context = "[]"
        print("No hay datos del cliente: procesando información pública")

    
    # Iniciar proceso con agentes
    ai_processing.set(True)
    
    try:
        print("🚀 Starting new 4-agent workflow...")
        
        # Procesar la consulta con los 3 agentes: planificación, subpreguntas y verificación 
        response_dict = process_ai_query_new_architecture(user_message, main_dict_content=main_dict_content, indicators_dict=indicators_dict, client_data_context=client_data_context)
        
        if response_dict["success"]:
            selected_indicators = response_dict["inds"]
            subqueries = response_dict["subqueries"]
            
            print(f"✅ Flujo completado")
            print(f"📊 Indicadores: {selected_indicators}")
            print(f"❓ Sub preguntas: {subqueries}")
            print(f"🎯 Resultado intento: {response_dict['final_attempt']}")
            
            if selected_indicators and subqueries:
                # Necesitamos hacer un set para llamar a la base de datos
                ai_selected_indicators.set(selected_indicators)
                
                # Preparar datos y llamar a los agentes de código e interpretación
                print("Preparando datos...")
                main_data_int = prepare_iso_with_data()
                if isinstance(main_data_int, bool):
                    print("No se han solicitado datos de información pública.")
                else:
                    print("Datos públicos listos")

                if client_data_context != "[]":
                    try:
                        main_data_client = prepare_iso_with_client_data(client_data_filtered)
                        print(f"Tipo de main_data_client: {type(main_data_client)}")
                        
                        if isinstance(main_data_client, pd.DataFrame) and not main_data_client.empty:
                            print(f"main_data_client shape: {main_data_client.shape}")
                            
                            # Ambos dataframes deben existir para el merge
                            if isinstance(main_data_client, pd.DataFrame) and not main_data_client.empty and isinstance(main_data_int, pd.DataFrame) and not main_data_int.empty:
                                print(main_data_int.columns)
                                print(main_data_client.columns)
                                main_data_int_f = pd.merge(main_data_int, main_data_client, on=["Intervalo", "hexagon_id"], how="inner")
                                print("Merge exitoso con datos del cliente")
                            elif isinstance(main_data_int, bool): # Si es booleano, es porque la función de creación de la base pública retornó False, lo cual indica que no se han solicitado datos públicos
                                main_data_int_f = main_data_client
                                print("Utilizando solo datos del cliente")
                        else:
                            main_data_int_f = main_data_int
                            
                            
                    except Exception as e: 
                        if isinstance(main_data_int, pd.DataFrame):
                            main_data_int_f = main_data_int
                        else: 
                            main_data_int_f = main_data_client

                        print(f"Error haciendo merge: {e}")
                        print(f"Tipo de excepción: {type(e).__name__}")
                else: 
                    main_data_int_f = main_data_int
                        
                if isinstance(main_data_int_f, pd.DataFrame):
                    print("Llamando al agente de datos...")
                    aggregated_result = aggregate_iso_with_data(main_data_int_f, subqueries)
                    print("✅ Datos agregados al nivel de isocrona")

                    # Final interpretation (unchanged)
                    area_info = {
                        "tipo_area": input.area_selector(),
                        "intervalo": input.interval_selector(),
                        "transporte": get_transport_mode()
                    }

                    print("Lllamar al agente de interpretación")
                    final_summary = generate_area_summary(aggregated_result, user_message, area_info, main_dict_content, client_data_context)
                    
                    # Metadatos --- solo para hacer verificaciones
                    workflow_info = f"\n\n🔧 Procesado con arquitectura de 4 agentes (intento {response_dict['final_attempt']}/2)"
                    final_summary_with_metadata = final_summary + workflow_info
                    final_ai_summary.set(final_summary_with_metadata)
                    
                    # Borrar valores anteriores en el chat
                    chat_messages.set("")
                    
                else:
                    final_ai_summary.set("❌ Error: No se pudieron obtener los datos del área seleccionada.")
                    
            else:
                final_ai_summary.set("❌ No se encontraron indicadores válidos para su consulta. Por favor, reformule su pregunta.")
        
        else:
            # En caso de fallos en la verificación
            error_msg = f"⚠️ El proceso de validación no fue completamente exitoso"
            if response_dict.get("inds") and response_dict.get("subqueries"):
                error_msg += f", pero se procesará con los últimos resultados generados."
                
                # Seguir el proceso con lo generado
                ai_selected_indicators.set(response_dict["inds"])
                main_data_int = prepare_iso_with_data()
                
                if isinstance(main_data_int, pd.DataFrame):
                    aggregated_result = aggregate_iso_with_data(main_data_int, response_dict["subqueries"])
                    
                    area_info = {
                        "tipo_area": input.area_selector(),
                        "intervalo": input.interval_selector(),
                        "transporte": get_transport_mode()
                    }
                    
                    final_summary = generate_area_summary(aggregated_result, user_message, area_info)
                    final_summary_with_warning = f"⚠️ {error_msg}\n\n{final_summary}"
                    final_ai_summary.set(final_summary_with_warning)
                    chat_messages.set("")
                else:
                    final_ai_summary.set(f"{error_msg} Además, no se pudieron obtener los datos del área.")
            else:
                error_msg += f". Error: {response_dict.get('error', 'Unknown error')}"
                final_ai_summary.set(error_msg)
    
    except Exception as e:
        error_msg = f"❌ Error procesando la consulta con nueva arquitectura: {str(e)}"
        final_ai_summary.set(error_msg)
        print(error_msg)
    
    finally:
        ai_processing.set(False)
        ui.update_text_area("chat_input", value="")

# Updated chat handler function
# def handle_chat_input_new_architecture_updated(client_data_context, main_dict_content, indicators_dict):
#     """
#     Updated chat handler that uses client data throughout the entire pipeline
#     """
    
#     # Get user message
#     user_message = input.chat_input()
    
#     if not user_message.strip():
#         return

#     # Load client files
#     client_res = get_uploaded_file_data()
#     if not client_res:
#         final_ai_summary.set("❌ Error: No se han cargado archivos válidos. Por favor, sube los archivos necesarios.")
#         return
        
#     client_data = client_res.get("data")
#     client_dict = client_res.get("dict")
    
#     if client_data is None or client_dict is None:
#         final_ai_summary.set("❌ Error: Los archivos cargados no tienen el formato correcto. Asegúrate de incluir datos y diccionario de datos.")
#         return

#     # Process client files
#     var_categories = extract_col_categories(client_dict)
#     ANALYSIS_COLS = var_categories.get("analysis")
#     GEO_REF_COLS = var_categories.get("geo_ref")
#     CAT_COLS = var_categories.get("categorical")
#     COUNT_COLS = var_categories.get("grouping")
#     ID_COLS = var_categories.get("id")

#     client_data_filtered = prepare_client_data(
#         df=client_data, 
#         selected_analysis_cols=ANALYSIS_COLS, 
#         geo_ref_cols=GEO_REF_COLS, 
#         cat_cols=CAT_COLS, 
#         count_cols=COUNT_COLS, 
#         id_cols=ID_COLS
#     )
#     print("Datos del cliente procesados:")
#     print(client_data_filtered.head())

#     client_data_context = update_dict(
#         processed_df=client_data_filtered, 
#         data_dict=client_dict, 
#         cat_cols=CAT_COLS
#     )

#     print("Diccionario de datos del cliente:")
#     print(client_data_context)
    
#     # Start AI processing
#     ai_processing.set(True)
    
#     try:
#         print("🚀 Iniciando workflow de 4 agentes con datos del cliente...")
        
#         # Process query with client data throughout the pipeline
#         response_dict = process_ai_query_new_architecture(user_message, client_data_context, main_dict_content, indicators_dict)
        
#         if response_dict["success"]:
#             selected_indicators = response_dict["inds"]
#             subqueries = response_dict["subqueries"]
            
#             print(f"✅ Flujo con datos del cliente completado")
#             print(f"📊 Indicadores: {selected_indicators}")
#             print(f"❓ Sub preguntas: {subqueries}")
#             print(f"🎯 Resultado intento: {response_dict['final_attempt']}")
            
#             if selected_indicators and subqueries:
#                 ai_selected_indicators.set(selected_indicators)
                
#                 # Prepare data using client data
#                 print("Preparando datos del cliente...")
#                 iso_geometry = create_isochrone_geometry()  # This function remains the same
                
#                 if isinstance(iso_geometry, gpd.GeoDataFrame):
#                     main_data_int = prepare_iso_with_data(
#                         client_data_filtered=client_data_filtered
#                     )
                    
#                     if isinstance(main_data_int, pd.DataFrame):
#                         print("Llamando al agente de código con datos del cliente...")
#                         # The coding agent remains the same as it works with any DataFrame
#                         aggregated_result = aggregate_iso_with_client_data(main_data_int, subqueries)
#                         print("✅ Datos del cliente agregados al nivel de isocrona")

#                         # Final interpretation using client data context
#                         area_info = {
#                             "tipo_area": input.area_selector(),
#                             "intervalo": input.interval_selector(),
#                             "transporte": get_transport_mode()
#                         }

#                         print("Llamando al agente de interpretación con contexto del cliente")
#                         interpretation_agent = create_interpretation_agent(client_data_context)
                        
#                         interpretation_prompt = f"""
#                         Consulta original del usuario: "{user_message}"
                        
#                         Información del área:
#                         - Tipo de área: {area_info.get('tipo_area', 'N/A')}
#                         - Intervalo: {area_info.get('intervalo', 'N/A')} {'minutos' if area_info.get('tipo_area') == 'Isocrona' else 'metros'}
#                         - Transporte: {area_info.get('transporte', 'N/A')} (si aplica)
                        
#                         Diccionario con los datos de los indicadores:
#                         {aggregated_result}
                        
#                         Por favor, genera un resumen natural y comprensible de esta área basado en los datos proporcionados del cliente.
#                         Incluye los valores específicos y contextualiza su significado.
#                         """
                        
#                         try:
#                             final_response = interpretation_agent.chat(interpretation_prompt)
#                             final_summary = final_response.content
#                         except Exception as e:
#                             final_summary = f"Error al generar el resumen con datos del cliente: {str(e)}"
                        
#                         # Add metadata
#                         workflow_info = f"\n\n🔧 Procesado con arquitectura de 4 agentes usando datos del cliente (intento {response_dict['final_attempt']}/2)"
#                         final_summary_with_metadata = final_summary + workflow_info
#                         final_ai_summary.set(final_summary_with_metadata)
                        
#                         # Clear chat
#                         chat_messages.set("")
                        
#                     else:
#                         final_ai_summary.set("❌ Error: No se pudieron procesar los datos del cliente para el área seleccionada.")
#                 else:
#                     final_ai_summary.set("❌ Error: No se pudo crear la geometría del área seleccionada.")
                    
#             else:
#                 final_ai_summary.set("❌ No se encontraron indicadores válidos en los datos del cliente para su consulta.")
        
#         else:
#             # Handle validation failures
#             error_msg = f"⚠️ El proceso de validación con datos del cliente no fue completamente exitoso"
#             if response_dict.get("inds") and response_dict.get("subqueries"):
#                 error_msg += f", pero se procesará con los últimos resultados generados."
                
#                 # Continue with generated results
#                 ai_selected_indicators.set(response_dict["inds"])
#                 iso_geometry = create_isochrone_geometry()
                
#                 if isinstance(iso_geometry, gpd.GeoDataFrame):
#                     main_data_int = prepare_iso_with_client_data(
#                         client_data_filtered=client_data_filtered
#                     )
                    
#                     if isinstance(main_data_int, pd.DataFrame):
#                         aggregated_result = aggregate_iso_with_client_data(main_data_int, subqueries)
                        
#                         area_info = {
#                             "tipo_area": input.area_selector(),
#                             "intervalo": input.interval_selector(),
#                             "transporte": get_transport_mode()
#                         }
                        
#                         interpretation_agent = create_interpretation_agent(client_data_context)
#                         interpretation_prompt = f"""
#                         Consulta original del usuario: "{user_message}"
#                         Área: {area_info}
#                         Datos: {aggregated_result}
#                         Genera un resumen basado en los datos del cliente.
#                         """
                        
#                         try:
#                             final_response = interpretation_agent.chat(interpretation_prompt)
#                             final_summary = final_response.content
#                             final_summary_with_warning = f"⚠️ {error_msg}\n\n{final_summary}"
#                             final_ai_summary.set(final_summary_with_warning)
#                         except Exception as e:
#                             final_ai_summary.set(f"{error_msg} Error adicional en interpretación: {str(e)}")
                        
#                         chat_messages.set("")
#                     else:
#                         final_ai_summary.set(f"{error_msg} Además, no se pudieron obtener los datos del área.")
#                 else:
#                     final_ai_summary.set(f"{error_msg} No se pudo crear la geometría del área.")
#             else:
#                 error_msg += f". Error: {response_dict.get('error', 'Unknown error')}"
#                 final_ai_summary.set(error_msg)
    
#     except Exception as e:
#         error_msg = f"❌ Error procesando la consulta con datos del cliente: {str(e)}"
#         final_ai_summary.set(error_msg)
#         print(error_msg)
    
#     finally:
#         ai_processing.set(False)
#         ui.update_text_area("chat_input", value="")



# # -- Procesador de la consulta
# #@reactive.calc
# def process_ai_query(user_message):
#     if not user_message or not ai_processing.get():
#         print(user_message)
#         print(ai_processing.get())
#         return {"inds": [], "subqueries": []}  # Return dict instead of list
    
#     try:
#         ai_processing.set(True)

#         # Enviar consulta al agente
#         response = chat_client.chat(user_message)

#         # Obtener la respuesta
#         agent_resp = response.content

#         # Extraer indicadores
#         indicators = re.findall(r"\((.*?)\)", agent_resp)

#         valid_indicators = [ind for ind in indicators if ind and ind != "None"]
#         ai_processing.set(False)

#         # Extraer las subpreguntas
#         subqueries = re.findall(r"\¿.*?\?", agent_resp)

#         # Crear diccionario de respuestas
#         res_dict = {
#             "inds": valid_indicators, 
#             "subqueries": subqueries
#         }

#         print(res_dict)
#         return res_dict

#     except Exception as e:
#         ai_processing.set(False)
#         print(f"Error de procesamiento del agente de IA: {e}")
#         return {"inds": [], "subqueries": []}  # Return dict instead of None



final_ai_summary = reactive.Value("")

# --- Función orquestradora
@reactive.effect
@reactive.event(input.send_query)
def handle_chat_input():
    global main_dict_content, INDICATORS_DICT
    handle_chat_input_new_architecture(main_dict_content, INDICATORS_DICT)

# @reactive.effect
# @reactive.event(input.send_query)
# def handle_chat_input():
#     # Load the data directly
#     main_dict_df = pd.read_excel(MAIN_DICT_RAW)
#     main_dict_content = main_dict_df.to_string(index=False)
    
#     with open(os.path.join(INPUT_PARAMS_DIR, "nombres_variables.json")) as f:
#         indicators_dict = json.load(f)
    
#     handle_chat_input_new_architecture(main_dict_content, indicators_dict)

# @reactive.effect
# @reactive.event(run_iso_bool, ai_selected_indicators)
# def update_chat_with_summary():
#     if ai_summary_ready.get():
#         return 


#     run_iso = run_iso_bool.get()
#     selected_indicators = ai_selected_indicators.get()
       
#     if not run_iso or not selected_indicators:
#         return

#     """Update chat with AI-generated summary when data is ready"""
#     summary = generate_ai_summary()
#     if summary and summary != "":
#         #current_messages = chat_messages.get()
#         #current_messages += f"\n\n🤖 Resumen: {summary}"
#         #chat_messages.set(current_messages)
#         ai_summary_ready.set(True)

# def reset_ai_state():
#     ai_summary_ready.set(False)
#     ai_processing.set(False)
#     ai_selected_indicators.set([])

# @render.text
# def final_candidate_display():
#     # Check if we have clicked coordinates first
#     clicked = clicked_coords.get()
#     if clicked:
#         lat, lon = clicked[0], clicked[1]
#         return f"Ubicación confirmada: ({lat:.4f}, {lon:.4f})"
    
#     # Otherwise check selected candidate
#     selected_candidate_address = input.candidate_selector()
#     candidates = dir_to_coords()
#     if candidates and selected_candidate_address != "None":
#         if selected_candidate_address in candidates:
#             lat, lon = candidates[selected_candidate_address]
#             return f"Ubicación confirmada: ({lat:.4f}, {lon:.4f})"
        
#     return "Ingrese una dirección o haga click en el mapa para obtener las coordenadas."



# Add an observer to update the candidate selector choices
# @reactive.effect
# @reactive.event(input.geo_button)
# def update_candidate_choices():
#     candidates = dir_to_coords()
#     if candidates:
#         choices = list(candidates.keys())
#         ui.update_select("candidate_selector", choices=choices, selected=choices[0] if choices else "None")

#     else:
#         ui.update_select("candidate_selector", choices=["None"], selected="None") 

#@reactive.effect
#@reactive.event(clicked_coords.get)
#def clear_candidate_choices():
#    ui.update_select("candidate_selector", choices=["None"], selected="None") 

# ==================================================== Visualización del mapa ==================================================
# --- Acción reactiva para crear la geometría de la isocrona
@reactive.calc
def create_isochrone_geometry():
    """Create just the isochrone geometry - fast operation"""
    run_iso = plot_iso_bool.get()
    if not run_iso:
        return False
        
    # Basic inputs for geometry
    delimitador = input.area_selector()
    intervalo = input.interval_selector()
    trans = get_transport_mode()
    final_coords_dict = select_final_coords()
    
    if not all([delimitador, intervalo, trans, final_coords_dict]):
        return False
        
    try:
        final_coords = list(final_coords_dict.values())[0]
        print("Creando geometría de isocrona")
        
        # Create just the isochrone geometry
        iso_geometry = iso_wrapper(
            coords=[[final_coords[1], final_coords[0]]], 
            intervalo=intervalo, 
            tipo=delimitador, 
            modo_transporte=trans, 
            penalty=0.6, 
            api_dict=API_D
        )
        
        print("Geometría de isocrona creada")
        return iso_geometry
        
    except Exception as e:
        print(f"Error creando geometría: {e}")
        return False

# --- Acción reactiva para actualizar al mapa
@reactive.calc
def current_map_view():
    # 1. Priority: Manual Map Click Coordinates (or cleared by address selection)
    #final_coords_click = clicked_coords.get()
    #if final_coords_click:
    #    return {"center": (final_coords_click[0], final_coords_click[1]), "zoom": 12} 
    
    # 2. Second Priority: Selected Geocode Candidate Coordinates
    final_coords_select = select_final_coords() # <--- Dependency on address selection
    if final_coords_select:
        coords_select = list(final_coords_select.values())[0]
        return {"center": (coords_select[0], coords_select[1]), "zoom": 12}
        
    # 3. Fallback: Municipality Centroid (MUST be robust)
    sel_muni = muni_filter()
    
    if not sel_muni.empty:
        centroid_point = sel_muni.geometry.centroid.iloc[0] 
        center_lat = centroid_point.y
        center_lon = centroid_point.x
        return {"center": (center_lat, center_lon), "zoom": 10}
    
    # 4. Final Absolute Default
    return {"center": (4.5709, -74.2973), "zoom": 5}

# @reactive.effect
# @reactive.event(input.go_button)
# def update_run_iso_bool():
#     run_iso_bool.set(True)

# @render.text
# def summarize_df():
#     run_iso = run_iso_bool.get()
#     if run_iso: 
#         main_df = create_iso()
#         if type(main_df) is bool:
#             return
#         # Inputs necesarios
#         # --- Tipo de área
#         delimitador = input.area_selector()
#         if delimitador == "Isocrona":
#             interval_id = "minutos"
#         else: 
#             interval_id = "metros"

#         # --- Intervalo
#         intervalo = input.interval_selector()
#         if type(intervalo) is not int:
#             intervalo = int(intervalo)
#         # --- Transporte
#         trans = input.trans_selector()
#         if trans == "Vehículo":
#             trans_id = "en carro"
#         else: 
#             trans_id = "a pie"
#         # --- Indicadores
#         inds = input.ind_selector()
#         inds = [ind for ind in inds if ind != "" or ind is not None]
#         if len(inds) == 1:
#             ind_prompt = f"el indicador {RENAME_DICT.get(inds[0])} es de"
#             ind_val = round(main_df[inds].iat[0, 0], 2)
#         elif len(inds) == 2:
#             ind_prompt = f"los indicadores {RENAME_DICT.get(inds[0])} y {RENAME_DICT.get(inds[1])} son de"
#             ind_val = f"{round(main_df[inds[0]].iat[0, 0], 2)} y {round(main_df[inds[1]].iat[0, 0], 2)}, respectivamente."

#         return f"En la {delimitador} de {intervalo} {interval_id} {trans_id}, {ind_prompt} {ind_val}."




# # Keep create_iso for backward compatibility, but now it uses the data version
# @reactive.calc
# def create_iso():
#     """Returns isochrone with data - used by AI summary"""
#     return create_iso_with_data()


@reactive.calc
def create_map():
    def handle_map_click(**kwargs):
        """Captures a map click event and stores the coordinates."""
        if kwargs.get("type") == "click":
            if 'coordinates' in kwargs:
                coordinates = kwargs['coordinates']
                clicked_coords.set([coordinates[0], coordinates[1]])
            elif 'latlng' in kwargs:
                # Alternative format
                coordinates = kwargs['latlng']
                clicked_coords.set([coordinates[0], coordinates[1]])

            ui.update_text("address_input", value="")

    
    # Definir la capa base del mapa
    view_state = current_map_view()
    m = Map(
        center=view_state["center"], 
        zoom=view_state["zoom"]
    )

    # Attach the click handler to the map
    m.on_interaction(handle_map_click)

    plot_iso = plot_iso_bool.get()
    if plot_iso:
        final_dataframe = create_isochrone_geometry() 
        if type(final_dataframe) is bool:
            return m
        
        # Capa del polígono
        pol_layer = GeoData(
            geo_dataframe = final_dataframe, 
            style={
                "color": "blue", 
                "fillColor": "lightblue", 
                "weight": 2,
                "fillOpacity": 0.6
            }, 
            name = "Area cubierta"
        )

        m.add_layer(pol_layer)

        # Capa del marcador
        final_coords = select_final_coords()
        lat, lon = list(final_coords.values())[0][0], list(final_coords.values())[0][1]
        candidate_layer = LayerGroup(name="Resultados")
        icon_f = AwesomeIcon(
            name='map-pin',  # Valid Font Awesome icon name
            marker_color='red',
            icon_color='white',
            spin=False
            )

        marker = Marker(
            location=(lat, lon), 
            draggable=False, 
            icon=icon_f
            )

        candidate_layer.add_layer(marker)
        m.add_layer(candidate_layer)
    
    else:

        # Añadir marcadores de las coordenadas candidatas si existen
        coords_candidatas = dir_to_coords()
        coords_click_sel = clicked_coords.get()
        if coords_candidatas or coords_click_sel:

            candidate_layer = LayerGroup(name="Direcciones candidatas")
            final_candidate = select_final_coords()

            f_dict = {}
            if coords_candidatas:
                for k, v in coords_candidatas.items():
                    f_dict[k] = v
            
            if coords_click_sel:
                f_dict["sel_coords"] = (coords_click_sel[0], coords_click_sel[1])

            for candidate_address, (lat, lon) in f_dict.items():
                if (final_candidate and list(final_candidate.keys())[0] == candidate_address):
                    icon_f = AwesomeIcon(
                        name='map-pin',  # Valid Font Awesome icon name
                        marker_color='red',
                        icon_color='white',
                        spin=False
                    )

                    marker = Marker(
                        location=(lat, lon), 
                        draggable=False, 
                        icon=icon_f
                    )
                else: 

                    marker = Marker(
                        location=(lat, lon),
                        draggable=False
                    )
            
                # Create a closure to capture the coordinates for this specific marker
                #def create_marker_handler(address, coordinates):
                #    def marker_click_handler(**kwargs):
                #        clicked_coords.set(list(coordinates))  # Set clicked coords to this marker's position
                #        ui.update_select("candidate_selector", selected=address)
                #    return marker_click_handler
            
                #marker.on_click(create_marker_handler(candidate_address, (lat, lon)))
                candidate_layer.add_layer(marker)

            m.add_layer(candidate_layer)   

    return m



