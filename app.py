import streamlit as st
import json
import os
import time
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Set page configuration
st.set_page_config(
    page_title="Sistema de Evaluación de Convocatorias",
    layout="wide",
    page_icon="📊"
)

# Cargar CSS personalizado desde archivo
def load_css(css_file):
    with open(css_file, 'r') as f:
        return f.read()

# Crear el archivo CSS si no existe
if not os.path.exists('style.css'):
    with open('style.css', 'w') as f:
        f.write("""/* Estilos para el sistema de evaluación */

/* Estilos generales */
.main-title {
    color: #1E88E5;
    text-align: center;
    margin-bottom: 20px;
    font-size: 2.2rem;
}

/* Indicador de procesamiento animado */
.processing-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: #ff4b4b;
    animation: pulse 1.5s infinite;
    margin: 0;
    padding: 10px;
    background-color: #fff3f3;
    border-radius: 5px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.spin-icon {
    animation: spin 2s linear infinite;
    display: inline-block;
    margin-right: 10px;
}

/* Botones */
.stButton>button {
    background-color: #1E88E5;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 5px;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #0D47A1;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transform: translateY(-2px);
}

/* Tarjetas para la información */
.info-card {
    padding: 1.5rem;
    border-radius: 10px;
    background-color: #f8f9fa;
    border-left: 5px solid #1E88E5;
    margin-bottom: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

/* Estilos para las métricas */
.metric-box {
    padding: 1rem;
    border-radius: 5px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-box.success {
    background-color: #e8f5e9;
    border-left: 5px solid #4CAF50;
}

.metric-box.warning {
    background-color: #fff8e1;
    border-left: 5px solid #FFC107;
}

.metric-box.danger {
    background-color: #ffebee;
    border-left: 5px solid #F44336;
}

/* Estilos para los expandibles */
.criterion-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.score-badge {
    padding: 5px 10px;
    border-radius: 10px;
    font-weight: bold;
    color: white;
}

.score-high {
    background-color: #4CAF50;
}

.score-medium {
    background-color: #FFC107;
}

.score-low {
    background-color: #F44336;
}""")

# Aplicar CSS personalizado
st.markdown(f"<style>{load_css('style.css')}</style>", unsafe_allow_html=True)

# Título principal con ícono
st.markdown("<h1 class='main-title'>📊 Sistema de Evaluación de Convocatorias Empresariales</h1>",
            unsafe_allow_html=True)

# Definición de criterios
criteria = {
    "Origen de la empresa": {
        "description": "La empresa debe tener su domicilio fiscal en cualquier país de América Latina que no sea Brasil.",
        "rubric": "1 - Empresa ubicada en Brasil...\n2 - Empresa registrada fuera de Brasil, pero sin prueba clara...\n3 - Empresa registrada en un país latinoamericano con inconsistencias...\n4 - Empresa registrada en un país latinoamericano distinto de Brasil, con documentación clara...\n5 - Empresa registrada en un país latinoamericano con prueba sólida de operaciones...",
        "demonstration": "Registro legal, facturación, cartera de clientes, contratos comerciales."
    },
    "Sector de actividad": {
        "description": "La empresa debe pertenecer al sector de las TIC.",
        "rubric": "1 - Empresa que no pertenece al sector de TIC...\n2 - Empresa con relación superficial con TIC...\n3 - Empresa que opera en el sector de TIC pero sin innovación...\n4 - Empresa que opera en el sector de TIC con ventaja competitiva clara...\n5 - Empresa completamente inserta en el sector de TIC, con solución innovadora...",
        "demonstration": "Uso de tecnología en el core del negocio, patentes, validación de mercado."
    },
    "Solución de base tecnológica": {
        "description": "Enmarcar la solución y/o el producto como de base tecnológica.",
        "rubric": "1 - La solución no tiene una base tecnológica clara...\n2 - La solución tiene algún componente tecnológico...\n3 - La solución tiene una base tecnológica relevante, pero poco diferenciada...\n4 - Solución basada en tecnología avanzada y diferenciada...\n5 - Solución altamente innovadora con base en conocimiento científico y tecnológico...",
        "demonstration": "Uso de tecnologías avanzadas, innovación en procesos, patentes tecnológicas."
    },
    "Etapa de desarrollo": {
        "description": "La empresa debe contar con un MVP validado y operaciones consolidadas.",
        "rubric": "1 - Sin MVP o producto conceptual...\n2 - MVP inicial sin validación comercial...\n3 - MVP validado con algunos clientes...\n4 - MVP validado y con base de clientes sólida...\n5 - MVP validado con crecimiento acelerado y adopción comercial robusta...",
        "demonstration": "Caso de uso implementado, métricas de adopción, testimonios de clientes."
    },
    "Propuesta de valor para Brasil": {
        "description": "Oferta del valor de la solución para el mercado brasileño y la región de São José dos Campos.",
        "rubric": "1 - Propuesta poco clara o irrelevante...\n2 - Algún potencial de adaptación sin diferenciación clara...\n3 - Propuesta razonable, pero necesita ajustes...\n4 - Propuesta bien estructurada con diferenciación clara...\n5 - Propuesta altamente relevante y diferenciada para Brasil...",
        "demonstration": "Análisis de mercado, adaptaciones específicas, estrategia de entrada."
    },
    "Viabilidad de mercado": {
        "description": "Viabilidad de mercado de la solución para el mercado brasileño.",
        "rubric": "1 - Sin evidencia de mercado en Brasil...\n2 - Interés inicial sin pruebas sólidas...\n3 - Potencial razonable con algunas evidencias...\n4 - Fuerte potencial con estudios estructurados...\n5 - Mercado altamente viable validado con socios estratégicos...",
        "demonstration": "Investigación de mercado, entrevistas, cartas de intención."
    },
    "Viabilidad técnica": {
        "description": "Viabilidad técnica de la solución.",
        "rubric": "1 - Tecnología inviable o no adaptable...\n2 - Solución viable con barreras técnicas significativas...\n3 - Viabilidad razonable pero requiere mejoras...\n4 - Solución bien desarrollada y técnicamente viable...\n5 - Solución robusta y técnicamente superior sin barreras relevantes...",
        "demonstration": "Arquitectura técnica, pruebas de compatibilidad, evaluaciones regulatorias."
    },
    "Grado de innovación": {
        "description": "Grado de innovación de la solución.",
        "rubric": "1 - Sin innovación aparente...\n2 - Poca innovación en relación al mercado...\n3 - Innovación moderada con diferenciación replicable...\n4 - Innovación avanzada con barreras de entrada...\n5 - Innovación disruptiva y única en el mercado...",
        "demonstration": "Patentes, elementos diferenciadores, reconocimientos de innovación."
    },
    "Claridad del dolor de mercado": {
        "description": "Claridad del problema de mercado que se soluciona con la internacionalización a Brasil.",
        "rubric": "1 - Problema indefinido o débil...\n2 - Problema identificado sin validación clara...\n3 - Problema real pero con fuerte competencia...\n4 - Problema bien fundamentado con oportunidades claras...\n5 - Problema crítico con fuerte demanda y sin soluciones competidoras viables...",
        "demonstration": "Investigación de mercado, testimonios, análisis competitivo."
    },
    "Ventaja competitiva": {
        "description": "Ventaja competitiva de la solución.",
        "rubric": "1 - Sin ventaja clara...\n2 - Alguna ventaja, pero replicable...\n3 - Ventaja relevante con riesgo de ser superada...\n4 - Fuerte ventaja competitiva sostenida...\n5 - Diferenciación única y sostenible a largo plazo con barreras significativas...",
        "demonstration": "Análisis competitivo, propiedad intelectual, estrategia diferenciadora."
    }
}

# Sidebar para la API key y detalle de criterios
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("OpenAI API Key", type="password")

    st.subheader("Criterios de Evaluación")
    for criterion, details in criteria.items():
        with st.expander(criterion):
            st.markdown(f"**Descripción**: {details['description']}")
            st.markdown("**Rúbrica**:")
            st.text(details['rubric'])
            st.markdown(f"**Demostración**: {details['demonstration']}")

# Contenido principal
tab1, tab2 = st.tabs(["Evaluación", "Resultados"])

with tab1:
    st.subheader("Respuestas del Formulario")

    # Opción para pegar texto o subir archivo
    input_method = st.radio("Método de entrada", ["Pegar texto", "Subir archivo"])

    if input_method == "Pegar texto":
        form_responses = st.text_area("Pegue las respuestas del formulario aquí", height=400)
    else:
        uploaded_file = st.file_uploader("Suba un archivo con las respuestas", type=["txt"])
        if uploaded_file is not None:
            form_responses = uploaded_file.getvalue().decode("utf-8")
            st.text_area("Contenido del archivo", form_responses, height=400)
        else:
            form_responses = ""

    # Inicializar estado de procesamiento si no existe
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    col1, col2 = st.columns([3, 1])
    with col1:
        evaluate_button = st.button("Evaluar Postulación", use_container_width=True)
    with col2:
        if st.session_state.processing:
            st.markdown("""
            <div class='processing-indicator'>
                <div class="spinner" style="width: 20px; height: 20px; border-width: 3px; margin-right: 10px;"></div>
                EVALUANDO...
            </div>
            """, unsafe_allow_html=True)

# Función para evaluar la postulación
def evaluate_application(api_key, form_responses):
    if not api_key:
        st.error("Por favor, ingrese una API key válida en la barra lateral")
        return None

    prompt_template = """
    Actúa como un evaluador experto para un programa de selección de empresas latinoamericanas que buscan expandirse a Brasil.

    Tu tarea es evaluar la siguiente postulación según los 10 criterios establecidos.

    RESPUESTAS DEL FORMULARIO:
    {form_responses}

    CRITERIOS DE EVALUACIÓN:
    {criteria_json}

    INSTRUCCIONES:
    1. Evalúa cada criterio en una escala del 1 al 5 según las rúbricas proporcionadas.
    2. Para cada criterio, proporciona:
       - Una justificación detallada del puntaje asignado
       - Recomendaciones específicas para que la empresa pueda mejorar en ese criterio
    3. Extrae un resumen conciso de la empresa con: Nombre, País de origen, y Propuesta de valor principal
    4. Calcula el puntaje total y determina si la empresa es seleccionada (>75% del puntaje máximo)

    Devuelve tu evaluación en formato JSON con la siguiente estructura exacta:
    {{
        "resumen": {{
            "nombre": "Nombre de la empresa",
            "origen": "País de origen",
            "propuesta_valor": "Breve descripción de la propuesta de valor"
        }},
        "evaluaciones": [
            {{
                "criterio": "Nombre del criterio",
                "puntaje": puntaje (1-5),
                "justificacion": "Justificación del puntaje asignado",
                "recomendacion": "Recomendación para mejorar el puntaje"
            }},
            ...
        ],
        "puntaje_total": suma de puntajes,
        "porcentaje": porcentaje del puntaje máximo posible,
        "seleccionada": true/false (si el porcentaje es >75%)
    }}
    """

    prompt = PromptTemplate(
        input_variables=["form_responses", "criteria_json"],
        template=prompt_template
    )

    # Inicializar LLM y cadena de procesamiento
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o", temperature=0.2)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Ejecutar la cadena
    criteria_json = json.dumps(criteria, ensure_ascii=False)
    result = chain.run(form_responses=form_responses, criteria_json=criteria_json)

    try:
        # Extraer el JSON de la respuesta (en caso de haber texto adicional)
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        json_str = result[json_start:json_end]

        # Parsear la respuesta JSON
        evaluation = json.loads(json_str)
        return evaluation
    except Exception as e:
        st.error(f"Error al procesar la respuesta: {e}")
        st.write("Respuesta del modelo:")
        st.write(result)
        return None

# Mostrar resultados en la pestaña "Resultados"
with tab2:
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

    if evaluate_button and form_responses and api_key:
        with st.spinner("Procesando evaluación, por favor espere..."):
            evaluation = evaluate_application(api_key, form_responses)

        if evaluation:
            st.success("✅ ¡Evaluación completada con éxito!")
            st.session_state.evaluation_results = evaluation

    if st.session_state.evaluation_results:
        evaluation = st.session_state.evaluation_results

        # Tarjeta resumen de la empresa
        st.header("Resumen de la Empresa")
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top:0;">Datos Generales</h3>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div style="padding: 10px;">
                <p><strong>🏢 Nombre:</strong> {evaluation['resumen']['nombre']}</p>
                <p><strong>🌎 Origen:</strong> {evaluation['resumen']['origen']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="padding: 10px;">
                <p><strong>💡 Propuesta de Valor:</strong></p>
                <p style="font-style:italic;">{evaluation['resumen']['propuesta_valor']}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Evaluación por criterios
        st.header("Evaluación por Criterios")
        score_cols = st.columns(3)
        percentage = evaluation['porcentaje']
        if percentage >= 75:
            score_class = "success"
            icon = "✅"
        elif percentage >= 60:
            score_class = "warning"
            icon = "⚠️"
        else:
            score_class = "danger"
            icon = "❌"

        with score_cols[0]:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Puntaje Total</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{evaluation['puntaje_total']}/{len(criteria) * 5}</p>
            </div>
            """, unsafe_allow_html=True)

        with score_cols[1]:
            st.markdown(f"""
            <div class="metric-box {score_class}">
                <h4>Porcentaje</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{evaluation['porcentaje']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with score_cols[2]:
            status = "SELECCIONADA" if evaluation["seleccionada"] else "NO SELECCIONADA"
            status_icon = "✅" if evaluation["seleccionada"] else "❌"
            status_class = "success" if evaluation["seleccionada"] else "danger"
            st.markdown(f"""
            <div class="metric-box {status_class}">
                <h4>Decisión Final</h4>
                <p style="font-size: 1.2rem; font-weight: bold;">{status_icon} {status}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Evaluación individual de cada criterio
        for eval_item in evaluation["evaluaciones"]:
            criterion_name = eval_item['criterio']
            score = eval_item['puntaje']
            if score >= 4:
                score_class = "score-high"
                emoji = "🔥"
            elif score >= 3:
                score_class = "score-medium"
                emoji = "⚠️"
            else:
                score_class = "score-low"
                emoji = "⚠️"

            expander_title = f"""
            <div class="criterion-header">
                <span>{criterion_name}</span>
                <span class="score-badge {score_class}">{emoji} {score}/5</span>
            </div>
            """

            with st.expander(criterion_name):
                st.markdown(expander_title, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>Descripción del criterio:</strong> {criteria[criterion_name]['description']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="margin: 15px 0;">
                    <h4>Justificación</h4>
                    <p>{eval_item['justificacion']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color: #fff8e1; padding: 15px; border-radius: 5px; border-left: 4px solid #FFC107; margin: 15px 0;">
                    <h4>💡 Recomendaciones para mejorar</h4>
                    <p>{eval_item['recomendacion']}</p>
                </div>
                """, unsafe_allow_html=True)

        # Opción para descargar los resultados
        st.divider()
        st.subheader("Exportar Resultados")
        json_str = json.dumps(evaluation, indent=2, ensure_ascii=False)
        st.download_button(
            label="Descargar Evaluación (JSON)",
            data=json_str,
            file_name=f"evaluacion_{evaluation['resumen']['nombre'].replace(' ', '_')}.json",
            mime="application/json"
        )
    else:
        st.info("Aún no hay resultados de evaluación. Por favor, complete el formulario en la pestaña 'Evaluación' y haga clic en 'Evaluar Postulación'.")

# Footer
st.markdown("---")
st.markdown("""
### Instrucciones de Uso
1. Ingrese su API key de OpenAI en la barra lateral
2. Introduzca las respuestas del formulario (pegando texto o subiendo un archivo)
3. Haga clic en "Evaluar Postulación"
4. Revise los resultados detallados en la pestaña "Resultados"

La aplicación evalúa 10 criterios en una escala de 1 a 5. Para ser seleccionada, la empresa debe obtener al menos el 75% del puntaje máximo posible (38 de 50 puntos).
""")
st.markdown("<p style='text-align: center; font-style: italic;'>Hecho con <3 para los mentores de Softlanding Power by Xpertia</p>", unsafe_allow_html=True)
