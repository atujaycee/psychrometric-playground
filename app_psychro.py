import streamlit as st
import matplotlib.pyplot as plt

from psychro_core import psychro_state, plot_psychrometric_chart

# =========================
# Page config
# =========================
st.set_page_config(page_title="Psychrometric Playground", layout="wide")

# =========================
# UI styling (CSS)
# =========================
st.markdown(
    """
    <style>
    /* =========================
       Global typography scaling
       ========================= */

    html, body, [class*="css"] {
        font-size: 18px;
    }

    /* =========================
       Main app background
       ========================= */
    .stApp {
        background: #FFF9E6;
    }

    /* =========================
       Sidebar background + text
       ========================= */
    section[data-testid="stSidebar"] {
        background: #EAF4FF;
    }

    section[data-testid="stSidebar"] * {
        color: #0B1F33;
        font-size: 20px;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0B3D66;
        font-size: 24px;
    }

    /* =========================
       Page title & subtitle
       ========================= */
    h1 {
        font-size: 2.4rem !important;
    }

    /* Streamlit caption under title */
    div[data-testid="stCaption"] {
        font-size: 1.15rem !important;
        color: #333333;
        margin-bottom: 1.2rem;
    }

    /* =========================
       Tabs (📈 Chart / 🧾 Computed State)
       ========================= */
    button[data-baseweb="tab"] {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 10px 18px;
    }

    /* =========================
       Metrics (Computed State)
       ========================= */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.80);
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.06);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 1.05rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 600;
    }

    /* =========================
       Expanders
       ========================= */
    details summary {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Title
# =========================
st.title("Psychrometric Chart Playground")
st.caption("Pick any two properties, watch the state move on the chart, and toggle overlays.")

# =========================
# Sidebar: inputs
# =========================
st.sidebar.header("1) Choose two known properties")

pair = st.sidebar.selectbox(
    "Property pair",
    [
        "T + RH",
        "T + W",
        "T + Tdpt",
        "T + Twb",
        "T + h",
        "W + h",
        "RH + Tdpt",
        "RH + Twb",
        "h + Tdpt",
        "h + RH",
        "h + Twb",
        "W + Tdpt (consistency check)",
    ],
)

st.sidebar.divider()
st.sidebar.header("2) Chart settings")

T_min = st.sidebar.number_input("T min (°C)", value=-10.0, step=1.0)
T_max = st.sidebar.number_input("T max (°C)", value=80.0, step=1.0)
W_max_g = st.sidebar.number_input("W max (g/kg)", value=50.0, step=1.0)
W_max = W_max_g / 1000.0

st.sidebar.subheader("Overlays")
show_RH = st.sidebar.checkbox("Show RH lines", value=True)
show_h = st.sidebar.checkbox("Show enthalpy lines", value=True)
show_Twb = st.sidebar.checkbox("Show wet-bulb lines", value=False)
show_v = st.sidebar.checkbox("Show specific volume lines", value=False)

RH_lines = (0.2, 0.4, 0.6, 0.8, 1.0) if show_RH else ()
h_lines = (20, 40, 60, 80, 100, 120) if show_h else ()
#Twb_lines = tuple(range(10, 41, 10)) if show_Twb else ()
Twb_lines = (10, 20, 30, 40) if show_Twb else ()

v_lines = (0.80, 0.85, 0.90, 0.95) if show_v else ()

st.sidebar.subheader("Speed / resolution")
nT = st.sidebar.slider("Resolution (RH/h/v)", 300, 1200, 800, 50)
#nT_twb = st.sidebar.slider("Resolution (Twb)", 80, 400, 200, 20)
nT_twb = st.sidebar.slider("Resolution (Twb)", 10, 40, 15, 5)


st.sidebar.divider()
st.sidebar.header("3) Enter values")

def input_T():
    return st.sidebar.number_input("Dry-bulb T (°C)", value=25.0, step=0.5)

def input_RH_percent():
    return st.sidebar.slider("Relative Humidity (%)", 1, 100, 50, 1)

def input_W():
    return st.sidebar.number_input("Humidity ratio W (kg/kg)", value=0.01000, step=0.0005, format="%.5f")

def input_Twb():
    return st.sidebar.number_input("Wet-bulb Twb (°C)", value=18.0, step=0.5)

def input_Tdpt():
    return st.sidebar.number_input("Dew-point Tdpt (°C)", value=14.0, step=0.5)

def input_h():
    return st.sidebar.number_input("Enthalpy h (kJ/kg dry air)", value=50.0, step=1.0)

# Build given dict
given = {}

if pair == "T + RH":
    T = input_T()
    RH = input_RH_percent() / 100.0
    given = {"T_C": T, "RH": RH}

elif pair == "T + W":
    given = {"T_C": input_T(), "W": input_W()}

elif pair == "T + Tdpt":
    given = {"T_C": input_T(), "Tdpt_C": input_Tdpt()}

elif pair == "T + Twb":
    given = {"T_C": input_T(), "Twb_C": input_Twb()}

elif pair == "T + h":
    given = {"T_C": input_T(), "h_kJkg": input_h()}

elif pair == "W + h":
    given = {"W": input_W(), "h_kJkg": input_h()}

elif pair == "RH + Tdpt":
    RH = input_RH_percent() / 100.0
    given = {"RH": RH, "Tdpt_C": input_Tdpt()}

elif pair == "RH + Twb":
    RH = input_RH_percent() / 100.0
    given = {"RH": RH, "Twb_C": input_Twb()}

elif pair == "h + Tdpt":
    given = {"h_kJkg": input_h(), "Tdpt_C": input_Tdpt()}

elif pair == "h + RH":
    RH = input_RH_percent() / 100.0
    given = {"h_kJkg": input_h(), "RH": RH}

elif pair == "h + Twb":
    given = {"h_kJkg": input_h(), "Twb_C": input_Twb()}

elif pair == "W + Tdpt (consistency check)":
    given = {"W": input_W(), "Tdpt_C": input_Tdpt()}

# =========================
# Caching
# =========================
@st.cache_data(show_spinner=False)
def compute_state(given_dict: dict) -> dict:
    clean = {k: float(v) for k, v in given_dict.items()}
    return psychro_state(clean)

@st.cache_data(show_spinner=False)
def compute_fig(
    state: dict,
    T_min: float, T_max: float, W_max: float,
    RH_lines: tuple, h_lines: tuple, Twb_lines: tuple, v_lines: tuple,
    nT: int, nT_twb: int
):
    fig = plot_psychrometric_chart(
        states=state,
        T_min=T_min, T_max=T_max,
        W_min=0.0, W_max=W_max,
        RH_lines=RH_lines,
        h_lines=h_lines,
        Twb_lines=Twb_lines,
        v_lines=v_lines,                 # safe if empty
        nT=nT,
        nT_twb=nT_twb,
        label_RH=True,
        label_h=True,
        label_Twb=True,
        label_v=True,                    # safe if v_lines=()
        figsize=(16, 8),                 # BIGGER chart
        dpi=160,                         # crisper
        font_base=16,                    # BIGGER labels/ticks inside chart
    )
    return fig

# =========================
# MAIN LAYOUT (OPTION 2)
# =========================
# =========================
# MAIN LAYOUT (TABS)
# =========================
tab_chart, tab_state = st.tabs(["📈 Chart", "🧾 Computed State"])

with tab_chart:
    st.subheader("Psychrometric Chart")

    try:
        with st.spinner("Computing state..."):
            state = compute_state(given)

        with st.spinner("Rendering chart..."):
            fig = compute_fig(
                state, T_min, T_max, W_max,
                RH_lines, h_lines, Twb_lines, v_lines,
                nT, nT_twb
            )

        st.pyplot(fig, clear_figure=True, use_container_width=True)

    except Exception as e:
        st.error(f"Could not compute this state: {e}")
        state = None

with tab_state:
    st.subheader("Computed State")

    if "state" in locals() and state is not None:
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.metric("T (°C)", f"{state['T_C']:.2f}")
            st.metric("W (g/kg)", f"{state['W']*1000.0:.2f}")
        with c2:
            st.metric("RH (%)", f"{state['RH']*100.0:.1f}")
            st.metric("Twb (°C)", f"{state['Twb_C']:.2f}")
        with c3:
            st.metric("Tdpt (°C)", f"{state['Tdpt_C']:.2f}")
            st.metric("h (kJ/kg)", f"{state['h_kJkg']:.2f}")

        with st.expander("Advanced"):
            st.write(f"Pv: **{state['Pv_Pa']:.0f} Pa**")
            st.write(f"Psat: **{state['Psat_Pa']:.0f} Pa**")
            st.write(f"Specific volume v: **{state['v_m3kg']:.3f} m³/kg dry air**")
    else:
        st.info("Compute a valid state (choose a pair and values) to see results here.")

st.caption("Tip: wet-bulb lines are computationally heavier; toggle them on when needed.")

# Footer attribution (only once)
st.caption(
    "Developed by **James Atuonwu, PhD, FHEA, MIET** · "
    "NMITE · "
    "Contact: james.atuonwu@nmite.ac.uk"
)

# Sidebar attribution at the bottom (optional UX pattern)
st.sidebar.divider()
st.sidebar.markdown(
    """
    **About this app**  
    Developed by  
    **James Atuonwu, PhD, FHEA, MIET**  
    NMITE  

    📧 james.atuonwu@nmite.ac.uk
    """
)
