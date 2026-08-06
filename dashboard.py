import streamlit as st
import requests
import pandas as pd
import plotly.express as px

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Smart Irrigation Digital Twin", layout="wide")

st.title("Smart Irrigation Digital Twin")
st.caption("Hybrid Machine Learning and Digital Twin Framework for Precision Agriculture")


def get_field_state():
    response = requests.get(BACKEND_URL + "/field-state")
    return response.json()["plots"]


def get_history():
    response = requests.get(BACKEND_URL + "/simulate/history")
    return response.json()["history"]


def simulate_days(days):
    response = requests.post(BACKEND_URL + "/simulate/" + str(days))
    return response.json()


def reset_field():
    response = requests.post(BACKEND_URL + "/reset")
    return response.json()


st.sidebar.header("Controls")
days_to_simulate = st.sidebar.number_input("Days to simulate", min_value=1, max_value=90, value=10)

simulate_clicked = st.sidebar.button("Run Simulation", type="primary")
reset_clicked = st.sidebar.button("Reset Field")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if reset_clicked:
    reset_field()
    st.session_state.last_result = None
    st.sidebar.success("Field reset to a fresh random state.")

if simulate_clicked:
    result = simulate_days(days_to_simulate)
    st.session_state.last_result = result
    st.sidebar.success("Simulated " + str(days_to_simulate) + " day(s).")

if st.session_state.last_result:
    result = st.session_state.last_result
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Days Simulated", result["current_day"])
    col2.metric("Water Used (Model)", str(result["model_water_used_mm"]) + " mm")
    col3.metric("Water Used (Baseline)", str(result["baseline_water_used_mm"]) + " mm")
    col4.metric("Water Saved", str(result["water_saved_pct"]) + " %")
else:
    st.info("Click 'Run Simulation' in the sidebar to start the digital twin.")

st.divider()

st.subheader("Field State (Digital Twin)")

plots = get_field_state()
df_plots = pd.DataFrame(plots)

if not df_plots.empty:
    grid_cols = 3
    df_plots["row"] = (df_plots.index // grid_cols)
    df_plots["col"] = (df_plots.index % grid_cols)

    fig = px.scatter(
        df_plots,
        x="col",
        y="row",
        color="soil_moisture_pct",
        size=[40] * len(df_plots),
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        hover_data=["plot_id", "crop_type", "soil_moisture_pct", "last_irrigated_day"],
        text="crop_type",
    )
    fig.update_traces(textposition="middle center", marker=dict(line=dict(width=2, color="black")))
    fig.update_yaxes(autorange="reversed", showticklabels=False, title=None)
    fig.update_xaxes(showticklabels=False, title=None)
    fig.update_layout(height=350, coloraxis_colorbar=dict(title="Moisture %"))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df_plots[["plot_id", "crop_type", "crop_day", "soil_moisture_pct", "temperature_c", "humidity_pct", "last_irrigated_day"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Soil Moisture Over Time")

history = get_history()
if history:
    df_hist = pd.DataFrame(history)
    df_hist["plot_label"] = "Plot " + df_hist["plot_id"].astype(str) + " (" + df_hist["crop_type"] + ")"

    fig2 = px.line(
        df_hist,
        x="day",
        y="soil_moisture_pct",
        color="plot_label",
        markers=True,
    )
    fig2.update_layout(height=400, xaxis_title="Simulated Day", yaxis_title="Soil Moisture (%)")
    st.plotly_chart(fig2, use_container_width=True)

    irrigated_days = df_hist[df_hist["irrigated"] == True]
    st.caption(str(len(irrigated_days)) + " irrigation events occurred across all plots during this simulation.")
else:
    st.info("No simulation history yet. Run a simulation to see the moisture trend over time.")