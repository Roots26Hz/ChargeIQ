"""Battery track, chapter 7 — estimate a cell's State of Health."""

import pandas as pd
import streamlit as st

from content import battery as text
from core import regression as reg
from core.artifacts import load_metadata, load_model
from ui import components as ui

meta = load_metadata(reg.TRACK)
model = load_model(reg.TRACK)
FIELDS = reg.input_fields(meta)
BATTERIES = reg.held_out_batteries(meta)
CYCLES = {b: {c.cycle_index: c for c in reg.held_out_cycles(b, meta)} for b in BATTERIES}

# Slider format per input; the step is a fraction of the input's range.
FORMATS = {"ambient_temperature": "%d", "cycle_index": "%d", "Re": "%.4f", "Rct": "%.4f",
           "early_voltage_sag_rate": "%.5f", "early_temp_rise_rate": "%.4f", "charge_duration": "%.0f",
           "charge_max_temp": "%.1f", "charge_time_to_cv": "%.0f"}


def key(field: str) -> str:
    return f"bp_{field}"


# ---------------------------------------------------------------- state
# The battery and cycle pickers choose a real held-out cycle; the sliders start
# from that cycle's measured inputs and can then be moved freely.


def _selected():
    return CYCLES[st.session_state[key("battery")]][st.session_state[key("cycle")]]


def _load_cycle() -> None:
    for field, value in _selected().inputs.items():
        st.session_state[key(field)] = value


def _battery_changed() -> None:
    st.session_state[key("cycle")] = min(CYCLES[st.session_state[key("battery")]])
    _load_cycle()


if key("battery") not in st.session_state:  # first visit: the longest-lived unseen battery, first cycle
    st.session_state[key("battery")] = max(BATTERIES, key=lambda b: len(CYCLES[b]))
    st.session_state[key("cycle")] = min(CYCLES[st.session_state[key("battery")]])
    _load_cycle()


def _temperatures(battery: str) -> str:
    return "/".join(str(t) for t in meta["batteries"]["ambient_temperatures"][battery]) + " °C"


# ---------------------------------------------------------------- page

ui.chapter_header(
    "Battery track · Chapter 7",
    "Estimate battery health",
    "Pick a real cycle from a battery the model never saw, then adjust its measurements.",
)

st.header("A battery the model never saw", anchor=False)
left, right = st.columns([1, 2])
with left:
    st.selectbox("Test battery", BATTERIES, key=key("battery"), on_change=_battery_changed,
                 format_func=lambda b: f"{b} · {_temperatures(b)} · {len(CYCLES[b])} cycles")
with right:
    st.select_slider("Discharge cycle", options=list(CYCLES[st.session_state[key("battery")]]),
                     key=key("cycle"), on_change=_load_cycle)

battery = st.session_state[key("battery")]
cycle = _selected()
cycles = list(CYCLES[battery].values())
predictions = reg.predict_many([c.inputs for c in cycles], model, meta)
predicted = float(predictions[[c.cycle_index for c in cycles].index(cycle.cycle_index)])

ui.stat_row([
    ("Measured SOH", f"{cycle.measured_soh:.1f}%", "100 × measured Capacity / 2.0 Ah"),
    ("Predicted SOH", f"{predicted:.1f}%", "The exported model's estimate from this cycle's nine inputs"),
    ("Error", f"{predicted - cycle.measured_soh:+.1f} pts", "Predicted minus measured, in SOH percentage points"),
])

st.line_chart(
    pd.DataFrame({"Cycle": [c.cycle_index for c in cycles],
                  "Measured SOH (%)": [c.measured_soh for c in cycles],
                  "Predicted SOH (%)": predictions}),
    x="Cycle", y=["Measured SOH (%)", "Predicted SOH (%)"], height=300,
)
mae = sum(abs(p - c.measured_soh) for p, c in zip(predictions, cycles)) / len(cycles)
st.caption(f"Every cycle of {battery}: measured and predicted SOH. Mean absolute error over its "
           f"{len(cycles)} cycles: {mae:.2f} percentage points.")

measured_outside = reg.outside_training_range(cycle.inputs, meta)
if measured_outside:
    st.info("This cycle's " + ", ".join(f"`{f}`" for f in measured_outside) + " lie outside the range seen in "
            "training, so this prediction is an extrapolation.", icon=":material/info:")

# ---------------------------------------------------------------- what if

st.header("Adjust the measurements", anchor=False)
st.caption(f"The sliders start from cycle {cycle.cycle_index} of {battery}. Each spans the range the data shows, "
           "including the held-out batteries.")

GROUPS = (("Cycle and impedance", ["ambient_temperature", "cycle_index", "Re", "Rct"]),
          ("Charge and discharge curves", [f for f in FIELDS if f.startswith(("early_", "charge_"))]))
assert sorted(f for _, group in GROUPS for f in group) == sorted(FIELDS)

for column, (title, group) in zip(st.columns(2), GROUPS):
    with column:
        st.markdown(f"**{title}**")
        for field in group:
            low, high = reg.input_range(field, meta)
            whole = meta["dtypes"][field].startswith("int")
            st.slider(field, min_value=int(low) if whole else float(low),
                      max_value=int(high) if whole else float(high), step=1 if whole else (high - low) / 500,
                      format=FORMATS[field], key=key(field), help=text.INPUT_DESCRIPTIONS[field])
st.button("Reset to the measured cycle", icon=":material/restart_alt:", on_click=_load_cycle)

adjusted = {field: st.session_state[key(field)] for field in FIELDS}
try:
    estimate = reg.predict(adjusted, model, meta)
except reg.ValidationError as error:
    st.error("These inputs can't be used:\n\n" + "\n".join(f"- **{f}** {m}" for f, m in error.errors.items()))
    st.stop()

changed = [f for f in FIELDS if adjusted[f] != cycle.inputs[f]]
st.metric("Predicted SOH for these inputs", f"{estimate:.1f}%",
          delta=f"{estimate - predicted:+.1f} pts vs the measured cycle's prediction" if changed else None,
          delta_color="off")
if changed:
    st.caption("Changed: " + ", ".join(f"`{f}`" for f in changed) + ". There is no measured SOH for an "
               "adjusted cycle, only the model's estimate.")
else:
    st.caption("These are the measured inputs of the selected cycle. Move any slider to see how the estimate responds.")

outside = reg.outside_training_range(adjusted, meta)
if outside:
    st.warning("Outside the training range: " + ", ".join(f"`{f}`" for f in outside) + ". The model never saw "
               "values like these in training, so this estimate is an extrapolation.", icon=":material/warning:")

ui.next_chapter("charging_data", "Next: the driver track")
