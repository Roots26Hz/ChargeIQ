"""Vehicle track, chapter 4 — try the model, then read its result."""

import html

import streamlit as st

from content import vehicle as text
from core import classification as clf
from core.artifacts import load_metadata, load_model
from ui import components as ui

meta = load_metadata(clf.TRACK)
model = load_model(clf.TRACK)
consulted = set(clf.consulted_features(model, meta))
domains = meta["category_domains"]
ranges = meta["numeric_ranges"]
EXAMPLES = {e.key: e for e in text.EXAMPLES}
LOCATION_CODES = ("State", "Postal Code", "Legislative District", "2020 GEOID", "Vehicle Location",
                  "Electric Utility")
PICKER = "vp_example"


def key(field: str) -> str:
    return f"vp_{field}"


# ---------------------------------------------------------------- form state
# Widgets read their values from session state. The callbacks below keep the
# dependent fields consistent: a Model always belongs to its Make, a City to its
# County, and the location codes start from that city's most common values.


def _whole(value):
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _fill_location() -> None:
    county, city = st.session_state[key("County")], st.session_state[key("City")]
    for field, value in clf.location_defaults(county, city, meta).items():
        st.session_state[key(field)] = _whole(value)


def _apply_example() -> None:
    example = EXAMPLES.get(st.session_state.get(PICKER))
    if example is None:
        return
    values = {"Make": example.make, "Model": example.model, "Model Year": example.model_year,
              "Electric Range": int(example.electric_range), "County": example.county, "City": example.city}
    for field, value in values.items():
        st.session_state[key(field)] = value
    _fill_location()


def _edited() -> None:
    """Any manual change means the inputs no longer match the selected example."""
    st.session_state[PICKER] = None


def _make_changed() -> None:
    st.session_state[key("Model")] = clf.models_for_make(st.session_state[key("Make")], meta)[0]
    _edited()


def _county_changed() -> None:
    st.session_state[key("City")] = clf.cities_for_county(st.session_state[key("County")], meta)[0]
    _fill_location()
    _edited()


def _city_changed() -> None:
    _fill_location()
    _edited()


if key("Make") not in st.session_state:  # first visit in this session: start from the first example
    st.session_state[PICKER] = text.EXAMPLES[0].key
    _apply_example()


def label(field: str) -> str:
    return f"{field} :material/account_tree:" if field in consulted else field


def help_for(field: str) -> str:
    if field in consulted:
        return "Appears in the decision tree's rules, so it can change the prediction."
    return "Required by the model, but never used in the tree's rules: it cannot change the prediction."


def number(field: str, step: int = 1) -> None:
    st.number_input(label(field), min_value=int(ranges[field]["min"]), max_value=int(ranges[field]["max"]),
                    step=step, key=key(field), on_change=_edited, help=help_for(field))


# ---------------------------------------------------------------- page

ui.chapter_header(
    "Vehicle track · Chapter 4",
    "Make a prediction",
    "Describe a vehicle and the trained model predicts whether it is a battery-electric or a plug-in hybrid.",
)

st.pills("Start from an example", list(EXAMPLES), format_func=lambda k: EXAMPLES[k].title,
         selection_mode="single", key=PICKER, on_change=_apply_example)
st.caption(
    "Each example is a make, model, year and range that occurs in the dataset, not an individual record. "
    "Fields marked :material/account_tree: appear in the tree's rules; the others cannot change its prediction."
)

with st.container(border=True):
    st.subheader("The vehicle", anchor=False)
    left, right = st.columns(2)
    with left:
        st.selectbox(label("Make"), domains["Make"], key=key("Make"), on_change=_make_changed,
                     help=help_for("Make"))
        number("Model Year")
    with right:
        st.selectbox(label("Model"), clf.models_for_make(st.session_state[key("Make")], meta),
                     key=key("Model"), on_change=_edited, help=help_for("Model"))
        number("Electric Range")

with st.container(border=True):
    st.subheader("Where it is registered", anchor=False)
    left, right = st.columns(2)
    with left:
        st.selectbox(label("County"), domains["County"], key=key("County"), on_change=_county_changed,
                     help=help_for("County"))
    with right:
        st.selectbox(label("City"), clf.cities_for_county(st.session_state[key("County")], meta),
                     key=key("City"), on_change=_city_changed, help=help_for("City"))
    with st.expander("Location codes", icon=":material/pin_drop:"):
        st.caption("Filled in with the most common values for the chosen city in the training rows. "
                   "Change them if you like.")
        left, right = st.columns(2)
        with left:
            st.selectbox(label("State"), domains["State"], key=key("State"), on_change=_edited,
                         help=help_for("State"))
            number("Postal Code")
            number("Legislative District")
        with right:
            number("2020 GEOID")
            st.selectbox(label("Vehicle Location"), domains["Vehicle Location"], key=key("Vehicle Location"),
                         on_change=_edited, help=help_for("Vehicle Location"))
            st.selectbox(label("Electric Utility"), domains["Electric Utility"], key=key("Electric Utility"),
                         on_change=_edited, help=help_for("Electric Utility"))

# ---------------------------------------------------------------- result

inputs = {field: st.session_state[key(field)] for field in clf.input_fields(meta)}
st.header("The model's answer", anchor=False)

try:
    prediction = clf.predict(inputs, model, meta)
except clf.ValidationError as error:
    st.error("These inputs can't be used:\n\n" + "\n".join(f"- **{f}** {m}" for f, m in error.errors.items()))
    st.stop()

st.markdown(
    f'<div class="ciq-result"><p class="ciq-kicker">Predicted type</p>'
    f'<p class="ciq-result-label">{html.escape(prediction.short_label)}</p>'
    f"<p>{html.escape(prediction.label)}</p></div>",
    unsafe_allow_html=True,
)
year = inputs["Model Year"]
st.caption(f"Vehicle Age, derived by the app as the notebook does: "
           f"{meta['vehicle_age_reference_year']} − {year} = {meta['vehicle_age_reference_year'] - year}.")

example = EXAMPLES.get(st.session_state.get(PICKER))
if example is not None:
    in_data = (f"In the {example.split} rows, {example.rows:,} vehicles have this make, model, year and range, "
               f"all labelled **{example.label_in_data}**.")
    if example.predicted == example.label_in_data:
        st.info(in_data, icon=":material/dataset:")
    else:
        matrix = meta["final_model"]["confusion_matrix"]["matrix"]
        errors = sum(map(sum, matrix)) - sum(matrix[i][i] for i in range(len(matrix)))
        st.warning(f"{in_data} The model predicts **{example.predicted}**: these are {example.rows} of the "
                   f"{errors} test vehicles it misclassifies.", icon=":material/error:")

st.subheader("Why: the path through the tree", anchor=False)
st.markdown("\n".join(
    f"{i}. **{step.condition}** — this vehicle: `{_whole(step.input_value)}` · "
    f"{step.node_samples:,} training vehicles reached this question"
    for i, step in enumerate(prediction.path, start=1)
))
st.markdown(f"→ **{prediction.short_label}**")
st.caption("These are the only questions the tree asked about this vehicle. "
           "Inputs that never appear here did not affect this prediction.")

st.subheader("Who else ended up here", anchor=False)
counts = ", ".join(f"{n:,} {clf.short_label(c)}" for c, n in prediction.leaf_class_counts.items())
share = prediction.leaf_class_shares[prediction.label]
ui.stat_row([
    ("Training vehicles in this leaf", f"{prediction.leaf_samples:,}", counts),
    (f"Share labelled {prediction.short_label}", f"{share:.0%}", None),
])
st.caption(
    "For a decision tree, this share is exactly what predict_proba returns. It describes the training "
    "vehicles in this leaf; it is not a calibrated probability."
)

ui.next_chapter("battery_data", "Next: the battery track")
