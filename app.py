import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# 🔐 AUTH SYSTEM
# =========================================================
if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:

    st.title("🍽 Restaurant SaaS Login")

    menu = st.radio("Choose Option", ["Login", "Sign Up"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if menu == "Sign Up":
        if st.button("Create Account"):
            if username in st.session_state.users:
                st.error("User already exists ❌")
            else:
                st.session_state.users[username] = password
                st.success("Account created ✅ Please login")

    else:
        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.success("Login successful 🎉")
                st.rerun()
            else:
                st.error("Invalid credentials ❌")

    st.stop()

# =========================================================
# 📊 LOAD DATA
# =========================================================
df = pd.read_csv("dataset.csv", engine="python", header=None, on_bad_lines="skip")

df.columns = [
    "Restaurant_ID", "Name", "Country_Code", "City", "Address",
    "Locality", "Locality_Verbose", "Longitude", "Latitude",
    "Cuisines", "Average_Cost", "Currency",
    "Has_Table_Booking", "Has_Online_Delivery",
    "Is_Delivering", "Switch_to_Order_Menu",
    "Price_Range", "Aggregate_Rating",
    "Rating_Color", "Rating_Text", "Votes"
]

# ---------------- CLEAN ----------------
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df["Aggregate_Rating"] = pd.to_numeric(df["Aggregate_Rating"], errors="coerce")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
df["Average_Cost"] = pd.to_numeric(df["Average_Cost"], errors="coerce")
df = df.dropna()

# =========================================================
# 🎨 THEME
# =========================================================
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        section[data-testid="stSidebar"] { background-color: #111827; }
        section[data-testid="stSidebar"] * { color: white !important; }
        </style>
    """, unsafe_allow_html=True)

# =========================================================
# 🤖 ML MODEL
# =========================================================
X = df[["Average_Cost", "Votes"]]
y = df["Aggregate_Rating"]

model = RandomForestRegressor()
model.fit(X, y)

# =========================================================
# 🚪 LOGOUT
# =========================================================
st.sidebar.success(f"Welcome {st.session_state.current_user} 👋")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.rerun()

# =========================================================
# 📌 NAVIGATION
# =========================================================
st.sidebar.title("🍽  Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "AI Prediction", "Cuisine Insights", "City Analytics", "World Map"]
)

map_search = st.sidebar.text_input("🌍 Search City / Restaurant")

# =========================================================
# 🏠 DASHBOARD
# =========================================================
if page == "Dashboard":

    st.title("👋 Welcome to Restaurant SaaS Platform")

    col1, col2 = st.columns(2)

    with col1:
        selected_city = st.selectbox("📍 Choose City", df["City"].dropna().unique())

    with col2:
        selected_cuisine = st.selectbox("🍛 Choose Cuisine", df["Cuisines"].dropna().unique())

    city_df = df[df["City"] == selected_city]
    cuisine_df = df[df["Cuisines"] == selected_cuisine]

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("📍 City Restaurants", len(city_df))
    col2.metric("🍛 Cuisine Matches", len(cuisine_df))
    col3.metric("⭐ Avg Rating", round(df["Aggregate_Rating"].mean(), 2))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📍 {selected_city}")
        st.dataframe(city_df[["Name", "Cuisines", "Aggregate_Rating", "Votes"]].head(10))

    with col2:
        st.subheader(f"🍛 {selected_cuisine}")
        st.dataframe(cuisine_df[["Name", "City", "Aggregate_Rating", "Votes"]].head(10))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Top Cities")
        st.bar_chart(df["City"].value_counts().head(8))

    with col2:
        st.subheader("🍛 Top Cuisines")
        st.bar_chart(df["Cuisines"].value_counts().head(8))

# =========================================================
# 🤖 AI PREDICTION
# =========================================================
elif page == "AI Prediction":

    st.title("🤖 AI Rating Predictor")

    cost = st.number_input("Average Cost", 0, 10000)
    votes = st.number_input("Votes", 0, 10000)

    if st.button("Predict"):
        pred = model.predict([[cost, votes]])[0]
        pred = max(1, min(5, pred))
        st.success(f"⭐ Predicted Rating: {round(pred,2)}")

# =========================================================
# 🍛 CUISINE INSIGHTS
# =========================================================
elif page == "Cuisine Insights":

    st.title("🍛 Cuisine Analytics")

    st.bar_chart(df["Cuisines"].value_counts().head(10))

    cuisine = st.selectbox("Select Cuisine", df["Cuisines"].dropna().unique())

    st.dataframe(df[df["Cuisines"] == cuisine][
        ["Name", "City", "Aggregate_Rating", "Votes"]
    ])

# =========================================================
# 📍 CITY ANALYTICS
# =========================================================
elif page == "City Analytics":

    st.title("📍 City Intelligence")

    city = st.selectbox("Select City", df["City"].dropna().unique())
    city_df = df[df["City"] == city]

    col1, col2 = st.columns(2)

    col1.metric("Restaurants", len(city_df))
    col2.metric("Avg Rating", round(city_df["Aggregate_Rating"].mean(), 2))

    st.bar_chart(city_df["Cuisines"].value_counts().head(8))

    st.dataframe(city_df[["Name", "Cuisines", "Aggregate_Rating"]].head(10))

# =========================================================
# 🌍 WORLD MAP
# =========================================================
elif page == "World Map":

    st.title("🌍 Smart Restaurant Map")

    map_df = df.dropna(subset=["Latitude", "Longitude"])

    if map_search:
        map_df = map_df[
            map_df["City"].str.contains(map_search, case=False, na=False)
            | map_df["Locality"].str.contains(map_search, case=False, na=False)
            | map_df["Name"].str.contains(map_search, case=False, na=False)
        ]

    if len(map_df) == 0:
        st.warning("No results found")
    else:
        st.success(f"Showing {len(map_df)} locations")

        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=map_df["Latitude"].mean(),
                longitude=map_df["Longitude"].mean(),
                zoom=10,
                pitch=40
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position='[Longitude, Latitude]',
                    get_color='[255, 80, 80, 160]',
                    get_radius=20000,
                )
            ],
            tooltip={
                "text": "{Name}\n{City}\nRating: {Aggregate_Rating}"
            }
        ))