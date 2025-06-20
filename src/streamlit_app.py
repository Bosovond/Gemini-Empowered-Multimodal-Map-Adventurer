import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import random
import ipywidgets as widgets
import folium
from folium import Map, Marker, TileLayer, LayerControl
from folium.plugins import BeautifyIcon, MousePosition
from streamlit_folium import st_folium
import os
from IPython.display import display, HTML
from dotenv import load_dotenv
import google.genai as genai
from google.cloud import secretmanager

# Imports for the updated Google Gen AI SDK and Pydantic for JSON mode
from google.genai import types
from pydantic import BaseModel, Field
from pyperclip import copy, paste

# Load API key from environment

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the client for the google-genai SDK
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in a .env file.")
    st.stop()

def get_gemini_response(messages):
    # Format messages for Gemini (role/content)
    formatted = [{"role": m["role"], "content": m["content"]} for m in messages]
    response = client.chat.completions.create(
        model="gemini-2.5-flash-exp",
        messages=formatted,
        max_tokens=1000,
        temperature=2,
        top_p=0.95,
        stop_sequences=["\n"],
    )
    # Adjust this line if SDK output changes
    return response.choices[0].message.content if response.choices else None


# --- Data and State Management ---

# Define location categories
categories = {
    "🌌 Ancient": [
        {
            "name": "Giza Pyramids",
            "summary": "The Great Pyramids at the Giza Plateau, Egypt",
            "coords": [29.9792, 31.1342],
            "url": "https://en.wikipedia.org/wiki/Great_Pyramid_of_Giza",
        },
        {
            "name": "Stonehenge",
            "summary": "Prehistoric monument in England",
            "coords": [51.1789, -1.8262],
            "url": "https://en.wikipedia.org/wiki/Stonehenge",
        },
        {
            "name": "Göbekli Tepe",
            "summary": "Ancient human construction.",
            "coords": [37.2231, 38.9226],
            "url": "https://en.wikipedia.org/wiki/G%C3%B6bekli_Tepe",
        },
        #        {"name": "", "summary": "", "coords": [], "url":},
        #        {"name": "", "summary": "", "coords": [], "url":}
    ],
    "🧿 Esoteric": [
        {
            "name": "Mount Shasta",
            "summary": "Sacred and legendary mountain.",
            "coords": [41.4091, -122.1946],
            "url": "https://en.wikipedia.org/wiki/Mount_Shasta",
        },
        {
            "name": "Easter Island",
            "summary": "Famous for its moai statues",
            "coords": [-27.1127, -109.3497],
            "url": "https://en.wikipedia.org/wiki/Easter_Island",
        },
        {
            "name": "Bermuda Triangle",
            "summary": "Region with mysterious disappearances",
            "coords": [25.0, -71.0],
            "url": "https://en.wikipedia.org/wiki/Bermuda_Triangle",
        },
        #        {"name": "", "summary": "", "coords": [], "url":}
    ],
    "🔮 Surreal": [
        {
            "name": "Socotra Island",
            "summary": "Mysterious and forgotten ancient hub of trade.",
            "coords": [12.4634, 54.0046],
            "url": "https://en.wikipedia.org/wiki/Socotra_island",
        },
        {
            "name": "Salar de Uyuni",
            "summary": "World's largest salt flat.",
            "coords": [-20.1338, -67.4891],
            "url": "https://en.wikipedia.org/wiki/Salar_de_Uyuni",
        },
        {
            "name": "Coral Castle",
            "summary": "Mysterious, near-megalithic, stone structure. Built by one man. Twice.",
            "coords": [25.5003, -80.4450],
            "url": "https://en.wikipedia.org/wiki/Coral_Castle",
        },
        {
            "name": "Ringing Rocks",
            "summary": "Rocks that resonate like a bell when struck.",
            "coords": [40.5652, -75.0997],
            "url": "https://en.wikipedia.org/wiki/Ringing_Rocks",
        },
        {
            "name": "McMurdo Dry Valleys",
            "summary": "A 1,500sq/mi section of Antarctica that remains snow and ice-free year round.",
            "coords": [77.4666, 162.5166],
            "url": "https://en.wikipedia.org/wiki/McMurdo_Dry_Valleys",
        },
    ],
    "💫 The Veil Thins..": [
        {
            "name": "Area 51",
            "summary": "Mysterious US Air Force facility",
            "coords": [37.235, -115.8111],
            "url": "https://en.wikipedia.org/wiki/Area_51",
        },
        {
            "name": "Sedona Vortices",
            "summary": "Rumored Energetic vortices that exist throughout Sedona.",
            "coords": [34.8697, -111.7610],
            "url": "",
        },
        {
            "name": "Sedona Airport",
            "summary": "Small airport with stunning views",
            "coords": [34.8516, -111.7900],
            "url": "https://en.wikipedia.org/wiki/Sedona_Airport",
        },
        {
            "name": "Denver International Airport",
            "summary": "Largest, and by far the strangest, airport in North America",
            "coords": [39.8617, -104.6731],
            "url": "https://en.wikipedia.org/wiki/Denver_International_Airport",
        },
        {
            "name": "Roswell UFO Incident",
            "summary": "Site of famous UFO incident.",
            "coords": [33.9504, -105.3145],
            "url": "https://en.wikipedia.org/wiki/Roswell_UFO_incident",
        },
        {
            "name": "The Tunguska Event",
            "summary": "Pre-nuclear explosion measuring between 3 and 50 megatons.",
            "coords": [60.9030, 101.9097],
            "url": "https://en.wikipedia.org/wiki/Tunguska_event",
        },
        #         {"name": "", "summary": "", "coords": [], "url": ""}
    ],
}

#------------------------------------------------------------------
# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_place" not in st.session_state:
    all_places = sum(categories.values(), [])
    st.session_state.selected_place = all_places[0] if all_places else None

#-------------------------------------------------------------
# ---- Import key from secretmanager and GEMINI SETUP ----
# Configure the Gemini API with your API key
# Try loading from Google Cloud Secret Manager first
try:
    # Replace with your Google Cloud project ID and secret name
    project_id = "geminichatfriend"
    secret_name = "GOOGLE_API_KEY"
    version_id = "latest" # Or a specific version number

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    GOOGLE_API_KEY = response.payload.data.decode("UTF-8")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API key loaded successfully from Google Cloud Secret Manager.")

except Exception as e_secret_manager:
    print(f"Failed to load API key from Secret Manager: {e_secret_manager}")
    print("Attempting to load API key from .env file...")
    # If Secret Manager fails, try loading from a .env file
    load_dotenv() # Load variables from .env file
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Gemini API key loaded successfully from .env file.")
    else:
        print("Google API key not found in .env file.")
        print("Please store your key in Google Cloud Secret Manager or in a .env file named 'GOOGLE_API_KEY'.")
client = genai.Client(api_key=GOOGLE_API_KEY)

def get_gemini_response(messages):
    formatted = [{"role": m["role"], "content": m["content"]} for m in messages]
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=formatted
    )
    return response.choices[0].message.content

#----------------------------------------------------------
# --- Streamlit UI Starts Here ---

st.set_page_config(page_title="G.E.M.M.A.", layout="wide")
st.title("🗺️ Gemini Emopowered Multimodal Map Adventurer/n/n-or-/n/nG.E.M.M.A.")
st.markdown("---")

main_col, chat_col = st.columns([2, 1])

with main_col:
    st.markdown("##### --- Choose a Path")
    button_cols = st.columns(4)
    
    if button_cols[0].button("🎲 Dealer's Choice", use_container_width=True):
        result = ai_suggest_location()
        if result:
            st.session_state["selected_place"] = {"name": result["name"], "coords": result["coords"]}
            st.session_state["chat_history"] = [("ai", result["intro"])]
            st.session_state["visited"].append({"name": result["name"], "coords": result["coords"]})
            st.rerun()

    for i, label in enumerate(categories):
        if button_cols[i + 1].button(label, use_container_width=True):
            place = pick_random_location(label)
            if place:
                st.session_state["selected_place"] = place
                initial_message = f"Tell me about this place: {place['name']}"
                st.session_state["chat_history"] = [("user", initial_message)]
                ai_response = ask_the_ai(st.session_state["chat_history"])
                if isinstance(ai_response, ChatResponse):
                     st.session_state["chat_history"].append(("ai", ai_response.response_text))
                st.session_state["visited"].append(place)
                st.rerun()

#-------------------------------------------------------------
# ---- FOLIUM MAP ----

    st.markdown("""
    <link rel="stylesheet" href="L.Control.MousePosition.css">
    <script src="https://unpkg.com/leaflet-mouse-position@1.0.0/src/L.Control.MousePosition.js"></script>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    selected_place = st.session_state['selected_place']
    st.markdown(f"#### 📍 Now at: {selected_place['name']}")
    
    m = Map(location=selected_place['coords'], zoom_start=6, tiles=None)
    
    TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
        attr='Esri', 
        name='Esri Satellite',
        overlay=True,
        control=True,
    ).add_to(m)
    TileLayer(
        'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi',
        attr='NASA',
        name='Nasa WMS',
        overlay=True,
        control=True,
    ).add_to(m)
    TileLayer(
        'Esri.WorldImagery',
        attr='Esri',
        name='Esri World Imagery',
        overlay=True,
        control=True,
    ).add_to(m)
    TileLayer(
        'Esri.OceanBasemap',
        attr='Esri',
        name='Esri Ocean Basemap',
        overlay=True,
        control=True,
    ).add_to(m)
    TileLayer(
        'USGS.USImagery',
        attr='USGS',
        name='USGS Imagery',
        overlay=True,
        control=True,
    ).add_to(m)
    TileLayer(
        'Stadia.AlidadeSatellite',
        attr='Stadia',
        name='Stadia Alidade Satellite',
        overlay=True,
        control=True,
    ).add_to(m)
    TileLayer(
        'Stadia.StamenTerrain',
        attr='Stadia',
        name='Stadia Stamen Terrain',
        overlay=False,
        control=True,
    ).add_to(m)
    TileLayer(
        'CartoDB positron',
        attr='CartoDB',
        name='Light Map',
        overlay=False,
        control=True,
    ).add_to(m)
    
    Marker(location=selected_place['coords'], tooltip=selected_place['name'], popup=selected_place['name']).add_to(m)
    
    MousePosition(position='topleft', prefix="Lat/Lng:").add_to(m)
    
    LayerControl().add_to(m)
    st_folium(m, width='100%', height=550, returned_objects=[])

#-------------------------------------------------------------
#----Chat UI Beside Map-------

with chat_col:
    st.markdown("#### 🧠 Chat with the Oracle")
    
    chat_box = st.container(height=600)
    with chat_box:
        for role, msg in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.markdown(msg)

    if user_input := st.chat_input("Message the Oracle..."):
        st.session_state["chat_history"].append(("user", user_input))
        
        # Get the AI's response based on the full chat history
        ai_response = ask_the_ai(st.session_state['chat_history'])

        # NEW: Check if the AI wants to change the location or just chat
        if isinstance(ai_response, NewLocationResponse):
            # If the AI provided a new location, update the map and chat
            st.session_state["selected_place"] = {"name": ai_response.name, "coords": ai_response.coords}
            st.session_state["chat_history"].append(("ai", ai_response.intro))
            st.session_state["visited"].append({"name": ai_response.name, "coords": ai_response.coords})
            st.rerun() # Rerun the app to update the map
        
        elif isinstance(ai_response, ChatResponse):
            # If the AI just wants to chat, add its response to the history
            st.session_state["chat_history"].append(("ai", ai_response.response_text))
            st.rerun() # Rerun to display the new message

with st.sidebar:
    st.header("Controls & History")
    if st.checkbox("🧭 Show visited places"):
        if st.session_state["visited"]:
            # Use a set to remove duplicate visited places for a cleaner list
            visited_set = {tuple(place.items()) for place in st.session_state["visited"]}
            for place_tuple in visited_set:
                place_dict = dict(place_tuple)
                st.write(f"- {place_dict.get('name', 'Unknown')}")
        else:
            st.write("No places visited yet.")
            
    if st.button("🛑 Quit App"):
        st.stop()
