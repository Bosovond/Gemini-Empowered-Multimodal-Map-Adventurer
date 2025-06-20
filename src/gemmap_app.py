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
        model="gemini-2.0-flash",
        messages=formatted,
        max_tokens=1000,
        temperature=2.0,
        top_p=0.95,
        stop_sequences=["\n"],
    )
#     Adjust this line if SDK output changes
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
        {
            "name": "Mt. Erebus",
            "summary": "Second tallest volcano in Antarctica; Also an active volcano home to a lavalake that is present year-round.",
            "coords": [-77.5291, 167.1522],
            "url": "https://en.wikipedia.org/wiki/Mount_Erebus",
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

# ---- Session State Setup----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "visited" not in st.session_state:
    st.session_state.visited = []
if "selected_place" not in st.session_state:
# Pick random starting location from any category
    import random

    all_places = sum(categories.values(), [])
    st.session_state.selected_place = random.choice(all_places) if all_places else None
   
# Memory for visited places and chat
if "visited" not in st.session_state:
    st.session_state["visited"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "selected_place" not in st.session_state:
    st.session_state["selected_place"] = random.choice(
        random.choice(list(categories.values()))
    )
#-------------------------------------------------------------

# --- Helper Functions ---


# NEW: More advanced Pydantic schemas to let the AI choose its response type.
class ChatResponse(BaseModel):
    """A standard, text-only response for a conversation."""

    response_text: str = Field(description="The AI's textual response to the user.")


class NewLocationResponse(BaseModel):
    """A command to update the map to a new location."""

    name: str = Field(description="The name of the new location.")
    coords: List[float] = Field(
        description="A list containing latitude and longitude for the new location."
    )
    intro: str = Field(
        description="A brief, esoteric introduction to the new location, which will be shown in the chat."
    )


# The AI's response will be one of the two types defined above.
AIResponse = Union[ChatResponse, NewLocationResponse]


def pick_random_location(category):
    options = categories.get(category, [])
    return random.choice(options) if options else None


def user_input(user_question):
    # Create embeddings for the user question using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load a FAISS vector database from a local file
    new_db = FAISS.load_local("faiss_index", embeddings)

    # Perform similarity search in the vector database based on the user question
    docs = new_db.similarity_search(user_question)

    # Obtain a conversational question-answering chain
    chain = get_conversational_chain()

    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # Print the response to the console
    print(response)

    # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
    st.write("Reply: ", response["output_text"])


# NEW: Updated to handle a flexible AI response (either chat or a new location).
def ask_the_ai(chat_history):
    # This prompt asks the AI to decide whether to chat or provide a new location.
    prompt = f"""
    You are an AI map assistant. Based on the conversation history, decide if the user wants to chat, or if they are asking for a new location.
    - If they want to chat, use the ChatResponse format.
    - If they want a new location (or you are suggesting one), use the NewLocationResponse format.

    Conversation History:
    {chat_history}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                # The AI will choose which schema to use
                "response_schema": AIResponse,
            },
        )
        return response.parsed
    except Exception as e:
        st.error(f"Error communicating with AI: {e}")
        return None


# Rewritten function to use JSON mode for reliability
def ai_suggest_location():
    prompt = "Suggest a real-world surreal or mystical location not from the standard well-known lists."
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": NewLocationResponse,  # Only expect a new location here
            },
        )
        parsed_response = response.parsed
        if parsed_response:
            return parsed_response.model_dump()
        return None
    except Exception as e:
        st.error(f"AI suggestion failed: {e}")
        return None


#-------------------------------------------------------------
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
    st.markdown(f"#### 📍 Now viewing: {selected_place['name']}")
    
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
    
    Marker(
        location=selected_place['coords'],
        tooltip=selected_place['name'],
        popup=selected_place['name'],
        icon=BeautifyIcon(icon="map-marker"),
    ).add_to(m)
    
    MousePosition(
        position='topleft',
        prefix="Lat/Lng:"
    ).add_to(m)
    
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
