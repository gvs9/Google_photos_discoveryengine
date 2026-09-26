import streamlit as st
import json
import os

# Set up the page configuration
st.set_page_config(
    page_title="GPDE Telemetry Dashboard", 
    layout="wide", 
    page_icon="🔍"
)

# Custom CSS for dark mode aesthetic matching the HTML mockup
st.markdown("""
<style>
    .stApp {
        background-color: #05070e;
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #00f2fe;
        font-family: 'Space Grotesk', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("Google Photos Discovery Engine — Telemetry & ML Insights")
st.write("Welcome to the interactive Streamlit Dashboard!")

# Sidebar
with st.sidebar:
    st.header("Pipeline Controls")
    st.write("Use this sidebar to trigger data pipelines.")
    
    if st.button("Run Data Ingestion"):
        st.info("Ingestion pipeline triggered. (Placeholder)")
            
    if st.button("Run Vector Clustering"):
        st.info("Clustering pipeline triggered. (Placeholder)")

# Tabs for organization
tab1, tab2 = st.tabs(["Opportunity Scorecard", "PM Discovery Copilot"])

with tab1:
    st.header("Top Retrieval Gaps")
    st.write("This tab will load data from `outputs/opportunity_scorecard.json`.")
    
    # Attempt to load actual scorecard data if it exists
    scorecard_path = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "opportunity_scorecard.json")
    if os.path.exists(scorecard_path):
        try:
            with open(scorecard_path, "r") as f:
                data = json.load(f)
            st.json(data)
        except Exception as e:
            st.error(f"Error loading scorecard: {e}")
    else:
        st.warning("Scorecard not found. Run the pipeline to generate data.")

with tab2:
    st.header("PM Discovery Copilot")
    st.write("Ask questions about the telemetry data here.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I am the Google Photos Discovery Engine Copilot. I have indexed 350,720 qualitative feedback logs. What product hypothesis would you like to investigate?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What are the top complaints?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Simple RAG simulation based on our previous HTML logic
        response = f'Analyzing 350,720 records for: "{prompt}". No specific insights found.'
        
        if "top complaints" in prompt.lower() or "struggle to retrieve" in prompt.lower():
            response = "Analysis of abandonment logs reveals users struggle most with 3 categories: 1) Utilitarian captures, 2) Ambient episodic memories, and 3) Visually ambiguous objects. Recommendation: Implement 'Personal Context Weighting'."
        elif "remember" in prompt.lower():
             response = "Users primarily retain 'Episodic Anchors' rather than specific visual facts. Top retained dimensions: 1) Spatial/Location (42%), 2) People/Social Context (28%), 3) Emotional State (18%)."

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
