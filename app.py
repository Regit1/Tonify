import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import umap
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from data_pipeline import process_videos, get_playlist_links
from threading import Thread


app = Dash(__name__)


def load_graph_figure(csv_file="ForceList.csv"):
    df = pd.read_csv(csv_file, header=None)
    X = df.iloc[:, :-2].values.astype(float)
    song_label = df.iloc[:, -2].astype(str).reset_index(drop=True)
    band_label = df.iloc[:, -1].astype(str).reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reducer = umap.UMAP(n_neighbors=15, metric="cosine", random_state=42)
    X_2d = reducer.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "Song": song_label,
        "Band": band_label
    })

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="Band",
        hover_data=["Song", "Band"]
    )
    fig.update_layout(
        height=800,
        width=1200,
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#111111"
    )
    return fig, song_label, band_label, X_scaled

# Load initial figure
fig, song_label, band_label, X_scaled = load_graph_figure()

# Distance matrix for closest song search
D = pairwise_distances(X_scaled, metric="cosine")

# ---------------------------
# Layout
# ---------------------------
app.layout = html.Div([
    
    html.H1(
        "Tonify",
        style={"textAlign": "center", "fontSize": "48px", "marginBottom": "30px","fontFamily": "Ariel, sans-serif"}
    ),
    html.Div("Welcome to Tonify. This is a tool to find a song similar to your favourite song. "
    "You can add a song or playlist to the algorithm by adding a YouTube URL to the box below. "
    "You can also find the five most similar songs to another song by searching in the box, "
    "and you can see a semi-accurate diagram of songs where closeness represents similarity.",
    style={"textAlign": "center", "width": "60%", "color": "#797979", "margin": "0 auto"}),

    # Input + button row
    html.Div([
        dcc.Input(
            id="playlist-url",
            type="text",
            placeholder="Enter YouTube URL (video or playlist)",
            style={
                "backgroundColor": "#222222",
                "color": "#ffffff",
                "border": "1px solid #444444",
                "padding": "10px",
                "fontSize": "16px",
                "borderRadius": "5px",
                "width": "60%",
                "marginRight": "10px"
            }
        ),
        html.Button(
            "Process link",
            id="process-btn",
            n_clicks=0,
            style={
                "padding": "10px 25px",
                "fontSize": "28px",
                "fontWeight": "bold",
                "color": "#B6B6B6",
                "backgroundColor": "#202020",
                "border": "none",
                "borderRadius": "12px",
                "cursor": "pointer",
                "boxShadow": "2px 2px 8px rgba(0,0,0,0.4)",
                "transition": "all 0.2s ease",
            }
        ),
    ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "20px"}),

    html.Div(id="process-status", style={"textAlign": "center", "marginBottom": "20px"}),

    # Dropdown
    dcc.Dropdown(
    id="song-search",
    placeholder="Type or select a song...",
    style={
        "backgroundColor": "#222222",  # dark background for the box
        "color": "#ffffff",             # text color
        "border": "1px solid #444444",
        "padding": "10px",
        "fontSize": "16px",
        "borderRadius": "5px",
        "width": "60%",
        "margin": "0 auto",
        "display": "block"}
),
    html.Div(id="output", style={"marginTop": 20, "fontSize": 20, "color": "#ffffff"}),

    # Graph
    dcc.Graph(id="graph", figure=fig),

    # Hidden store
    dcc.Store(id="graph-update", data=0),

    # Output
],
style={
    "backgroundColor": "#111111",
    "color": "#ffffff",
    "minHeight": "100vh",
    "width": "100vw",
    "padding": "50px",
    "boxSizing": "border-box",
    "fontFamily": "Calibri, sans-serif"  # <--- change default font here

})

# ---------------------------
# Callback to process playlist/video
# ---------------------------
@app.callback(
    Output("process-status", "children"),
    Output("graph-update", "data"),
    Input("process-btn", "n_clicks"),
    State("playlist-url", "value"),
    State("graph-update", "data"),
    prevent_initial_call=True
)
def run_pipeline(n_clicks, url, graph_data):
    if not url:
        return "Please enter a URL.", graph_data

    NoteFile = pd.read_csv("notes.csv", header=None)
    ChordsFile = pd.read_csv("chords.csv", header=None)

    urls = get_playlist_links(url) if "list=" in url else [url]

    # Run the processing in a background thread
    def process():
        process_videos(urls, NoteFile, ChordsFile)
    Thread(target=process).start()

    # Increment store so graph callback knows to reload
    return f"Processing {len(urls)} video(s)... (each song will take 30 seconds, refresh to view changes)", graph_data + 1

# ---------------------------
# Callback to refresh graph and dropdown after processing
# ---------------------------
@app.callback(
    Output("graph", "figure"),
    Output("song-search", "options"),
    Input("graph-update", "data")
)
def update_graph(store_data):
    fig, song_label, band_label, _ = load_graph_figure()
    options = [{"label": s, "value": s} for s in song_label]
    return fig, options

# ---------------------------
# Callback to find closest song
# ---------------------------
@app.callback(
    Output("output", "children"),
    Input("song-search", "value")
)
def find_closest(selected_song):
    if selected_song is None:
        return html.Div("Select a song to see its closest match.", style={"textAlign": "center"})

    # Reload ForceList.csv to include newly processed songs
    df = pd.read_csv("ForceList.csv", header=None)
    X = df.iloc[:, :-2].values.astype(float)
    song_label = df.iloc[:, -2].astype(str).reset_index(drop=True)
    band_label = df.iloc[:, -1].astype(str).reset_index(drop=True)

    # Recompute distance matrix
    D = pairwise_distances(X, metric="cosine")

    # Find all indices that match the selected song
    matching_indices = song_label[song_label == selected_song].index
    if len(matching_indices) == 0:
        return html.Div(f"Song '{selected_song}' not found yet (still processing?).", style={"textAlign": "center"})

    # Compute distances for each match
    all_distances = []
    for idx in matching_indices:
        dist = D[idx].copy()
        # Exclude rows where both song and band match exactly
        duplicate_mask = (song_label == song_label[idx]) & (band_label == band_label[idx])
        dist[duplicate_mask] = np.inf
        all_distances.append(dist)

    # Combine distances and get 5 closest songs
    combined_distances = np.min(np.array(all_distances), axis=0)
    closest_indices = np.argsort(combined_distances)[:5]

    # Build HTML output
    output_elements = [
        html.Div(
            f"{rank+1}. '{song_label[i]}' by {band_label[i]} (distance: {combined_distances[i]:.4f})",
            style={"marginBottom": "8px"}
        )
        for rank, i in enumerate(closest_indices)
    ]

    return html.Div([
        html.Div(f"Closest songs to '{selected_song}':", style={"marginBottom": "12px", "fontWeight": "bold"}),
        *output_elements
    ], style={"textAlign": "center"})
# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)