import os
import json
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# 1) Read multiple JSON files to get embeddings and mask_ids
data_folder = '/home/xuanl/UMAP_local/20images_jsons_file'
all_data = []

for file_name in os.listdir(data_folder):
    if file_name.endswith('.json'):
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            all_data.extend(json_data)

mask_ids = []
embeddings = []

for item in all_data:
    mask_ids.append(item["mask_id"])
    embeddings.append(item["embedding"])

embeddings = np.array(embeddings)
print(f"Total samples: {embeddings.shape[0]}, Embedding dim: {embeddings.shape[1]}")

# 2) Use UMAP to reduce embeddings to a lower-dimensional space 
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=60, random_state=42)
embedding_lowdim = umap_reducer.fit_transform(embeddings)

# 3) Test DBSCAN with different epsilon and min_samples
best_score = -1
best_eps = None
best_min_samples = None
best_labels = None

eps_range = [0.3, 0.5, 1.0]
min_samples_range = [10, 20, 30]

for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embedding_lowdim)

        # Ignore cases where all points are assigned to noise (-1)
        if len(set(cluster_labels)) <= 1:
            continue

        # Calculate Silhouette Score
        sil_score = silhouette_score(embedding_lowdim, cluster_labels)
        print(f"eps={eps}, min_samples={min_samples}, Silhouette Score={sil_score:.4f}")

        if sil_score > best_score:
            best_score = sil_score
            best_eps = eps
            best_min_samples = min_samples
            best_labels = cluster_labels

print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best Silhouette Score: {best_score:.4f}")

# 4) Use UMAP to reduce embeddings to 2D for visualization
umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding_2d = umap_2d.fit_transform(embeddings)

# 5) Build a DataFrame for Plotly visualization
df = pd.DataFrame({
    "x": embedding_2d[:, 0],
    "y": embedding_2d[:, 1],
    "cluster": best_labels.astype(str),  # Use the best cluster labels
    "mask_id": mask_ids
})

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    hover_data=["mask_id"],
    title=f"UMAP + DBSCAN (Best eps={best_eps}, min_samples={best_min_samples}, Silhouette Score={best_score:.4f})"
)

# 6) Create a Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("UMAP + DBSCAN + mask_id Viewer"),
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Div(id='display-info', style={'marginTop': '20px'})
])

@app.callback(
    Output('display-info', 'children'),
    Input('scatter-plot', 'clickData')
)
def update_info(clickData):
    """Display mask_id and image when a point in the scatter plot is clicked."""
    if not clickData:
        return "Please click a point in the scatter plot to view its mask image."

    point_index = clickData["points"][0]["pointIndex"]
    m_id = df.iloc[point_index]["mask_id"]
    img_src = f"/static/{m_id}.png"  # Adjust this path based on your image location

    return html.Div([
        html.H3(f"mask_id: {m_id}"),
        html.Img(src=img_src, style={'width': '200px', 'border': '1px solid #ccc'})
    ])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
