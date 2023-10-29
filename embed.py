import pandas as pd
import numpy as np
import altair as alt
from datasets import load_dataset
from flask import Flask, render_template_string
from sklearn.manifold import TSNE

from embedwise.embedders import SentenceEncoder

app = Flask(__name__)

encoder = SentenceEncoder('all-MiniLM-L6-v2')

dataset = load_dataset("PolyAI/banking77")
# print(dataset['train']['text'][0])
# print(dataset['test']['text'][0])

text_to_embed = dataset['train']['text'][:5000]
# print(f"text_to_embed: \n{text_to_embed}")
embeddings = encoder.transform(text_to_embed)
# print(f"embedding: \n{embeddings}")
# print(type(embeddings))
# for emb in embeddings:
#     print(f"emb.shape: {emb.shape}\t emb.dtype: {emb.dtype}")


tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings)

print(embeddings_2d.shape)


def create_chart():
    # Create a DataFrame
    # data = pd.DataFrame({
    #     'x': text_to_embed,
    #     'y': embedding
    # })
    # # Create a Chart
    # chart = alt.Chart(data).mark_bar().encode(
    #     x='x',
    #     y='y'
    # )
    # return chart

    data = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    data["text"] = text_to_embed


    # Create a selection tool
    brush = alt.selection(type='interval', resolve='global')

    # Scatter Plot
    scatter = alt.Chart().mark_point().encode(
        x='x',
        y='y',
        color=alt.condition(brush, 'Origin:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=300,
        height=300
    )

    # Table Panel
    table = alt.Chart().mark_text(align='left').encode(
        y=alt.Y('row_number:O', axis=None),
        text='text:N',
        color=alt.condition(brush, alt.value('black'), alt.value('lightgray'))
    ).transform_window(
        row_number='row_number()'
    ).transform_filter(
        alt.datum._vgsid_ > 0  # No texts initially
    ).transform_filter(
        brush
    ).properties(
        width=300,
        height=300
    )

    # Combine both
    chart = alt.hconcat(scatter, table, data=data).configure_view(
        stroke=None
    )
    
    return chart



@app.route("/")
def index():
    chart = create_chart()
    chart_json = chart.to_json()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    </head>
    <body>
        <div id="vis"></div>
        <script>
            var spec = {chart_json};
            vegaEmbed('#vis', spec);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

app.run()