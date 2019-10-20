from archive.genres import classify, get_barplot_html
from flask import Flask, request, render_template, jsonify, send_file

# with open('archive/10genre_clf.pkl', 'rb') as f:
#     model = pickle.load(f)



app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index_bs.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return the predicted probabilities of belonging to genres."""
    data = request.json
    song_url = data['user_input']
    prediction = classify(url=song_url)
    viz_html = get_barplot_html(genre_probs=prediction)    
    return jsonify({
        'probabilities': prediction,
        'viz_html': viz_html
        })