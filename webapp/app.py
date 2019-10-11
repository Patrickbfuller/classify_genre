from archive.genres import classify, save_genre_barplot
from flask import Flask, request, render_template, jsonify


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
    save_genre_barplot(
        genre_probs=prediction,
        fp='webapp/static/img/app_visual.png'
        )
    return jsonify({'probabilities': prediction})