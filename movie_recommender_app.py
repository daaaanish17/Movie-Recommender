from movie_recommendation_system import recommendation_function
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'POST':
		movies = request.form.get('movie')
		recomm = recommendation_function('Avatar')
	else:
		recomm = ''

	return render_template('recommender_page.html', value=recomm)


if __name__ == "__main__":
	app.run(host="localhost", port=int("5000"))
