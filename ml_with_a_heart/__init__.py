from flask import Flask
import ml_with_a_heart.api.views


app = Flask(__name__)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
