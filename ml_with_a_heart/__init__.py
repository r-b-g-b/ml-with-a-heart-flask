from flask import Flask
app = Flask(__name__)

import ml_with_a_heart.api.views


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
