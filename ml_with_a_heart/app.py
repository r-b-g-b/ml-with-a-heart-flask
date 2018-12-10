from flask import Flask, render_template, request, jsonify
import json


app = Flask(__name__)

@app.route('/')
def home():
    return """<h1>Machine learning with a heart</h1>""".format()


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)