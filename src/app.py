from flask import Flask
from src.routes.routes import predict

app = Flask(__name__)

app.register_blueprint(predict)

if __name__ == '__main__':
    app.run(debug=True)
