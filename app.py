from flask import Flask
from extensions import db, cors
from config import DATABASE_URI
import os

from routes.device_routes import device_bp
from routes.model_routes import model_bp
from routes.prediction_routes import prediction_bp
from routes.user_routes import user_bp
from routes.visualization_routes import visualization_bp


def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    cors.init_app(app)

    app.register_blueprint(device_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(visualization_bp)

    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    with app.app_context():
        db.create_all()

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
