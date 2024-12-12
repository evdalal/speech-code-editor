from flask import Flask

def create_app():
    app = Flask(__name__)
    from app.views import app_views
    app.register_blueprint(app_views)
    return app
