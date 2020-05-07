import logging

from flask import Flask
from werkzeug.middleware.profiler import ProfilerMiddleware

from skeleton import api
from skeleton.inference import models

logging.basicConfig(level=logging.DEBUG)


def run():
    app = Flask(__name__.split('.')[0])
    app.register_blueprint(api.blueprint)
    app.config['PROFILE'] = True
    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[10])
    app.model = models.AlexNet()
    app.run(host='0.0.0.0', debug=True, port=5067, threaded=False, use_reloader=True)


if __name__ == '__main__':
    run()
