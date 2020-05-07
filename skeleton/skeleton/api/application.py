from flask import Flask

from skeleton import api


def run():
    app = Flask(__name__.split('.')[0])
    app.register_blueprint(api.blueprint)
    app.config['PROFILE'] = True
    app.run(host='0.0.0.0', debug=True, port=5067, threaded=False, use_reloader=True)


if __name__ == '__main__':
    run()
