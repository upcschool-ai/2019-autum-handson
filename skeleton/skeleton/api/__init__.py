from flask import Blueprint

from . import views

blueprint = Blueprint('skeleton', __name__, url_prefix='')
blueprint.add_url_rule('/ping',
                       view_func=views.ping,
                       methods=['GET'])
blueprint.add_url_rule('/classify',
                       view_func=views.classify,
                       methods=['POST'])
