from flask import Blueprint, jsonify
from database.models import ModelVisualization

visualization_bp = Blueprint('visualization_bp', __name__)


@visualization_bp.route('/visualization/roc-curve', methods=['GET'])
def get_roc_curve():
    viz = ModelVisualization.query.filter_by(visualization_type='roc_curve') \
        .order_by(ModelVisualization.created_at.desc()).first()

    if not viz:
        return jsonify({"error": "No ROC curve available"}), 404

    return jsonify({
        "visualization_type": viz.visualization_type,
        "file_path": viz.file_path.replace("\\", "/"),
        "related_models": viz.related_models,
        "created_at": viz.created_at.strftime('%Y-%m-%d %H:%M:%S')
    })


@visualization_bp.route('/visualization/model-comparison', methods=['GET'])
def get_model_comparison():
    viz = ModelVisualization.query.filter_by(visualization_type='model_comparison') \
        .order_by(ModelVisualization.created_at.desc()).first()

    if not viz:
        return jsonify({"error": "No model comparison available"}), 404

    return jsonify({
        "visualization_type": viz.visualization_type,
        "file_path": viz.file_path.replace("\\", "/"),
        "related_models": viz.related_models,
        "created_at": viz.created_at.strftime('%Y-%m-%d %H:%M:%S')
    })
