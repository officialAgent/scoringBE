from flask import Blueprint, request, jsonify, send_file, send_from_directory
from services.training_service import train_models
from database.models import ModelPerformance, TrainedModel
from extensions import db
from generator import generate_credit_scoring_csv
from config import PLOTS_DIR
import os

model_bp = Blueprint('model_bp', __name__)

@model_bp.route('/train', methods=['POST'])
def train_route():
    return train_models(request.json)

@model_bp.route('/model-performance', methods=['GET'])
def get_model_performance():
    performances = ModelPerformance.query.order_by(ModelPerformance.accuracy.desc()).all()
    return jsonify([
        {
            'model_name': m.model_name,
            'accuracy': m.accuracy,
            'roc_auc': m.roc_auc,
            'precision_1': m.precision_1,
            'recall_1': m.recall_1,
            'f1_score_1': m.f1_score_1,
            'created_at': m.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for m in performances
    ])

@model_bp.route('/trained-models', methods=['GET'])
def get_trained_models():
    models = TrainedModel.query.order_by(TrainedModel.created_at.desc()).all()
    return jsonify([
        {
            'id': m.id,
            'model_name': m.model_name,
            'version': m.version,
            'created_at': m.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'status': m.status,
            'model_path': m.model_path
        } for m in models
    ])

@model_bp.route('/trained-models/<int:model_id>/activate', methods=['PUT'])
def activate_trained_model(model_id):
    model = TrainedModel.query.get_or_404(model_id)

    TrainedModel.query.update({TrainedModel.status: 'inactive'})
    model.status = 'active'
    db.session.commit()

    return jsonify({'message': f'Model {model.model_name} activated successfully'})

@model_bp.route('/trained-models/<int:model_id>', methods=['DELETE'])
def delete_trained_model(model_id):
    model = TrainedModel.query.get_or_404(model_id)

    if model.model_path and os.path.exists(model.model_path):
        os.remove(model.model_path)

    db.session.delete(model)
    db.session.commit()

    return jsonify({'message': f'Model {model.model_name} deleted successfully'})

@model_bp.route('/generate-csv', methods=['POST'])
def trigger_csv_generation():
    try:
        generate_credit_scoring_csv("data/credit_scoring_data.csv")
        return jsonify({"message": "CSV generated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@model_bp.route('/download-csv', methods=['GET'])
def download_csv():
    try:
        file_path = os.path.join("data", "credit_scoring_data.csv")
        if not os.path.exists(file_path):
            return jsonify({"error": "CSV file not found"}), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@model_bp.route('/plots/<path:filename>', methods=['GET'])
def serve_plot_file(filename):   # ✅ CHANGED function name to be UNIQUE
    return send_from_directory(PLOTS_DIR, filename)
