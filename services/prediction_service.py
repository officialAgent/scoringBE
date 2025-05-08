from database.models import TrainedModel
from models import load_and_predict
from flask import jsonify

def make_prediction(data):
    input_data = data.get('input', {})
    model_name = data.get('model')

    if not input_data:
        return jsonify({"error": "Missing input data"}), 400

    if not model_name:
        active_model = TrainedModel.query.filter_by(status='active').first()
        if not active_model:
            return jsonify({"error": "No active model available"}), 400
        model_name = active_model.model_name

    try:
        pred, prob = load_and_predict('models', model_name, input_data)
        return jsonify({
            "model": model_name,
            "prediction": int(pred),
            "probability_settled": float(prob)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
