from flask import Blueprint, request, jsonify
from services.prediction_service import make_prediction
from database.models import TrainedModel
from models import load_and_predict
from utils.auth_utils import decode_token
from database.models import User, BillingProfile, DeviceCatalog, LoanApplication, LoanPrediction, TrainedModel
from extensions import db

prediction_bp = Blueprint('prediction_bp', __name__)

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
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
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@prediction_bp.route('/scoring', methods=['POST'])
def scoring():
    try:
        data = request.json

        token = data.get('token')
        device_id = data.get('device_id')
        loan_duration = data.get('Loan_Duration')
        monthly_installment = data.get('Monthly_Installment_EUR')

        if not token or not device_id or loan_duration is None or monthly_installment is None:
            return jsonify({'error': 'Missing required fields'}), 400

        decoded_token = decode_token(token)
        customer_number = decoded_token.get('customerNumber')

        if not customer_number:
            return jsonify({'error': 'Missing customerNumber in token'}), 400

        user = User.query.filter_by(customer_number=customer_number).first()
        billing = BillingProfile.query.filter_by(user_id=user.id).first()
        device = DeviceCatalog.query.get(device_id)

        if not user or not billing or not device:
            return jsonify({'error': 'Required user, billing, or device data not found'}), 404

        model_input = {
            "Age": user.age,
            "Gender": user.gender,
            "Customer_Duration": billing.customer_duration,
            "Avg_Monthly_Bill_EUR": float(billing.avg_monthly_bill_eur),
            "Billing_Delay_Count": billing.billing_delay_count,
            "crifScore": billing.crif_score,
            "Device_Price_EUR": float(device.price),
            "Loan_Price_EUR": float(device.price),
            "Loan_Duration": loan_duration,
            "Monthly_Installment_EUR": monthly_installment,
            "Payment_Delay_Count": billing.payment_delay_count or 0,
            "Payment_Delay_Total_Days": billing.payment_delay_total_days or 0
        }

        active_model = TrainedModel.query.filter_by(status='active').first()
        if not active_model:
            return jsonify({'error': 'No active model available'}), 500

        prediction, probability = load_and_predict('models', active_model.model_name, model_input)

        loan_app = LoanApplication(
            user_id=user.id,
            device_type=device.name,
            device_price_eur=device.price,
            loan_price_eur=device.price,
            loan_duration=loan_duration,
            monthly_installment_eur=monthly_installment,
            loan_status='Pending',
            loan_settled=None
        )
        db.session.add(loan_app)
        db.session.flush()

        loan_pred = LoanPrediction(
            loan_application_id=loan_app.id,
            predicted_approved=bool(prediction),
            confidence_score=probability,
            model_version=active_model.model_name
        )
        db.session.add(loan_pred)
        db.session.commit()

        return jsonify({
            'prediction': int(prediction),
            'probability_settled': float(probability),
            'model_used': active_model.model_name,
            'loan_application_id': loan_app.id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
