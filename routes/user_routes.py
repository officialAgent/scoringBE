from flask import Blueprint, jsonify, request
from database.models import User, BillingProfile
from extensions import db
from extensions import fake
from utils.keycloak_utils import create_keycloak_user
import random

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    result = []
    for user in users:
        billing = BillingProfile.query.filter_by(user_id=user.id).first()
        result.append({
            "user": {
                "id": user.id,
                "customer_number": user.customer_number,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "gender": user.gender,
                "age": user.age
            },
            "billing_profile": {
                "id": billing.id if billing else None,
                "user_id": billing.user_id if billing else None,
                "customer_duration": billing.customer_duration if billing else None,
                "avg_monthly_bill_eur": float(billing.avg_monthly_bill_eur) if billing else None,
                "billing_delay_count": billing.billing_delay_count if billing else None,
                "crif_score": billing.crif_score if billing else None,
                "payment_delay_count": billing.payment_delay_count if billing else None,
                "payment_delay_total_days": billing.payment_delay_total_days if billing else None
            }
        })
    return jsonify(result)

@user_bp.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.get_json()
    user_data = data.get("user")
    billing_data = data.get("billing_profile")

    user = User.query.get_or_404(user_id)

    for key in ['first_name', 'last_name', 'gender', 'age', 'customer_number']:
        if key in user_data:
            setattr(user, key, user_data[key])

    billing = BillingProfile.query.filter_by(user_id=user.id).first()

    if billing and billing_data:
        for key in ['customer_duration', 'avg_monthly_bill_eur', 'billing_delay_count', 'crif_score', 'payment_delay_count', 'payment_delay_total_days']:
            if key in billing_data:
                setattr(billing, key, billing_data[key])

    db.session.commit()

    return jsonify({"message": "User and billing profile updated successfully"})



@user_bp.route('/generateUser', methods=['GET'])
def generate_fake_user():
    user = {
        "customer_number": fake.unique.bothify(text='CUST####'),
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "gender": random.choice(["Male", "Female"]),
        "age": random.randint(18, 85)
    }
    billing = {
        "customer_duration": random.randint(1, 100),
        "avg_monthly_bill_eur": round(random.uniform(20, 150), 2),
        "billing_delay_count": random.randint(0, 15),
        "crif_score": random.randint(0, 100),
        "payment_delay_count": random.randint(0, 100),
        "payment_delay_total_days": random.randint(0, 100)
    }
    return jsonify({"user": user, "billing_profile": billing})



def save_user_and_billing(data):
    user_data = data.get("user")
    billing_data = data.get("billing_profile")

    if not user_data or not billing_data:
        return jsonify({"error": "Missing user or billing_profile data"}), 400

    try:
        user = User(
            customer_number=user_data["customer_number"],
            first_name=user_data["first_name"],
            last_name=user_data["last_name"],
            gender=user_data["gender"],
            age=user_data["age"]
        )
        db.session.add(user)
        db.session.commit()

        billing = BillingProfile(
            user_id=user.id,
            customer_duration=billing_data["customer_duration"],
            avg_monthly_bill_eur=billing_data["avg_monthly_bill_eur"],
            billing_delay_count=billing_data["billing_delay_count"],
            crif_score=billing_data["crif_score"],
            payment_delay_count=billing_data.get("payment_delay_count", 0),
            payment_delay_total_days=billing_data.get("payment_delay_total_days", 0)
        )
        db.session.add(billing)
        db.session.commit()

        create_keycloak_user(
            username=user.last_name,
            first_name=user.first_name,
            last_name=user.last_name,
            customer_number=user.customer_number,
            password="user"
        )

        return jsonify({"message": "Data saved successfully"}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


    # ... your other endpoints like /users, /generateUser, etc.

@user_bp.route('/saveUser', methods=['POST'])
def save_user():
        data = request.get_json()

        user_data = data.get("user")
        billing_data = data.get("billing_profile")

        if not user_data or not billing_data:
            return jsonify({"error": "Missing user or billing_profile data"}), 400

        try:
            # Save user to DB
            user = User(
                customer_number=user_data["customer_number"],
                first_name=user_data["first_name"],
                last_name=user_data["last_name"],
                gender=user_data["gender"],
                age=user_data["age"]
            )
            db.session.add(user)
            db.session.commit()

            # Save billing profile
            billing = BillingProfile(
                user_id=user.id,
                customer_duration=billing_data["customer_duration"],
                avg_monthly_bill_eur=billing_data["avg_monthly_bill_eur"],
                billing_delay_count=billing_data["billing_delay_count"],
                crif_score=billing_data["crif_score"],
                payment_delay_count=billing_data.get("payment_delay_count", 0),
                payment_delay_total_days=billing_data.get("payment_delay_total_days", 0)
            )
            db.session.add(billing)
            db.session.commit()

            # Create Keycloak user
            create_keycloak_user(
                username=user.last_name,
                first_name=user.first_name,
                last_name=user.last_name,
                customer_number=user.customer_number,
                password="user"
            )

            return jsonify({"message": "Data saved successfully"}), 201

        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500
