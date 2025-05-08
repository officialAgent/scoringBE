from extensions import db

class DeviceCatalog(db.Model):
    __tablename__ = 'device_catalog'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    status = db.Column(db.Enum('Active', 'Inactive'), default='Active')
    description = db.Column(db.Text)
    image_url = db.Column(db.Text)
    long_description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class ModelPerformance(db.Model):
    __tablename__ = 'model_performance'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float)
    roc_auc = db.Column(db.Float)
    precision_1 = db.Column(db.Float)
    recall_1 = db.Column(db.Float)
    f1_score_1 = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class ModelVisualization(db.Model):
    __tablename__ = 'model_visualizations'
    id = db.Column(db.Integer, primary_key=True)
    visualization_type = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.Text, nullable=False)
    related_models = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class TrainedModel(db.Model):
    __tablename__ = 'trained_models'
    id = db.Column(db.BigInteger, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    version = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    status = db.Column(db.String(10))
    model_path = db.Column(db.Text)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    customer_number = db.Column(db.String(50), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(20))
    age = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class BillingProfile(db.Model):
    __tablename__ = 'billing_profiles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    customer_duration = db.Column(db.Integer)
    avg_monthly_bill_eur = db.Column(db.Numeric(10, 2))
    billing_delay_count = db.Column(db.Integer)
    crif_score = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    payment_delay_count = db.Column(db.Integer)
    payment_delay_total_days = db.Column(db.Integer)

class LoanApplication(db.Model):
    __tablename__ = 'loan_applications'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    device_type = db.Column(db.String(100))
    device_price_eur = db.Column(db.Numeric(10, 2))
    loan_price_eur = db.Column(db.Numeric(10, 2))
    loan_duration = db.Column(db.Integer)
    monthly_installment_eur = db.Column(db.Numeric(10, 2))
    payment_delay_count = db.Column(db.Integer)
    payment_delay_total_days = db.Column(db.Integer)
    loan_status = db.Column(db.Enum('Pending', 'Approved', 'Rejected'), default='Pending')
    loan_settled = db.Column(db.Boolean, nullable=True, default=None)
    created_at = db.Column(db.DateTime, server_default=db.func.current_timestamp())

class LoanPrediction(db.Model):
    __tablename__ = 'loan_predictions'
    id = db.Column(db.Integer, primary_key=True)
    loan_application_id = db.Column(db.Integer, db.ForeignKey('loan_applications.id'))
    predicted_approved = db.Column(db.Boolean)
    confidence_score = db.Column(db.Numeric(5, 4))
    model_version = db.Column(db.String(50))
    prediction_time = db.Column(db.DateTime, server_default=db.func.current_timestamp())