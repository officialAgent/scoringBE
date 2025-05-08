from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from faker import Faker
from concurrent.futures import ThreadPoolExecutor

db = SQLAlchemy()
cors = CORS(resources={r"/*": {"origins": "*"}})
fake = Faker()
executor = ThreadPoolExecutor()
