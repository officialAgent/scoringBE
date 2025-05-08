import os

DATABASE_URI = 'mysql+pymysql://root:root_password@localhost/scoring'
MODEL_DIR = "models"
DATA_PATH = "data/credit_scoring_data.csv"
PLOTS_DIR = "plots"
KEYCLOAK_REALM = 'master'
KEYCLOAK_HOST = 'http://localhost:8080'
KEYCLOAK_AUDIENCE = 'account'
KEYCLOAK_CERTS_URL = f"{KEYCLOAK_HOST}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
