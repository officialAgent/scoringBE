import requests
from jose import jwt
from config import KEYCLOAK_CERTS_URL, KEYCLOAK_AUDIENCE

def get_jwk_keys():
    response = requests.get(KEYCLOAK_CERTS_URL)
    response.raise_for_status()
    return response.json()

def decode_token(token):
    jwks = get_jwk_keys()
    return jwt.decode(
        token,
        jwks,
        algorithms=['RS256'],
        audience=KEYCLOAK_AUDIENCE
    )
