import requests
from config import KEYCLOAK_HOST, KEYCLOAK_REALM

def create_keycloak_user(username, first_name, last_name, customer_number, password='user'):
    try:
        token_url = f"{KEYCLOAK_HOST}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
        users_url = f"{KEYCLOAK_HOST}/admin/realms/{KEYCLOAK_REALM}/users"

        data = {
            'grant_type': 'password',
            'client_id': 'admin-cli',
            'username': 'admin',  # 🔥 Keycloak admin username
            'password': 'admin_password',  # 🔥 Keycloak admin password
        }
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token = response.json()['access_token']

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "username": username,
            "enabled": True,
            "firstName": first_name,
            "lastName": last_name,
            "attributes": {
                "customerNumber": [customer_number]
            },
            "credentials": [{
                "type": "password",
                "value": password,
                "temporary": False
            }]
        }
        create_response = requests.post(users_url, headers=headers, json=payload)

        if create_response.status_code not in [201, 204]:
            raise Exception(f"Failed to create Keycloak user: {create_response.text}")

    except Exception as e:
        print(f"❌ Exception in create_keycloak_user: {str(e)}")
        raise
