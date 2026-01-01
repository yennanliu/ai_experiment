from fastapi import status


def get_auth_token(client, test_user_data):
    client.post("/api/v1/auth/register", json=test_user_data)
    response = client.post(
        "/api/v1/auth/login",
        data={"username": test_user_data["email"], "password": test_user_data["password"]}
    )
    return response.json()["access_token"]


def test_get_current_user(client, test_user_data):
    token = get_auth_token(client, test_user_data)
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["email"] == test_user_data["email"]
    assert data["username"] == test_user_data["username"]


def test_get_current_user_no_token(client):
    response = client.get("/api/v1/users/me")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_current_user_invalid_token(client):
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_change_password_success(client, test_user_data):
    token = get_auth_token(client, test_user_data)
    response = client.put(
        "/api/v1/users/me/password",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "current_password": test_user_data["password"],
            "new_password": "NewSecurePass456"
        }
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["message"] == "Password updated successfully"

    response = client.post(
        "/api/v1/auth/login",
        data={"username": test_user_data["email"], "password": "NewSecurePass456"}
    )
    assert response.status_code == status.HTTP_200_OK


def test_change_password_wrong_current(client, test_user_data):
    token = get_auth_token(client, test_user_data)
    response = client.put(
        "/api/v1/users/me/password",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "current_password": "WrongPassword123",
            "new_password": "NewSecurePass456"
        }
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Current password is incorrect" in response.json()["detail"]


def test_change_password_no_token(client):
    response = client.put(
        "/api/v1/users/me/password",
        json={
            "current_password": "OldPass123",
            "new_password": "NewSecurePass456"
        }
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
