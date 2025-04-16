# Auth API Documentation

This documentation describes the endpoints provided by the `auth.py` module for user authentication, authorization, and password management.

---

## Base URL Prefix

All routes are prefixed with `/auth`

---

## 1. **POST /auth/signup**
**Description:** Create a new user account.

**Request JSON Body:**
```json
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "yourpassword"
}
```

**Responses:**
- `201 Created` – User successfully created.
- `400 Bad Request` – Missing fields.
- `409 Conflict` – User already exists.
- `500 Internal Server Error` – Database error.

---

## 2. **POST /auth/login**
**Description:** Authenticate a user and return JWT tokens.

- **Access Token:** Valid for `1 hour`.
- **Refresh Token:** Valid for `30 days`.

**Request JSON Body:**
```json
{
  "username": "johndoe",
  "password": "yourpassword"
}
```

**Responses:**
- `200 OK` – Returns access and refresh tokens.
- `400 Bad Request` – Missing credentials.
- `401 Unauthorized` – Invalid credentials.
- `500 Internal Server Error` – Database error.

**Success Response Example:**
```json
{
  "access_token": "...",
  "refresh_token": "..."
}
```

---

## 3. **POST /auth/refresh**
**Description:** Generate a new access token using a refresh token.

**Headers:**
- `Authorization: Bearer <refresh_token>`

**Responses:**
- `200 OK` – Returns new access token.
- `401 Unauthorized` – Invalid or missing refresh token.

**Success Response Example:**
```json
{
  "access_token": "..."
}
```

---

## 4. **POST /auth/forget-password**
**Description:** Initiates password reset by generating a short-lived token.
**reset_token:** in your /auth/forget-password endpoint expires in `15 minutes`.
**Request JSON Body:**
```json
{
  "email": "john@example.com"
}
```

**Responses:**
- `200 OK` – Returns reset token.
- `400 Bad Request` – Email missing.
- `404 Not Found` – No user with that email.
- `500 Internal Server Error` – DB error.

**Success Response Example:**
```json
{
  "reset_token": "..."
}
```

---

## 5. **POST /auth/reset-password**
**Description:** Resets user password using reset token.

**Request JSON Body:**
```json
{
  "reset_token": "...", //Response from /auth/forget-password
  "new_password": "newpassword123"
}
```

**Responses:**
- `200 OK` – Password successfully reset.
- `400 Bad Request` – Missing token or password.
- `401 Unauthorized` – Expired or invalid token.
- `500 Internal Server Error` – DB error.

---

## 6. **POST /auth/logout**
**Description:** Log out a user by blacklisting their access token.

**Headers:**
- `Authorization: Bearer <access_token>`

**Responses:**
- `200 OK` – Token blacklisted.
- `401 Unauthorized` – Token invalid/missing.
- `500 Internal Server Error` – DB error.



