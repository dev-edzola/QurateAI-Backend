## Authentication API Endpoints

This document describes the available authentication endpoints, their HTTP methods, possible status codes, and example JSON responses.

---

### 1. **POST** `/signup`
Create a new user account (inactive until activation).

**Request Body (JSON)**
```json
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "securepassword"
}
```

**Responses**

- **400 Bad Request**

  ```json
  {
    "message": "Missing required fields."
  }
  ```

- **409 Conflict**
  - If a user with the same email already exists and is active, or username conflict.

  ```json
  {
    "message": "User already exists."
  }
  ```

- **200 OK**
  - Account exists but inactive; activation link resent.

  ```json
  {
    "message": "Account already exists but is inactive; activation link resent."
  }
  ```

- **201 Created**
  - New user created; activation email sent.

  ```json
  {
    "message": "User created. Activation email sent."
  }
  ```

- **500 Internal Server Error**

  ```json
  {
    "error": "Signup failed."
  }
  ```

---

### 2. **GET** `/activate`
Activate a newly created account using token.

**Query Parameters**
- `token` (string, required): Activation token received via email.

**Responses**

- **400 Bad Request**
  - Missing token:

  ```json
  {
    "message": "Activation token required."
  }
  ```

  - Expired token:

  ```json
  {
    "message": "Activation link expired."
  }
  ```

  - Invalid token:

  ```json
  {
    "message": "Invalid activation token."
  }
  ```

- **200 OK**

  ```json
  {
    "message": "Account activated successfully.",
    "activated": true
  }
  ```

- **500 Internal Server Error**

  ```json
  {
    "error": "Account activation failed."
  }
  ```

---

### 3. **POST** `/login`
Authenticate a user and issue JWT tokens.

**Request Body (JSON)**
```json
{
  "username": "johndoe",    // or email
  "password": "securepassword"
}
```

**Responses**

- **400 Bad Request**

  ```json
  {
    "message": "Missing credentials."
  }
  ```

- **401 Unauthorized**

  ```json
  {
    "message": "Invalid credentials."
  }
  ```

- **200 OK**

  ```json
  {
    "access_token": "<jwt_access_token>",
    "refresh_token": "<jwt_refresh_token>"
  }
  ```

- **500 Internal Server Error**

  ```json
  {
    "error": "Login failed."
  }
  ```

---

### 4. **POST** `/refresh`
Obtain a new access token using a refresh token.

**Headers**
- `Authorization: Bearer <refresh_token>`

**Responses**

- **200 OK**

  ```json
  {
    "access_token": "<new_jwt_access_token>"
  }
  ```

---

### 5. **POST** `/forget-password`
Request a password reset link.

**Request Body (JSON)**
```json
{
  "email": "john@example.com"
}
```

**Responses**

- **400 Bad Request**

  ```json
  {
    "message": "Email required."
  }
  ```

- **200 OK**
  - Always returns 200 to avoid disclosing email existence.

  ```json
  {
    "message": "If that email exists, a reset link has been sent."
  }
  ```

- **500 Internal Server Error**

  ```json
  {
    "error": "Unable to send reset email."
  }
  ```

---

### 6. **POST** `/reset-password`
Reset a user’s password using reset token.

**Request Body (JSON)**
```json
{
  "reset_token": "<token>",
  "new_password": "newsecurepassword"
}
```

**Responses**

- **400 Bad Request**
  - Missing fields:

  ```json
  {
    "message": "Token and new password required."
  }
  ```

- **401 Unauthorized**
  - Token expired:

  ```json
  {
    "message": "Reset token expired."
  }
  ```

  - Invalid token:

  ```json
  {
    "message": "Invalid reset token."
  }
  ```

- **200 OK**

  ```json
  {
    "message": "Password reset successful."
  }
  ```

- **500 Internal Server Error**

  ```json
  {
    "error": "<error message>"
  }
  ```

---

### 7. **POST** `/logout`
Revoke the current access token (blacklist).

**Headers**
- `Authorization: Bearer <access_token>`

**Responses**

- **200 OK**

  ```json
  {
    "message": "Logged out successfully."
  }
  ```

- **500 Internal Server Error**

  ```json
  {
    "error": "<error message>"
  }
  ```

