## Base URL Prefix

All routes are under `/auth`

---

## 1. **POST /auth/signup**  
**Description:** Create a new user account _inactive_ by default, send an activation email.  
**Activation Token Expiry:** 24 hours

**Request JSON Body:**
```json
{
  "username": "johndoe",
  "email":    "john@example.com",
  "password": "yourpassword"
}
```

**Responses:**
- `201 Created`  
  ```json
  {
    "message": "User created. Activation email sent."
  }
  ```
- `400 Bad Request` – Missing one or more of `username`, `email`, or `password`.  
- `409 Conflict` – Username or email already exists.  
- `500 Internal Server Error` – Signup or email delivery failed.

---

## 2. **GET /auth/activate**  
**Description:** Activate a newly created account.  
**Token Expiry:** 24 hours

**Query Parameters:**  
- `token` (string, required) — the itsdangerous activation token from the signup email.

**Responses:**
- `200 OK`  
  ```json
  {
    "message":   "Account activated successfully.",
    "activated": true
  }
  ```
- `400 Bad Request`  
  - Missing `token`:  
    ```json
    { "message": "Activation token required." }
    ```
  - Expired token (older than 24 h):  
    ```json
    { "message": "Activation link expired." }
    ```
  - Invalid token:  
    ```json
    { "message": "Invalid activation token." }
    ```
- `500 Internal Server Error` – Failed to update user in database.

---

## 3. **POST /auth/login**  
**Description:** Authenticate an **activated** user and return JWT tokens.  
> **Note:** Only users with `Active = 1` may log in.  
**Access Token Expiry:** 1 hour  
**Refresh Token Expiry:** 30 days (default)

**Request JSON Body:**
```json
{
  "username": "johndoe",
  "password": "yourpassword"
}
```

**Responses:**
- `200 OK` – Returns both tokens:
  ```json
  {
    "access_token":  "...",   // valid for 1 hour
    "refresh_token": "..."    // valid for 30 days
  }
  ```
- `400 Bad Request` – Missing `username` or `password`.  
- `401 Unauthorized` – Invalid credentials or account not activated.  
- `500 Internal Server Error` – Database error.

---

## 4. **POST /auth/refresh**  
**Description:** Generate a new access token using a valid refresh token.  
**New Access Token Expiry:** 1 hour

**Headers:**
```
Authorization: Bearer <refresh_token>
```

**Responses:**
- `200 OK`  
  ```json
  {
    "access_token": "..."  // new token valid for 1 hour
  }
  ```
- `401 Unauthorized` – Invalid or missing refresh token.

---

## 5. **POST /auth/forget-password**  
**Description:** Initiate password reset by generating a 15-minute, itsdangerous-signed token, sending it via email, and returning it in JSON.  
**Reset Token Expiry:** 15 minutes

**Request JSON Body:**
```json
{
  "email": "john@example.com"
}
```

**Responses:**
- `200 OK`  
  ```json
  {
    "message": "If that email exists, a reset link has been sent."  // valid for 15 minutes
  }
  ```
- `400 Bad Request` – Missing `email`.  
- `404 Not Found` – No user with that email.  
- `500 Internal Server Error` – Database or email error.

---

## 6. **POST /auth/reset-password**  
**Description:** Reset the user’s password using the 15-minute reset token.  

**Request JSON Body:**
```json
{
  "reset_token":  "...",        // from /auth/forget-password
  "new_password": "newpassword"
}
```

**Responses:**
- `200 OK`  
  ```json
  {
    "message": "Password reset successful."
  }
  ```
- `400 Bad Request` – Missing token or new password.  
- `401 Unauthorized`  
  - Expired token (> 15 min old):  
    ```json
    { "message": "Reset token expired." }
    ```
  - Invalid token:  
    ```json
    { "message": "Invalid reset token." }
    ```
- `500 Internal Server Error` – Database error.

---

## 7. **POST /auth/logout**  
**Description:** Log out a user by blacklisting their current access token.  
(The blacklisted token remains invalid until its original expiry.)

**Headers:**
```
Authorization: Bearer <access_token>
```

**Responses:**
- `200 OK`  
  ```json
  {
    "message": "Logged out successfully."
  }
  ```
- `401 Unauthorized` – Token invalid or missing.  
- `500 Internal Server Error` – Database error.