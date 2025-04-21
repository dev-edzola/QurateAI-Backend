# Communications API Documentation

## Overview
The Communications API allows authenticated users to retrieve records of their interactions (communications) including raw transcripts (`collected_answers`) and structured parsed responses (`field_parsed_answers`). Results are sorted by most recently updated.

---

## Authentication
All endpoints require a valid JSON Web Token (JWT) in the `Authorization` header.

```
Authorization: Bearer <YOUR_JWT_TOKEN>
```

Requests without a valid token will receive a **401 Unauthorized** response.

---

## Base URL
```
https://{API_HOST}
```
Replace `{API_HOST}` with your server’s domain or IP.

---

## Endpoints

### GET /communications
Retrieve all communications for the current user, optionally filtered by form.

#### Request
```
GET /communications?form_fields_id={form_fields_id}
Host: {API_HOST}
Authorization: Bearer <JWT>
Accept: application/json
```

- **Query Parameters**
  - `form_fields_id` (optional, integer): If provided, only communications associated with this form will be returned. Omit to fetch all communications for the user.

#### Response
**Status Code**: `200 OK`

```json
{
  "communications": [
    {
      "communication_id": 123,
      "form_fields_id": 45,
      "collected_answers": { /* transcript text or JSON */ },
      "field_parsed_answers": { /* parsed JSON responses */ },
      "updated_at": "2025-04-21T12:34:56.789Z"
    },
    {
      "communication_id": 122,
      "form_fields_id": 45,
      "collected_answers": null,
      "field_parsed_answers": {...},
      "updated_at": "2025-04-20T11:22:33.444Z"
    }
    // … more records …
  ]
}
```

- **Fields**:
  - `communication_id` (integer): Unique identifier of the communication record.
  - `form_fields_id` (integer): ID of the form definition used.
  - `collected_answers` (object|null): Raw transcript or text of the communication; may be JSON or plain text.
  - `field_parsed_answers` (object|null): Structured JSON of parsed field responses.
  - `updated_at` (string): ISO‑8601 timestamp of last update, sorted descending by default.

#### Error Responses

| Status Code | Condition                              | Response Body                        |
|-------------|----------------------------------------|--------------------------------------|
| 400 Bad Request | Malformed query parameter (e.g., non-integer `form_fields_id`) | `{ "error": "..." }` |
| 401 Unauthorized | Missing or invalid JWT              | `{ "msg": "Missing Authorization Header" }` |
| 500 Internal Server Error | Database or server error       | `{ "error": "<error message>" }` |

---

## Examples

1. **Fetch all communications**

   ```bash
   curl -X GET "https://API_HOST/communications" \
        -H "Authorization: Bearer YOUR_JWT_TOKEN" \
        -H "Accept: application/json"
   ```

2. **Fetch communications for form ID 42**

   ```bash
   curl -X GET "https://API_HOST/communications?form_fields_id=42" \
        -H "Authorization: Bearer YOUR_JWT_TOKEN" \
        -H "Accept: application/json"
   ```

