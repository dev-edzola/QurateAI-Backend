# Text Chat API Documentation

## Base URL

```
https://<YOUR_API_HOST>/api/text-chat
```

*All endpoints are relative to this base URL.*

---

## Authentication

* **`POST /collect`**: No authentication required.
* **`PATCH /communication/metadata`**: Requires a valid JWT access token in the `Authorization` header.

  ```http
  Authorization: Bearer <JWT_ACCESS_TOKEN>
  ```

---

## Endpoints

### 1. `POST /collect`

Continues or starts a conversational flow by managing a communication session, parsing answers, returning the next question, and (upon completion) optionally sending parsed results to a callback.

#### Request

* **URL**: `/collect`
* **Method**: `POST`
* **Headers**: `Content-Type: application/json`
* **Body Parameters**:

  | Field              | Type    | Required | Description                                                                                  |
  | ------------------ | ------- | -------- | -------------------------------------------------------------------------------------------- |
  | `form_fields_id`   | integer | Yes      | ID of the form definition to use for this session.                                           |
  | `communication_id` | integer | No       | Existing session ID (omit to start a new session).                                           |
  | `answer`           | string  | No       | User’s answer to the previous question.                                                      |
  | `question`         | string  | No       | The previous question text (used as the key for collected answers).                          |
  | `field_id`         | string  | No       | Identifier of the field associated with the answer.                                          |
  | `reset`            | boolean | No       | If `true` with a valid `communication_id`, resets the session and restarts the conversation. |

##### Example: Start a new session (initial message)

```bash
curl -X POST https://api.example.com/api/text-chat/collect \
  -H "Content-Type: application/json" \
  -d '{
        "form_fields_id": 42
      }'
```

##### Example: Continue existing session with answer

```bash
curl -X POST https://api.example.com/api/text-chat/collect \
  -H "Content-Type: application/json" \
  -d '{
        "form_fields_id": 42,
        "communication_id": 101,
        "answer": "john.doe@example.com",
        "question": "What is your email address?",
        "field_id": "email"
      }'
```

##### Example: Reset an existing session

```bash
curl -X POST https://api.example.com/api/text-chat/collect \
  -H "Content-Type: application/json" \
  -d '{
        "form_fields_id": 42,
        "communication_id": 101,
        "reset": true
      }'
```

#### Response

* **Status Code**: `200 OK`

* **Body**:

  | Field                  | Type           | Description                                                            |
  | ---------------------- | -------------- | ---------------------------------------------------------------------- |
  | `form_fields_id`       | integer        | Echoes the form definition ID.                                         |
  | `communication_id`     | integer        | Session identifier (create or existing).                               |
  | `message`              | string         | The next question or final summary message.                            |
  | `field_id`             | string or null | ID of the field for the `message`; `null` if conversation is complete. |
  | `field_parsed_answers` | object         | All parsed answers collected so far (keyed by field IDs).              |

* **Behavior on Completion**:

  If `field_id` is `null` **and** a `callback_url` was configured for the form, the API will POST to `callback_url` with the JSON payload:

  ```json
  {
    "field_parsed_answers": { /* parsed answers by field */ },
    "source_id": "<source_id value>"
  }
  ```

* **`400 Bad Request`**

  * Missing or invalid `form_fields_id` or `communication_id`.

* **`500 Internal Server Error`**

  * Unexpected server error; check server logs for details.

---

### 2. `PATCH /communication/metadata`

Creates or updates metadata for an existing communication session.

#### Request

* **URL**: `/communication/metadata`
* **Method**: `PATCH`
* **Headers**:

  * `Content-Type: application/json`
  * `Authorization: Bearer <JWT_ACCESS_TOKEN>`
* **Body Parameters**:

  | Field                   | Type    | Required | Description                                                                   |
  | ----------------------- | ------- | -------- | ----------------------------------------------------------------------------- |
  | `form_fields_id`        | integer | Yes      | ID of the form definition to associate.                                       |
  | `communication_type`    | string  | Yes      | Communication mode: `text_chat` or `phone_call`.                              |
  | `communication_id`      | integer | No       | Existing session ID (omit to create a new session).                           |
  | `communication_context` | string  | No       | Arbitrary context to prepend to future AI prompts.                            |
  | `source_id`             | string  | No       | External identifier for linking with other systems.                           |

##### Example Request (Create)

```bash
curl -X PATCH https://api.example.com/api/text-chat/communication/metadata \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJI..." \
  -d '{
        "form_fields_id": 42,
        "communication_type": "text_chat",
        "communication_context": "<context>"
      }'
```

##### Example Request (Update)

```bash
curl -X PATCH https://api.example.com/api/text-chat/communication/metadata \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJI..." \
  -d '{
        "form_fields_id": 42,
        "communication_type": "text_chat",
        "communication_id": 101,
        "communication_context": "User prefers email follow-ups",
        "source_id": "CRM-555"
      }'
```

#### Response

* **Status Code**: `200 OK`
* **Body**:

Returns the frontend chat URL (with newly created or updated `communication_id`).

```json
{
  "url": "https://app.example.com/chat?form_fields_id=42&communication_id=101"
}
```

#### Error Responses

* **`400 Bad Request`**

  * Missing or invalid `form_fields_id`, `communication_type`, or `communication_id`.

* **`401 Unauthorized`**

  * Missing or invalid JWT access token.

* **`500 Internal Server Error`**

  * Unexpected server error.

---

## Common Error Codes

| HTTP Status | Code           | Message                        | Description                                                  |
| ----------- | -------------- | ------------------------------ | ------------------------------------------------------------ |
| 400         | `BAD_REQUEST`  | `Invalid form_fields id`       | The provided `form_fields_id` does not exist or is inactive. |
| 400         | `BAD_REQUEST`  | `Invalid communication_id`     | The provided `communication_id` does not exist.              |
| 401         | `UNAUTHORIZED` | `Missing Authorization Header` | No JWT token provided for secured endpoint.                  |
| 500         | `SERVER_ERROR` | `<error details>`              | Generic server error; see logs for details.                  |
