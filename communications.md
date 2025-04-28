# Communications Endpoint

`GET /communications`

Retrieves a list of communications for the authenticated user, excluding those with status `Not Started`.

---

## Authorization

| Header         | Value                      |
|----------------|----------------------------|
| Authorization  | `Bearer <access_token>`    |

---

## Query Parameters

| Parameter       | Type    | Required | Description                                      |
|-----------------|---------|----------|--------------------------------------------------|
| `form_fields_id`| integer | No       | Filter by a specific `form_fields_id`.           |

---

## Responses

### 200 OK

```json
{
  "communications": [
    {
      "communication_id": 1,
      "communication_type": "call",
      "form_fields_id": 42,
      "collected_answers": { /* ... */ },
      "field_parsed_answers": { /* ... */ },
      "updated_at": "2025-04-28T17:52:11Z",
      "communication_status": "In Progress"
    },
    {
      "communication_id": 2,
      "communication_type": "sms",
      "form_fields_id": 42,
      "collected_answers": { /* ... */ },
      "field_parsed_answers": { /* ... */ },
      "updated_at": "2025-04-28T18:00:00Z",
      "communication_status": "Completed"
    }
  ]
}
```

### 500 Internal Server Error

```json
{
  "error": "<error message>"
}
```

---

## Implementation Details

- **Blueprint**: `communications_bp`
- **Route**: `/communications` (GET)
- **Authentication**: JWT via `@jwt_required()`
- **Database**: MySQL
  - **Table**: `communications` (alias `c`)
  - **Join**: `form_fields` (alias `f`) on `c.form_fields_id = f.id`
  - **Filters**:
    - `f.user_id = get_jwt_identity()`
    - `c.communication_status != 'Not Started'`
    - Optional `c.form_fields_id = <form_fields_id>`
  - **Ordering**: `c.updated_at DESC`


### Utilities

- `_safe_json_load(value)` — Safely parse JSON strings into Python objects, returns `{}` if empty or invalid.
- `_format_datetime(dt)` — Convert `datetime` object to ISO 8601 string.

