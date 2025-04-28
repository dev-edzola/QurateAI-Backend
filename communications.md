# Communications Endpoints

This document covers the two Flask routes for listing and exporting communications data.

---

## 1. List Communications

`GET /communications`

Retrieves a list of communications for the authenticated user, excluding those with status `Not Started`.

### Authorization

| Header        | Value                   |
|---------------|-------------------------|
| Authorization | `Bearer <access_token>` |

### Query Parameters

| Parameter       | Type    | Required | Description                        |
|-----------------|---------|----------|------------------------------------|
| `form_fields_id`| integer | No       | Filter by a specific form_fields.  |

### Responses

#### 200 OK

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

#### 500 Internal Server Error

```json
{
  "error": "<error message>"
}
```

---

## 2. Export Communications as CSV

`GET /communications/export/<int:form_fields_id>`

Downloads all communications associated with the specified `form_fields_id` in CSV format. The CSV will include dynamic columns for each key found in `field_parsed_answers`.

### Path Parameters

| Name             | Type    | Required | Description                        |
|------------------|---------|----------|------------------------------------|
| `form_fields_id` | integer | Yes      | ID of the form_fields to export.   |

### Responses

#### 200 OK (text/csv)

- **Headers:**
  - `Content-Disposition: attachment; filename=communications.csv`
  - `Content-Type: text/csv`

The CSV columns will be:
```
communication_id,communication_type,form_field_name,<dynamic field keys>,updated_at,communication_status
```

#### 500 Internal Server Error

```json
{
  "error": "<error message>"
}
```

---

## Implementation Details

- **Blueprint**: `communications_bp`
- **Authentication**: JWT via `@jwt_required()`

### List Endpoint
- **Route**: `/communications` (GET)
- **Database**:
  - **Table**: `communications` (alias `c`)
  - **Join**: `form_fields` (alias `f`) on `c.form_fields_id = f.id`
  - **Filters**:
    - `f.user_id = get_jwt_identity()`
    - `c.communication_status != 'Not Started'`
    - Optional `c.form_fields_id = <form_fields_id>`
  - **Ordering**: `c.updated_at DESC`

### Export Endpoint
- **Route**: `/communications/export/<int:form_fields_id>` (GET)
- **Database**:
  - **Table**: `communications` (alias `c`)
  - **Join**: `form_fields` (alias `f`) on `c.form_fields_id = f.id`
  - **Filters**:
    - `f.user_id = get_jwt_identity()`
    - `c.communication_status != 'Not Started'`
    - `c.form_fields_id = <form_fields_id>`
  - **Ordering**: `c.updated_at DESC`
- **Data Processing**:
  - **`_safe_json_load`** to parse JSON columns
  - **Pandas** DataFrame to assemble records
  - **`fillna('')`** to replace missing values with empty strings
  - **CSV export** via `DataFrame.to_csv()`

### Utilities

- **`_safe_json_load(value)`** — Safely parse JSON strings into Python objects, returns `{}` if empty or invalid.
- **`_format_datetime(dt)`** — Convert `datetime` object to ISO 8601 string.


