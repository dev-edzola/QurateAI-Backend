**API Documentation**

---

### 1. `POST /make-call`
**Description:** Initiates a phone call with dynamic or predefined form fields using Twilio.

**Headers:**
- `Authorization: Bearer <access_token>`

**Request JSON or Form Data:**
```json
{
  "to": "+1234567890",
  "form_fields_id": 12
}
```

**Response:**
```json
{
  "message": "Call initiated with SID: CAxxxx",
  "call_id": "uuid-xxxx",
  "form_fields": [ ... ]
}
```

**Error Codes:**  
- `400` – Missing phone number or form ID  
- `500` – Twilio error  

---

### 2. `POST /generate_form_fields`
**Description:** Generates LLM-driven form fields from a user query and stores them.

**Headers:**
- `Authorization: Bearer <access_token>`

**Request Body:**
```json
{
  "user_query": "Book a meeting with a dentist",     // required: text to parse
  "form_field_name": "Dental Meeting",               // required: unique name for this form
  "form_context": { ... }                            // optional: any JSON context you want saved
}
```

**Response:** `201 Created`
```json
{
  "form_link": "https://{host}/chat?form_fields_id=13",
  "form_fields_id": 13,
  "form_fields": [
    {
      "id": "date",
      "label": "Appointment Date",
      "type": "date",
      "required": true
    },
    {
      "id": "time",
      "label": "Appointment Time",
      "type": "time",
      "required": true
    },
    ...
  ]
}
```

**Error Codes:**  
- `400` – Missing `user_query` or `form_field_name`, or no fields generated  
- `409` – A form with this name already exists (returns `form_fields_id`)  
- `500` – Database or LLM error  

---

### 3. `GET /forms`
**Description:** List all **active** form configurations for the current user, ordered by most recent update.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:** `200 OK`
```json
[
  {
    "id": 13,
    "form_field_name": "Dental Meeting",
    "form_fields": [ ... ],
    "is_active": 1,
    "created_at": "2025-04-27T12:34:56Z",
    "updated_at": "2025-04-28T09:10:11Z"
  },
  ...
]
```

**Error Codes:**  
- `500` – Database error  

---

### 4. `GET /all_forms`
**Description:** List **all** form configurations (active and inactive) for the current user, ordered by most recent update.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:** `200 OK`
```json
[
  {
    "id": 11,
    "form_field_name": "Old Survey",
    "form_fields": [ ... ],
    "is_active": 0,
    "created_at": "2025-03-15T08:00:00Z",
    "updated_at": "2025-04-10T14:22:33Z"
  },
  ...
]
```

**Error Codes:**  
- `500` – Database error  

---

### 5. `GET /forms/<form_id>`
**Description:** Retrieve a specific form configuration by its ID.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:** `200 OK`
```json
{
  "id": 13,
  "form_field_name": "Dental Meeting",
  "form_context": { ... },
  "form_fields": [ ... ],
  "is_active": 1,
  "created_at": "2025-04-27T12:34:56Z",
  "updated_at": "2025-04-28T09:10:11Z"
}
```

**Error Codes:**  
- `404` – Form not found or inactive  
- `500` – Database error  

---

### 6. `PATCH /forms/<form_id>`
**Description:** Update one or more properties of a form configuration.

**Headers:**
- `Authorization: Bearer <access_token>`

**Request Body:** (any subset of fields below)
```json
{
  "form_field_name": "Updated Name",    // optional
  "form_context": { ... },              // optional
  "form_fields": [ ... ],               // optional: full new fields array
  "is_active": 0                        // optional: 1 = active, 0 = inactive
}
```

**Response:** `200 OK`
```json
{
  "message": "Form updated successfully."
}
```

**Error Codes:**  
- `400` – No valid fields provided  
- `404` – Form not found or no permission  
- `500` – Database error  

---

### 7. `DELETE /forms/<form_id>`
**Description:** Soft-delete (deactivate) a form configuration.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:** `200 OK`
```json
{
  "message": "Form marked as inactive."
}
```

**Error Codes:**  
- `404` – Form not found or no permission  
- `500` – Database error  

---

### 8. `POST /collect`
**Description:** Continue a conversational flow by sending a user’s answer to a specific form field and receiving the next prompt.

**Authentication:** None

**Request Body:**
```json
{
  "form_fields_id": 3,
  "communication_id": 45,
  "field_id": "date",
  "question": "When do you want to schedule it?",
  "answer": "Tomorrow at 4 PM"
}
```

**Response:** `200 OK`
```json
{
  "form_fields_id": 3,
  "communication_id": 45,
  "field_id": "name",
  "question": "What is your name?",
  "field_parsed_answers": {
    "date": "2025-04-29",
    "time": "16:00"
  }
}
```

**Error Codes:**  
- `400` – Missing ID or malformed data  
- `500` – Database error  
```