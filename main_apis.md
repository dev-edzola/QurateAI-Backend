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

**Error Codes:** 400 (Missing phone number or form ID), 500 (Twilio error)

---

### 2. `POST /generate_form_fields`
**Description:** Generates and stores form fields based on a user query.

**Headers:**
- `Authorization: Bearer <access_token>`

**Request Body:**
```json
{
  "user_query": "Book a meeting with a dentist",
  "form_field_name": "Dental Meeting"
}
```

**Response:**
```json
{
  "form_link": "https://{host}/chat?form_fields_id=13",
  "form_fields_id": 13,
  "form_fields": [ ... ]
}
```

**Error Codes:** 400 (Missing input), 409 (Duplicate form), 500 (DB Error)

---

### 3. `GET /forms`
**Description:** Retrieves all active form field configurations.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:**
```json
[
  {
    "id": 1,
    "form_field_name": "Demo",
    "form_fields": [...],
    ...
  }
]
```

**Error Codes:** 500 (DB Error)

---

### 3.1. `GET /all_forms`
**Description:** Retrieves all form field configurations.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:**
```json
[
  {
    "id": 1,
    "form_field_name": "Demo",
    "form_fields": [...],
    ...
  }
]
```

**Error Codes:** 500 (DB Error)

---



### 4. `GET /forms/<form_id>`
**Description:** Retrieves a specific form field configuration by ID.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:**
```json
{
  "id": 1,
  "form_field_name": "Demo",
  "form_fields": [...]
}
```

**Error Codes:** 404 (Not found), 500 (DB Error)

---

### 5. `PATCH /forms/<form_id>`
**Description:** Updates specific fields of a form field configuration.

**Headers:**
- `Authorization: Bearer <access_token>`

**Request Body (any subset):**
```json
{
  "form_field_name": "Updated Name",
  "form_fields": [ ... ],
  "is_active": true
}
```

**Response:**
```json
{
  "message": "Form updated successfully."
}
```

**Error Codes:** 400 (Invalid input), 500 (DB Error)

---

### 6. `DELETE /forms/<form_id>`
**Description:** Soft deletes (deactivates) a form configuration.

**Headers:**
- `Authorization: Bearer <access_token>`

**Response:**
```json
{
  "message": "Form marked as inactive."
}
```

**Error Codes:** 500 (DB Error)

---

### 7. `POST /collect`
**Description:** Handles a conversational interaction by collecting answers to dynamic form fields and advancing the chat.

**Authentication:** None (assumes public)

**Request Body:**
```json
{
  "form_fields_id": 3,
  "communication_id": 45,
  "answer": "Tomorrow at 4 PM",
  "field_id": "date",
  "question": "When do you want to schedule it?"
}
```

**Response:**
```json
{
  "form_fields_id": 3,
  "message": "What is your name?",
  "field_id": "name",
  "field_parsed_answers": { ... },
  "communication_id": 45
}
```

**Error Codes:** 400 (Missing ID or bad data), 500 (DB Error)

