# API Documentation for form_fields Endpoints

This documentation covers the following endpoints:

1. **POST /generate_form_fields** – Generate new form fields.
2. **GET /forms** – Retrieve all active forms.
3. **PATCH /forms/<form_id>** – Update a form.
4. **DELETE /forms/<form_id>** – Mark a form as inactive.

> **Note:** All error responses include an `"error"` key. For example:  
> `{"error": "Missing user_query"}`

---

## 1. POST /generate_form_fields

**Description:**  
This endpoint generates a new form by parsing a user query into form fields. It then stores the new form in the database and returns the new form's ID, the generated fields, and a link to access the form.

**Request Body:**

- **user_query**: *(string, required)* A query or instruction from the user to generate the form fields.
- **form_field_name**: *(string, required)* A descriptive name for the form.

```json
{
    "user_query": "Please collect the following details: name",
    "form_field_name": "General Info"
}
```

**Success Response (HTTP 201):**

```json
{
  "form_link": "https://{frontend_host}/chat?form_fields_id=27",
  "form_fields_id": 27,
  "form_fields": [
      {
          "datatype": "string",
          "field_id": "language",
          "field_name": "Language",
          "additional_info": "This question is asked so that further communication can happen in that language"
      },
      {
          "datatype": "string",
          "field_id": "name",
          "field_name": "Name",
          "additional_info": "User's full name"
      }
  ]
}
```

**Error Responses:**

- Missing `user_query`:
  ```json
  {"error": "Missing user_query"}
  ```
- Missing `form_field_name`:
  ```json
  {"error": "Missing form_field_name"}
  ```
- No form fields generated:
  ```json
  {"error": "No form fields generated"}
  ```

**Example cURL Command:**

```bash
curl --location --request POST 'https://twilio-flask-ysez.onrender.com/generate_form_fields' \
--header 'Content-Type: application/json' \
--data '{
    "user_query": "Please collect the following details: name",
    "form_field_name": "General Info"
}'
```

---

## 2. GET /forms

**Description:**  
Retrieves all active forms from the database (i.e. forms where `is_active` equals 1).

**Success Response (HTTP 200):**

```json
[
    {
        "id": 27,
        "form_field_name": "General Info",
        "form_fields": "[{\"datatype\": \"string\", \"field_id\": \"language\", \"field_name\": \"Language\", \"additional_info\": \"...\"}, {\"datatype\": \"string\", \"field_id\": \"name\", \"field_name\": \"Name\", \"additional_info\": \"...\"}]",
        "is_active": 1
    },
    {
        "id": 28,
        "form_field_name": "Test Form",
        "form_fields": "[...]",
        "is_active": 1
    }
]
```

**Example cURL Command:**

```bash
curl --location 'https://twilio-flask-ysez.onrender.com/forms'
```

---

## 3. PATCH /forms/<form_id>

**Description:**  
Updates a form by its ID. Accepts a JSON payload that may include updates for `form_field_name`, `form_fields`, and/or `is_active`.

**Request Parameters:**

- **form_id**: *(integer, required)* The ID of the form to update (provided as a URL parameter).

**Request Body Example:**

```json
{
    "form_field_name": "New Form - Updated",
    "form_fields": [
        {
            "datatype": "string",
            "field_id": "language",
            "field_name": "Language",
            "additional_info": "This question is asked so that further communication can happen in that language"
        }
    ]
}
```

**Success Response (HTTP 200):**

```json
{
  "message": "Form updated successfully."
}
```

**Error Response:**

- If no data is provided:
  ```json
  {"error": "No data provided"}
  ```

**Example cURL Command:**

```bash
curl --location --request PATCH 'https://twilio-flask-ysez.onrender.com/forms/5' \
--header 'Content-Type: application/json' \
--data '{
    "form_field_name": "New Form - Updated",
    "form_fields": [
        {
            "datatype": "string",
            "field_id": "language",
            "field_name": "Language",
            "additional_info": "This question is asked so that further communication can happen in that language"
        }
    ]
}'
```

---

## 4. DELETE /forms/<form_id>

**Description:**  
Marks a form as inactive (i.e. "deletes" it) by setting its `is_active` value to 0. The form is not physically removed from the database.

**Request Parameters:**

- **form_id**: *(integer, required)* The ID of the form to mark as inactive.

**Success Response (HTTP 200):**

```json
{
  "message": "Form marked as inactive."
}
```

**Example cURL Command:**

```bash
curl --location --request DELETE 'https://twilio-flask-ysez.onrender.com/forms/5'
```

---

# Error Handling

- All endpoints return errors with a JSON payload containing an `"error"` key if the request is invalid or if an exception occurs. For example:
  ```json
  {"error": "Missing user_query"}
  ```
