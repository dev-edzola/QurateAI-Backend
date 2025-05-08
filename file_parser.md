# Resume Parsing API Documentation


## Base URL

```
https://<your-domain>/api
```

> Replace `<your-domain>` with your server's hostname or IP.

---

## Authentication

All endpoints require a valid JWT access token in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

Obtain an access token via your authentication flow before calling the Resume Parsing endpoint.

---

## Endpoint: `POST /parse_resume`

Processes an uploaded resume file against a provided job description (JD) and returns:

* `user_info`: key candidate details
* `ats_score`: an integer (0–100) indicating the match quality

### Request

* **URL**: `/parse_resume`

* **Method**: `POST`

* **Headers**:

  * `Authorization: Bearer <access_token>`
  * `Content-Type: multipart/form-data`

* **Form Data**:

  | Field             | Type   | Description                                                          |
  | ----------------- | ------ | -------------------------------------------------------------------- |
  | `resume_file`     | file   | Candidate's resume file. Supported formats: `.pdf`, `.docx`, `.txt`. |
  | `job_description` | string | Plain-text job description against which to evaluate the resume.     |

#### Example (cURL)

```bash
curl -X POST \
  https://<your-domain>/api/parse_resume \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "resume_file=@/path/to/resume.pdf" \
  -F "job_description=We are looking for a Senior Python Developer..."
```

### Successful Response

* **Status**: `200 OK`
* **Body** (JSON):

```json
{
  "user_info": {
    "Name": "Jane Doe",
    "Email": "jane.doe@example.com",
    "Phone": "+1-555-1234",
    "Total Experience": 5.2,
    "Relevant Experience": 3.0,
    "Current Role": "Software Engineer",
    "Current Company": "TechCorp Inc.",
    "Skills": ["Python", "Flask", "AWS", "Docker"],
    "Education": "M.S. Computer Science, University X",
    "Notable Achievements": [
      "Led migration of monolith to microservices",
      "Speaker at PyCon 2024"
    ]
  },
  "ats_score": 82
}
```

### Error Responses

| HTTP Status | Error Code | Description                                     |
|-------------|------------|-------------------------------------------------|
|`400 Bad Request`|`Missing resume file`       | No`resume\_file`provided in the request.     |
|`400 Bad Request`|`Missing job description`   | No`job description`provided.                |
|`400 Bad Request`|`Unsupported file format`   | File format not`.pdf`, `.docx`, or `.txt`.  | 
| `500 Internal Server Error`|`Failed to parse resume` | Unexpected server error during parsing.       |

#### Example (Missing File)

```json
HTTP/1.1 400 Bad Request
{
  "error": "Missing resume_file"
}
```

---

## Implementation Details

* **Language**: Python 3.x
* **Framework**: Flask
* **File Parsing**:

  * PDF: [pdfplumber]() for text extraction
  * DOCX: `python-docx` for paragraphs
  * TXT: UTF-8 text read
* **AI Processing**:

  * `langchain` messages via `llm_reasoning` utility
  * JSON extraction through `extract_json`
* **Security**:

  * JWT validation via `flask_jwt_extended`
