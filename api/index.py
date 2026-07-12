import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK_MODELS = list(dict.fromkeys([DEFAULT_MODEL, "gemini-flash-latest", "gemini-flash-lite-latest"]))


class InvoiceRequest(BaseModel):
    invoice_text: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/extract")
def extract(req: InvoiceRequest):
    prompt = f"""
Extract the following fields from the invoice.

Return ONLY valid JSON.

Schema:
{{
  "invoice_no": string|null,
  "date": string|null,
  "vendor": string|null,
  "amount": number|null,
  "tax": number|null,
  "currency": string|null
}}

Rules:
- date must be YYYY-MM-DD
- amount = subtotal before tax
- tax = tax amount only
- currency like INR, USD
- If missing use null.

Invoice:

{req.invoice_text}
"""

    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            text = response.text.strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            result = json.loads(text)
            for key in ("invoice_no", "date", "vendor", "amount", "tax", "currency"):
                result.setdefault(key, None)
            return result

        except Exception as e:
            last_error = e
            if "not found" in str(e).lower() or "unsupported" in str(e).lower():
                continue
            break

    return {
        "invoice_no": None,
        "date": None,
        "vendor": None,
        "amount": None,
        "tax": None,
        "currency": None,
        "error": str(last_error)
    }