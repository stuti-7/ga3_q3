import os
import re
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ["AIPIPE_TOKEN"], base_url="https://aipipe.org/openai/v1")

DEFAULT_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")
FALLBACK_MODELS = list(dict.fromkeys([DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"]))

DATE_FORMATS = [
    "%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y",
]


class InvoiceRequest(BaseModel):
    invoice_text: str


def _normalize_date(value):
    if not isinstance(value, str) or not value.strip():
        return value
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value


def _normalize_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value)
        if not cleaned or cleaned in ("-", "."):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/extract")
def extract(req: InvoiceRequest):
    prompt = f"""
Extract the following fields from the invoice text below.

Return ONLY valid JSON with exactly this schema:
{{
  "invoice_no": string|null,
  "date": string|null,
  "vendor": string|null,
  "amount": number|null,
  "tax": number|null,
  "currency": string|null
}}

Rules:
- date must be converted to ISO format YYYY-MM-DD, regardless of the input format
  (e.g. "15 March 2026" -> "2026-03-15", "03/15/2026" -> "2026-03-15").
- amount = subtotal before tax (not the grand total). If only a grand total and
  tax are given, compute amount = total - tax.
- tax = the tax amount only (e.g. GST/VAT amount), not the tax rate percentage.
  If multiple tax lines exist (e.g. CGST + SGST), sum them into a single tax value.
- currency must be a 3-letter code (e.g. INR, USD, EUR, GBP). Infer it from
  symbols if needed: Rs./₹ -> INR, $ -> USD, € -> EUR, £ -> GBP.
- amount and tax must be plain numbers with no currency symbols, commas, or units.
- If a field cannot be found in the text, use null. Do not guess.

Invoice:

{req.invoice_text}
"""

    for model_name in FALLBACK_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )

            text = response.choices[0].message.content.strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            result = json.loads(text)
            return {
                "invoice_no": result.get("invoice_no"),
                "date": _normalize_date(result.get("date")),
                "vendor": result.get("vendor"),
                "amount": _normalize_number(result.get("amount")),
                "tax": _normalize_number(result.get("tax")),
                "currency": result.get("currency"),
            }

        except Exception as e:
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
    }
