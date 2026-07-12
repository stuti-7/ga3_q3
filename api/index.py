import os
import json
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

    for model_name in FALLBACK_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content.strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            result = json.loads(text)
            return {
                key: result.get(key)
                for key in ("invoice_no", "date", "vendor", "amount", "tax", "currency")
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
