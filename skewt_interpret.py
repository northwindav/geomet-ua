
"""
skewt_interpret.py

# Some settings to pop into an .env file:

AZURE_INFERENCE_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_INFERENCE_KEY=<your-key>
AZURE_MODEL_DEPLOYMENT=<your-vision-deployment-name>


Reads a static Skew-T PNG, calls Azure OpenAI (vision), and returns:
  1) A readable summary tailored to your checklist
  2) Structured JSON with key fields (surface, stability, inversion, warming needed, ventilation, crossover, mixing down, summary)

Usage:
  python skewt_interpret_azure.py /path/to/skewt.png

Or import and call `interpret_skewt_with_azure(...)` from your pipeline.
"""

import os
import re
import json
import base64
from typing import Tuple, Dict, Optional

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage, UserMessage,
    ImageContentItem, TextContentItem
)


# ---------- Preferences baked into the prompt ----------
SYSTEM_PROMPT = (
    "You are M365 Copilot. Interpret Skew‑T (log‑P) soundings for operational use in metric units. "
    "Always provide TWO sections:\n"
    "1) 'Summary' — a readable forecast-style interpretation focusing on:\n"
    "   • Surface: temperature, dew point/humidity, wind.\n"
    "   • Lower-level stability; presence/absence of inversions; inversion depth and strength.\n"
    "   • If an inversion exists, estimate daytime surface warming (°C) needed to erode/break it.\n"
    "   • Ventilation synopsis: mixing height (m), transport wind (km/h), qualitative rating (poor|fair|good) with brief rationale.\n"
    "   • Near-surface crossover potential (report using both T≥RH and RH-threshold ~25–30%).\n"
    "   • Potential for stronger aloft winds to mix to the surface.\n"
    "   • Concise operational context (e.g., winter sun angle, high albedo).\n"
    "2) 'Data' — a single JSON object with keys:\n"
    "{\n"
    "  \"surface\": {\"temp_c\": number|null, \"dewpoint_c\": number|null, \"rh_percent\": number|null, \"wind_dir_deg\": number|null, \"wind_speed_kmh\": number|null},\n"
    "  \"stability\": {\"inversion_present\": true|false|null, \"base_hpa\": number|null, \"top_hpa\": number|null, \"strength_c\": number|null, \"depth_m\": number|null},\n"
    "  \"warming_required_c\": number|null,\n"
    "  \"ventilation\": {\"mixing_height_m\": number|null, \"transport_wind_kmh\": number|null, \"rating\": \"poor\"|\"fair\"|\"good\"|null},\n"
    "  \"crossover\": {\"threshold_percent\": 30, \"expected_today\": true|false|null, \"method\": \"T>=RH and RH<=threshold\"},\n"
    "  \"mixing_down\": {\"aloft_winds_kmh\": number|null, \"could_mix\": true|false|null},\n"
    "  \"summary\": \"string\"\n"
    "}\n"
    "If the image is ambiguous, be conservative: populate nulls where uncertain and explain them in 'Summary'."
)

USER_PROMPT = (
    "Analyze this observed Skew‑T image per the checklist. "
    "Assume the image contains Skew‑T (log‑P) with temperature (red), dew point (green), and wind barbs/hodograph."
)


# ---------- Helper functions ----------

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json_block(text: str) -> Optional[Dict]:
    """
    Extract a JSON object from the model's text.
    Handles fenced blocks (```json ... ```) or raw { ... } content.
    Returns dict or None if parsing fails.
    """
    # Try fenced JSON first
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Fallback: find the first { ... } that looks like JSON
    brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace:
        candidate = brace.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def split_summary_and_json(full_text: str) -> Tuple[str, Optional[Dict]]:
    """
    Returns (summary_text, json_dict or None).
    If the model follows instructions, 'Summary' appears as a section and JSON is parsed.
    Otherwise, returns the whole text as summary with json=None.
    """
    data = extract_json_block(full_text)

    # Remove JSON block from summary for cleanliness
    summary = full_text
    if data is not None:
        # Remove fenced block if present
        summary = re.sub(r"```json\s*\{.*?\}\s*```", "", summary, flags=re.DOTALL)
        # Or raw brace block
        summary = re.sub(r"\{.*\}", "", summary, flags=re.DOTALL)

    # Trim extraneous whitespace
    summary = summary.strip()
    return summary, data


# ---------- Main API call ----------

def interpret_skewt_with_azure(
    image_path: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    deployment: Optional[str] = None,
    temperature: float = 0.2
) -> Tuple[str, Optional[Dict]]:
    """
    Reads a Skew‑T PNG and calls Azure OpenAI (vision) to get:
      - summary_text (str)
      - data_json (dict or None)

    If endpoint/key/deployment not provided, tries environment variables:
      AZURE_INFERENCE_ENDPOINT, AZURE_INFERENCE_KEY, AZURE_MODEL_DEPLOYMENT
    """
    load_dotenv()
    endpoint = endpoint or os.getenv("AZURE_INFERENCE_ENDPOINT")
    api_key  = api_key or os.getenv("AZURE_INFERENCE_KEY")
    deployment = deployment or os.getenv("AZURE_MODEL_DEPLOYMENT")

    if not endpoint or not api_key or not deployment:
        raise ValueError(
            "Missing endpoint/key/deployment. "
            "Set env vars AZURE_INFERENCE_ENDPOINT, AZURE_INFERENCE_KEY, AZURE_MODEL_DEPLOYMENT "
            "or pass them as parameters."
        )

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )

    b64 = encode_image_to_base64(image_path)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        UserMessage(content=[
            TextContentItem(text=USER_PROMPT),
            ImageContentItem(image_url=None, image_data=b64, mime_type="image/png")
        ])
    ]

    result = client.complete(
        model=deployment,
        messages=messages,
        temperature=temperature
    )

    # Collect text parts from the response
    text_parts = []
    for part in result.output_message.content:
        if hasattr(part, "text") and isinstance(part.text, str):
            text_parts.append(part.text)

    full_text = "\n".join(text_parts).strip()
    summary, data = split_summary_and_json(full_text)
    return summary, data


# ---------- Optional CLI for quick testing ----------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python skewt_interpret_azure.py /path/to/skewt.png")
        sys.exit(1)

    img = sys.argv[1]
    summary_text, data_json = interpret_skewt_with_azure(img)

    print("\n=== Summary ===\n")
    print(summary_text)

    print("\n=== Data (JSON) ===\n")
    print(json.dumps(data_json or {}, indent=2))
