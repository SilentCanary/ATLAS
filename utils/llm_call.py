import json
import boto3
from io import BytesIO
from typing import List, Dict

# ---------------------------
# AWS Bedrock setup
# ---------------------------
REGION = "us-east-1"
LLM_MODEL = "mistral.devstral-2-123b"  # Replace with your model ID

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
    # credentials read from env or ~/.aws/credentials
)


# ---------------------------
# Invoke Mistral via Bedrock
# ---------------------------
def ask_llm(prompt: str, max_tokens: int = 4096, temperature: float = 0.5) -> str:
    payload = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = bedrock_client.invoke_model(
        modelId=LLM_MODEL,
        body=BytesIO(json.dumps(payload).encode("utf-8")),
        contentType="application/json",
        accept="application/json"
    )

    raw_body = response["body"].read()
    raw_text = raw_body.decode()
    print("\n--- RAW RESPONSE ---\n", raw_text, "\n--- END RAW RESPONSE ---\n")

    try:
        result = json.loads(raw_text)
        # Check multiple structures
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        elif "content" in result:
            return result["content"][0].get("text", "")
        elif "completion" in result:
            return result["completion"]
        else:
            return str(result)
    except json.JSONDecodeError:
        return raw_text