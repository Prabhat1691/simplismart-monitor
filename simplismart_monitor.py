import argparse
import base64
import json
import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse

import requests
import schedule

# ----------------- LOGGING ----------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("simplismart_monitor.log"),
        logging.StreamHandler()
    ]
)


# ----------------- CONFIG LOADING ----------------- #

def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


# ----------------- RANDOM INPUT GENERATION ----------------- #

def random_chat_prompt() -> str:
    prompts = [
        "Explain what artificial intelligence is in one sentence.",
        "Tell me a random fun fact.",
        "Give me a one-line productivity tip.",
        "Describe a sunrise in a poetic way.",
        "Explain overfitting in machine learning in simple terms.",
        "What is an API? Explain like I'm 10.",
        "What is the difference between RAM and storage?",
        "Summarize the benefits of regular exercise in one sentence.",
        "Suggest a 3-day itinerary for a trip to Goa.",
        "How do I stay focused while studying?"
    ]
    return random.choice(prompts)


def generate_chat_payload(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    For type = 'openai_chat'
    Uses client.chat.completions.create-style payload.
    """
    return {
        "model": model_cfg.get("model_name"),
        "messages": [
            {
                "role": "system",
                "content": "You are a health-check bot. Reply concisely."
            },
            {
                "role": "user",
                "content": random_chat_prompt()
            }
        ],
        "max_tokens": 256,
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.9
    }


def generate_vision_payload(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    For type = 'openai_vision' (DeepSeek OCR).
    Generate a random text image using dummyimage.com so OCR is actually tested.
    """
    words = ["HELLO", "WORLD", "SIMPLISMART", "OCR", "CHECK", "MODEL", "DAILY", "HEALTH"]
    text = f"{random.choice(words)}-{random.randint(100, 999)}"
    encoded_text = urllib.parse.quote_plus(text)

    image_url = f"https://dummyimage.com/600x300/000/fff&text={encoded_text}"

    return {
        "model": model_cfg.get("model_name"),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "OCR the text in this image and respond with the extracted text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 256,
        "stream": False,
        "temperature": 0.0
    }


def generate_flux_kontext_payload(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    For type = 'flux_kontext'
    Uses random conditioning image from config (test_image_urls),
    falls back to single test_image_url or a random photo if needed.
    """
    urls = model_cfg.get("test_image_urls")
    image_url: Optional[str] = None

    if isinstance(urls, list) and urls:
        image_url = random.choice(urls)
    else:
        image_url = model_cfg.get(
            "test_image_url",
            "https://picsum.photos/600/600"
        )

    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
        img_b64 = base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to download test image for Flux Kontext: {e}")

    prompts = [
        "Display the logo in a minimalist style printed in white on a matte black ceramic coffee mug.",
        "Generate a cinematic portrait in a studio lighting setup.",
        "Create a futuristic city skyline at night with neon lights.",
        "Design a clean tech startup landing page hero illustration."
    ]
    prompt = random.choice(prompts)

    return {
        "image": img_b64,
        "prompt": prompt,
        "guidance_scale": 2.5,
        "num_inference_steps": 28,
        "num_images_per_prompt": 1,
        "height": 1024,
        "width": 1024,
        "threshold": 0.8
    }


def generate_flux_dev_payload(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    For type = 'flux_dev'
    Matches Flux dev payload (text → image) with random prompt + seed.
    """
    prompts = [
        "Futuristic city skyline at sunset, ultra detailed, cinematic lighting.",
        "Cozy reading corner with a warm lamp and a stack of books, realistic.",
        "Modern workspace with dual monitors and a laptop on a wooden desk.",
        "Hyperrealistic photograph of a mountain landscape at golden hour."
    ]
    prompt = random.choice(prompts)

    return {
        "prompt": prompt,
        "height": 1024,
        "width": 1024,
        "seed": random.randint(1, 2_000_000_000),
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
        "guidance_scale": 7,
        "negative_prompt": "blurry, distorted, low quality, low resolution"
    }


def generate_whisper_payload(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    For types = 'whisper_v2', 'whisper_v3'
    Uses random local audio files from test_audio_paths.
    """
    mtype = (model_cfg.get("type") or "").lower()

    # Local paths list
    paths = model_cfg.get("test_audio_paths")
    if not paths or not isinstance(paths, list):
        raise RuntimeError(
            "Whisper model requires 'test_audio_paths' in config.json (list of WAV files)."
        )

    # Choose a random audio file
    audio_path = random.choice(paths)

    # Load file
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read Whisper test audio file '{audio_path}': {e}")

    base_payload = {
        "audio_data": audio_b64,
        "language": "en",
        "task": "transcribe",
        "batch_size": 4,
        "length_penalty": 1,
        "patience": 1,
        "vad_onset": 0.5,
        "vad_offset": 0.363
    }

    # Whisper V2 specific params
    if mtype == "whisper_v2":
        params = {
            "beam_size": 5,
            "best_of": 5,
            "word_timestamps": 1,
            "diarization": 0,
            "streaming": 0,
            "vad_filter": 1
        }

    # Whisper V3 specific params
    else:  # whisper_v3
        params = {
            "word_timestamps": True,
            "diarization": False,
            "streaming": False,
            "vad_filter": True
        }

    payload = dict(base_payload)
    payload.update(params)
    return payload


def generate_default_payload(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback if model type is unknown.
    """
    return {
        "model": model_cfg.get("model_name"),
        "messages": [
            {
                "role": "user",
                "content": f"Health check for model {model_cfg.get('name')}."
            }
        ],
        "max_tokens": 64,
        "stream": False,
        "temperature": 0.5
    }


def generate_payload_for_model(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    mtype = (model_cfg.get("type") or "").lower()

    if mtype == "openai_chat":
        return generate_chat_payload(model_cfg)

    if mtype == "openai_vision":
        return generate_vision_payload(model_cfg)

    if mtype == "flux_kontext":
        return generate_flux_kontext_payload(model_cfg)

    if mtype == "flux_dev":
        return generate_flux_dev_payload(model_cfg)

    if mtype in ("whisper_v2", "whisper_v3"):
        return generate_whisper_payload(model_cfg)

    return generate_default_payload(model_cfg)


# ----------------- RESPONSE VALIDATION ----------------- #

def extract_text_from_non_stream_response(resp_json: Any) -> Optional[str]:
    """
    For non-streaming chat responses:
      choices[0].message.content or choices[0].delta.content or choices[0].text
    """
    if not isinstance(resp_json, Dict):
        return None

    choices = resp_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first = choices[0]
    if not isinstance(first, Dict):
        return None

    message = first.get("message")
    if isinstance(message, Dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    delta = first.get("delta")
    if isinstance(delta, Dict):
        content = delta.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return None


def validate_response(model_cfg: Dict[str, Any], resp_json: Any) -> Tuple[bool, str]:
    """
    Returns (is_valid, reason_if_invalid)
    - For openai_* models: require non-empty text content.
    - For flux_* and whisper_*: require non-empty JSON with expected shape.
    """
    if resp_json is None:
        return False, "Empty / non-JSON response."

    mtype = (model_cfg.get("type") or "").lower()

    if mtype in ("openai_chat", "openai_vision"):
        text = extract_text_from_non_stream_response(resp_json)
        if not text:
            return False, "No non-empty text content in response."
        return True, ""

    if mtype in ("flux_kontext", "flux_dev"):
        if not isinstance(resp_json, Dict) or not resp_json:
            return False, "Empty JSON object from Flux model."
        return True, ""

    if mtype in ("whisper_v2", "whisper_v3"):
        if not isinstance(resp_json, Dict):
            return False, "Unexpected Whisper response format."
        transcription = resp_json.get("transcription")
        if not transcription or not isinstance(transcription, list):
            return False, "No transcription field or empty transcription list."
        return True, ""

    # Default fallback
    if not isinstance(resp_json, Dict):
        return False, "Unexpected response format."
    return True, ""


# ----------------- MODEL MONITOR ----------------- #

class ModelMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.models: List[Dict[str, Any]] = [
            m for m in config.get("models", []) if m.get("enabled", True)
        ]

        test_settings = config.get("test_settings", {})
        self.timeout = int(test_settings.get("timeout", 60))
        self.max_retries = int(test_settings.get("max_retries", 2))
        self.retry_delay = int(test_settings.get("retry_delay", 5))
        self.include_response_times = bool(test_settings.get("include_response_times", True))
        self.slow_threshold_seconds = int(test_settings.get("slow_threshold_seconds", 30))

        self.slack_webhook_url = config.get("slack_webhook_url")

    def _call_model_once(
        self,
        model_cfg: Dict[str, Any],
        payload: Dict[str, Any]
    ) -> Tuple[Optional[requests.Response], float, Optional[str]]:
        """
        Single HTTP attempt. Returns (response_or_none, latency_seconds, error_message_if_any).
        """
        base_endpoint = (model_cfg.get("endpoint") or "").rstrip("/")
        mtype = (model_cfg.get("type") or "").lower()
        name = model_cfg.get("name")

        # Determine URL per model type
        if mtype in ("openai_chat", "openai_vision"):
            url = f"{base_endpoint}/chat/completions"
        else:
            # Flux & Whisper endpoints are full paths already
            url = base_endpoint

        headers: Dict[str, str] = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Only DeepSeek/Devstral style models use 'id' header
        id_header_value = model_cfg.get("id_header")
        if mtype in ("openai_chat", "openai_vision") and id_header_value:
            headers["id"] = str(id_header_value)

        start = time.time()
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            latency = time.time() - start
            return resp, latency, None
        except requests.exceptions.Timeout:
            latency = time.time() - start
            msg = f"Timeout after {self.timeout}s"
            logging.warning(f"[{name}] {msg}")
            return None, latency, msg
        except requests.RequestException as e:
            latency = time.time() - start
            msg = f"Request error: {e}"
            logging.warning(f"[{name}] {msg}")
            return None, latency, msg

    def test_single_model(self, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a dict with:
          name, type, endpoint, status, details, latency_ms, is_slow, attempts
        Status ∈ {UP, DOWN, REMOVED, TIMEOUT, ERROR}
        """
        name = model_cfg.get("name")
        mtype = model_cfg.get("type")
        endpoint = model_cfg.get("endpoint")

        logging.info("=" * 40)
        logging.info(f"Testing model: {name} | type={mtype} | endpoint={endpoint}")

        payload = generate_payload_for_model(model_cfg)

        attempts = self.max_retries + 1
        last_error: Optional[str] = None
        last_status: str = "ERROR"
        last_latency: Optional[float] = None

        for attempt in range(1, attempts + 1):
            resp, latency, err = self._call_model_once(model_cfg, payload)
            last_latency = latency

            # Network-level failure
            if resp is None:
                last_error = err or "Network error"
                last_status = "TIMEOUT" if (err and "Timeout" in err) else "ERROR"

                if attempt < attempts:
                    logging.info(f"[{name}] {last_status} on attempt {attempt}/{attempts}, retrying after {self.retry_delay}s.")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    break

            status_code = resp.status_code

            # 404 -> REMOVED, no retries
            if status_code == 404:
                last_status = "REMOVED"
                last_error = "HTTP 404 – model or endpoint not found."
                logging.warning(f"[{name}] Marked as REMOVED (404).")
                break

            # Parse JSON for all statuses to inspect errors
            try:
                resp_json = resp.json()
            except ValueError:
                resp_json = None

            # 2xx: validate content
            if 200 <= status_code < 300:
                is_valid, reason = validate_response(model_cfg, resp_json)
                if not is_valid:
                    last_status = "DOWN"
                    last_error = f"Invalid response content: {reason}"
                    logging.warning(f"[{name}] Invalid response: {reason}")

                    if attempt < attempts:
                        logging.info(f"[{name}] Retrying due to invalid response after {self.retry_delay}s.")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        break

                last_status = "UP"
                last_error = None
                break

            # Non-2xx: decide retryability
            body_snippet = (resp.text or "")[:300]
            last_error = f"HTTP {status_code}: {body_snippet}"

            if 500 <= status_code < 600:
                # Server error – retryable
                last_status = "DOWN"
                logging.warning(f"[{name}] Server error {status_code}: {body_snippet}")
                if attempt < attempts:
                    logging.info(f"[{name}] Retrying due to server error after {self.retry_delay}s.")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    break
            else:
                # 4xx/other – usually not retryable (auth, bad request, etc.)
                last_status = "DOWN"
                logging.warning(f"[{name}] Non-retryable error {status_code}: {body_snippet}")
                break

        latency_ms: Optional[int] = int(last_latency * 1000) if last_latency is not None else None
        is_slow: bool = latency_ms is not None and (latency_ms / 1000.0) > self.slow_threshold_seconds

        result = {
            "name": name,
            "type": mtype,
            "endpoint": endpoint,
            "status": last_status,
            "details": last_error,
            "latency_ms": latency_ms,
            "is_slow": is_slow,
            "attempts": attempts
        }

        if last_status == "UP":
            logging.info(f"[{name}] ✅ UP | latency={latency_ms}ms | slow={is_slow}")
        else:
            logging.warning(f"[{name}] ❌ {last_status} | details={last_error} | latency={latency_ms}ms")

        return result

    # ----------------- SLACK NOTIFICATION ----------------- #

    def send_slack_notification(self, results: List[Dict[str, Any]]) -> None:
        print("DEBUG SLACK URL =", self.slack_webhook_url)
        if not self.slack_webhook_url:
            logging.warning("Slack webhook URL not configured; skipping Slack notification.")
            return

        total = len(results)
        up = [r for r in results if r["status"] == "UP"]
        down = [r for r in results if r["status"] == "DOWN"]
        removed = [r for r in results if r["status"] == "REMOVED"]
        timeouts = [r for r in results if r["status"] == "TIMEOUT"]
        errors = [r for r in results if r["status"] == "ERROR"]
        slow = [r for r in results if r.get("is_slow")]

        if len(down) == 0 and len(timeouts) == 0 and len(errors) == 0:
            color = "good"
        elif len(down) + len(timeouts) + len(errors) <= 2:
            color = "warning"
        else:
            color = "danger"

        latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
        if latencies:
            avg_ms = sum(latencies) / len(latencies)
            max_ms = max(latencies)
            latency_summary = f"Avg: {avg_ms:.0f} ms | Max: {max_ms:.0f} ms"
        else:
            latency_summary = "No latency data"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        text_lines: List[str] = []
        text_lines.append(f"Test completed at {now_str}")
        text_lines.append("")
        text_lines.append(f"Total Models: *{total}*")
        text_lines.append(f"✅ UP: *{len(up)}*")
        text_lines.append(f"❌ DOWN: *{len(down)}*")
        text_lines.append(f"⏱ TIMEOUT: *{len(timeouts)}*")
        text_lines.append(f"🗑 REMOVED: *{len(removed)}*")
        text_lines.append(f"⚠ ERROR: *{len(errors)}*")
        text_lines.append("")
        text_lines.append(f"⏱ Response Times: {latency_summary}")
        if slow:
            text_lines.append(f"🐢 Slow models (> {self.slow_threshold_seconds}s): *{len(slow)}*")

        problematic = down + timeouts + removed + errors
        if problematic:
            text_lines.append("")
            text_lines.append("*Problematic models:*")
            for r in problematic:
                line = f"• *{r['name']}* – `{r['status']}`"
                if r.get("latency_ms") is not None:
                    line += f" – {r['latency_ms']} ms"
                if r.get("details"):
                    line += f" – _{r['details'][:150]}_"
                text_lines.append(line)

        message = {
            "attachments": [
                {
                    "color": color,
                    "title": "SimpliSmart Model Health Check",
                    "text": "\n".join(text_lines),
                    "footer": "SimpliSmart Monitoring"
                }
            ]
        }

        try:
            resp = requests.post(self.slack_webhook_url, json=message, timeout=10)
            if resp.status_code == 200:
                logging.info("Slack notification sent successfully.")
            else:
                logging.error(f"Failed to send Slack notification: HTTP {resp.status_code} {resp.text}")
        except Exception as e:
            logging.error(f"Error sending Slack notification: {e}")

    # ----------------- RUN ONE FULL TEST PASS ----------------- #

    def run_once(self) -> None:
        logging.info("=" * 70)
        logging.info("Starting SimpliSmart Model Health Check")
        logging.info("=" * 70)

        if not self.models:
            logging.warning("No models configured or enabled in config.json")
            return

        results: List[Dict[str, Any]] = []
        for model_cfg in self.models:
            try:
                res = self.test_single_model(model_cfg)
            except Exception as e:
                logging.error(f"Unexpected error while testing model {model_cfg.get('name')}: {e}")
                res = {
                    "name": model_cfg.get("name"),
                    "type": model_cfg.get("type"),
                    "endpoint": model_cfg.get("endpoint"),
                    "status": "ERROR",
                    "details": f"Exception: {e}",
                    "latency_ms": None,
                    "is_slow": False,
                    "attempts": 0
                }
            results.append(res)
            time.sleep(1)

        total = len(results)
        up = len([r for r in results if r["status"] == "UP"])
        down = len([r for r in results if r["status"] == "DOWN"])
        removed = len([r for r in results if r["status"] == "REMOVED"])
        timeouts = len([r for r in results if r["status"] == "TIMEOUT"])
        errors = len([r for r in results if r["status"] == "ERROR"])

        logging.info("-" * 70)
        logging.info(f"Summary: total={total} | UP={up} | DOWN={down} | TIMEOUT={timeouts} | REMOVED={removed} | ERROR={errors}")
        logging.info("-" * 70)

        self.send_slack_notification(results)


# ----------------- MAIN / ENTRYPOINT ----------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="SimpliSmart Model Health Monitor")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json"
    )
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run once now and exit (no scheduler)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    monitor = ModelMonitor(config)

    if args.now:
        monitor.run_once()
        return

    schedule_time = config.get("schedule_time", "09:00")
    logging.info("Running initial health check immediately.")
    monitor.run_once()

    logging.info(f"Scheduling daily health check at {schedule_time} (server local time).")
    schedule.every().day.at(schedule_time).do(monitor.run_once)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logging.info("Monitor stopped by user.")


if __name__ == "__main__":
    main()
