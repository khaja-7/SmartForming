import os
import logging

logger = logging.getLogger("gemini_advisor")

FALLBACK_MODELS = [
    'gemini-2.5-flash',
    'gemini-3.6-flash',
]

def generate_crop_advice(input_data: dict, prediction: dict) -> str:
    """
    Generates agricultural advice using the Gemini AI API with multi-model fallback
    and an instant local agronomic advice fallback if cloud LLM is unavailable.
    """
    top_crop = prediction.get('top_crop', 'Selected Crop')
    alts = prediction.get('alternatives', [])
    alt_str = ', '.join(alts) if alts else 'standard rotational crops'

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip() and not api_key.startswith("your_"):
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            prompt = f"""You are an agricultural expert AI.

Soil and Climate Data:
Nitrogen: {input_data.get('N')} kg/ha
Phosphorus: {input_data.get('P')} kg/ha
Potassium: {input_data.get('K')} kg/ha
Temperature: {input_data.get('temperature') or input_data.get('Temperature')} °C
Humidity: {input_data.get('humidity') or input_data.get('Humidity')} %
pH: {input_data.get('ph') or input_data.get('pH')}
Rainfall: {input_data.get('rainfall') or input_data.get('Rainfall')} mm

Machine Learning Prediction:
Recommended Crop: {top_crop}
Alternatives: {alt_str}

Explain briefly:
1. Why the recommended crop fits the soil/climate.
2. Why the alternatives are viable.
3. One key action to maximize yield.

CRITICAL: Provide your answer as exactly 3 short bullet points. No introductory or concluding chatter."""

            for model_name in FALLBACK_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if response and response.text and response.text.strip():
                        return response.text.strip()
                except Exception as e:
                    logger.warning(f"Gemini model {model_name} error: {e}")
                    # If permission denied / leaked key, stop trying more models to avoid latency
                    if "403" in str(e) or "PERMISSION_DENIED" in str(e) or "API_KEY_INVALID" in str(e):
                        break
        except Exception as e:
            logger.warning(f"Gemini client initialization failed: {e}")

    # ── Instant Local Agronomic Fallback (zero latency, 100% reliable) ──
    n = input_data.get('N', 0)
    temp = input_data.get('temperature') or input_data.get('Temperature', 25)
    rain = input_data.get('rainfall') or input_data.get('Rainfall', 100)

    return (
        f"• **Suitability**: {top_crop.capitalize()} matches the local climate ({temp}°C, {rain}mm rainfall) and current NPK nutrient profile.\n"
        f"• **Alternatives**: Viable secondary options include {alt_str} if market price or water availability changes.\n"
        f"• **Key Agronomic Action**: Maintain balanced soil fertility and timely irrigation during the critical flowering/grain-filling stages."
    )
