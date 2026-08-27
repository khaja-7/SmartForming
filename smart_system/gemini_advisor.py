import os
import logging
from google import genai

logger = logging.getLogger("gemini_advisor")

FALLBACK_MODELS = [
    'gemini-2.5-flash',
    'gemini-3.5-flash-lite',
    'gemini-3.6-flash',
    'gemini-3-flash-preview',
]

def generate_crop_advice(input_data: dict, prediction: dict) -> str:
    """
    Generates agricultural advice using the Gemini AI API with multi-model fallback.
    
    Args:
        input_data (dict): The soil and weather parameters (N, P, K, temperature, humidity, ph, rainfall).
        prediction (dict): The ML prediction results including top_crop and alternatives.
        
    Returns:
        str: The generated advice text, or a fallback message if the API fails or is not configured.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "AI advice temporarily unavailable. Please configure the GEMINI_API_KEY in your .env file."

    prompt = f"""
    You are an agricultural expert AI.

    Soil and Climate Data:
    Nitrogen: {input_data.get('N')} kg/ha
    Phosphorus: {input_data.get('P')} kg/ha
    Potassium: {input_data.get('K')} kg/ha
    Temperature: {input_data.get('temperature') or input_data.get('Temperature')} °C
    Humidity: {input_data.get('humidity') or input_data.get('Humidity')} %
    pH: {input_data.get('ph') or input_data.get('pH')}
    Rainfall: {input_data.get('rainfall') or input_data.get('Rainfall')} mm

    Machine Learning Prediction:
    Recommended Crop: {prediction.get('top_crop')}
    Alternatives: {', '.join(prediction.get('alternatives', []))}

    Explain briefly:
    1. Why the recommended crop fits the soil/climate.
    2. Why the alternatives are viable.
    3. One key action to maximize yield.
    
    CRITICAL: Provide your answer as exactly 3 very short, concise bullet points. Do not include any introductory or concluding text. Maximum 3 sentences total.
    """

    client = genai.Client(api_key=api_key)

    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}. Trying fallback...")

    return "AI advice is temporarily experiencing high server demand. Please try again shortly."
