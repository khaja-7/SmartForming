import os
import logging
from google import genai

logger = logging.getLogger("farm_ai_assistant")

FALLBACK_MODELS = [
    'gemini-2.5-flash',
    'gemini-3.5-flash-lite',
    'gemini-3.6-flash',
    'gemini-3-flash-preview',
]

def generate_farming_response(user_question: str) -> str:
    """
    Generates an agricultural advice response to a user's question using Gemini AI
    with multi-model fallback for maximum reliability.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "AI assistant is currently unavailable. Please check that GEMINI_API_KEY is configured in your .env file."

    prompt = f"""
    You are an expert AI Agricultural Advisor assisting farmers, agronomists, and growers with precision farming.

    User Question / Input:
    "{user_question}"

    INSTRUCTIONS:
    1. If the input is a GREETING, INTRODUCTORY, or GENERAL QUESTION (e.g., "hi", "hello", "who are you", "what can you do", "help"):
       - Respond warmly, naturally, and concisely in 2-3 sentences.
       - Introduce yourself as the Smart Agriculture AI Advisor and invite them to ask about crop recommendations, disease symptoms, pest control, soil management, or harvest forecasting.
       - DO NOT use the 3-section structured diagnosis format for simple greetings.

    2. If the input is a SPECIFIC FARMING / AGRONOMIC QUESTION (e.g., diseases, pests, crops, fertilizer, irrigation, soil health, yield):
       - Provide an expert, highly practical, and actionable answer strictly formatted as:
         ### 1. Diagnosis / Assessment
         (Clear analysis of the specific crop/pest/soil issue)
         ### 2. Immediate Action Steps
         (2-3 actionable steps with exact organic/chemical product names, irrigation adjustments, or methods)
         ### 3. Preventative Measures
         (1-2 long-term preventative agronomic practices to safeguard future yields)

    Tone: Knowledgeable, encouraging, authoritative, and practical.
    """

    last_error = None
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
            last_error = e

    return f"AI assistant is temporarily experiencing high server demand. Please try again in a few moments."
