import os
import re
import logging

logger = logging.getLogger("farm_ai_assistant")

FALLBACK_MODELS = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-2.0-flash',
]

def generate_farming_response(user_question: str) -> str:
    """
    Generates an agricultural advice response to a user's question using Gemini AI
    with fallback to an agronomic knowledge engine.
    """
    q_clean = (user_question or "").strip()
    if not q_clean:
        return "Hello! How can I assist with your crops, soil, or farm management today?"

    # Check for simple greeting / intro patterns first
    if re.match(r'^(hi+|hello+|hey+|namaste|greetings|hola|good\s*(morning|afternoon|evening))\b', q_clean, re.I):
        return (
            "Hello! I am your **Smart Agriculture AI Assistant** 🌱.\n\n"
            "I can help you with:\n"
            "• **Crop Selection & Suitability** (e.g., 'What crops grow best in sandy loam with pH 6.5?')\n"
            "• **Plant Disease & Pest Management** (e.g., 'How to treat early blight in tomatoes?')\n"
            "• **Fertilizer & Soil Nutrition** (e.g., 'NPK dosage for wheat during vegetative stage')\n"
            "• **Irrigation & Water Scheduling** (e.g., 'Best watering schedule for paddy')\n\n"
            "What would you like to explore today?"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip() and not api_key.startswith("your_"):
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            prompt = f"""You are an expert AI Agricultural Advisor assisting farmers, agronomists, and growers with precision farming.

User Question:
"{q_clean}"

Provide an expert, practical, and highly actionable response formatted with:
### 1. Diagnosis & Agronomic Assessment
(Clear, concise explanation of the question/issue)

### 2. Immediate Action Steps
(2-3 specific, actionable steps with exact dosage, organic/chemical recommendations, or practices)

### 3. Long-Term Preventative & Yield Optimization Measures
(1-2 strategic recommendations to safeguard soil health and future yields)

Tone: Knowledgeable, encouraging, practical, and precise."""

            for model_name in FALLBACK_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if response and response.text and response.text.strip():
                        return response.text.strip()
                except Exception as model_err:
                    logger.warning(f"Gemini model {model_name} failed: {model_err}")
        except Exception as e:
            logger.warning(f"Gemini API initialization failed: {e}")

    # ── Robust Built-In Agronomic Knowledge Fallback ──
    return _generate_knowledge_fallback(q_clean)


def _generate_knowledge_fallback(question: str) -> str:
    """Intelligent agronomic advice engine when cloud LLM is unavailable."""
    q = question.lower()

    if any(w in q for w in ['blight', 'fungus', 'fungal', 'spot', 'rot', 'mildew', 'rust', 'disease']):
        return (
            "### 1. Diagnosis & Agronomic Assessment\n"
            "Your query indicates a potential **fungal or bacterial foliar disease** (e.g., Blight, Leaf Spot, or Powdery Mildew). These pathogens thrive under high humidity, poor air circulation, and prolonged leaf wetness.\n\n"
            "### 2. Immediate Action Steps\n"
            "• **Foliar Fungicide Application**: Apply Copper Oxychloride (2.5g/L) or Mancozeb (2g/L) for fungal infections. For organic management, spray **Neem Oil (3-5 ml/L)** mixed with mild soapy water.\n"
            "• **Sanitation**: Prune and safely dispose of heavily infected leaves to stop spore transmission.\n"
            "• **Watering Technique**: Switch to drip or base irrigation; avoid overhead sprinklers that wet foliage.\n\n"
            "### 3. Long-Term Preventative Measures\n"
            "• Implement 2-to-3 season crop rotation with non-host crops.\n"
            "• Maintain optimal plant spacing to maximize sunlight and airflow across the canopy."
        )

    if any(w in q for w in ['fertilizer', 'npk', 'urea', 'nutrient', 'nitrogen', 'phosphorus', 'potassium', 'manure']):
        return (
            "### 1. Diagnosis & Agronomic Assessment\n"
            "Balanced soil nutrition is fundamental for robust vegetative growth, root development, and maximum grain/fruit yield. Nitrogen (N) drives canopy growth, Phosphorus (P) fuels root and flower formation, and Potassium (K) enhances disease resistance and grain filling.\n\n"
            "### 2. Immediate Action Steps\n"
            "• **Basal Dose**: Apply balanced NPK (e.g., 10:26:26 or 12:32:16) at sowing or transplanting.\n"
            "• **Split Nitrogen Application**: Apply Urea in 2-3 split doses (at tillering/vegetative and panicle initiation) rather than all at once to prevent leaching.\n"
            "• **Micronutrient Boost**: If leaves show yellowing between veins (chlorosis), apply Zinc Sulphate (0.5%) or Ferrous Sulphate as a foliar spray.\n\n"
            "### 3. Long-Term Preventative Measures\n"
            "• Conduct an annual soil test before each planting season.\n"
            "• Incorporate 5–10 tons/ha of well-decomposed Farmyard Manure (FYM) or vermicompost to boost soil organic carbon."
        )

    if any(w in q for w in ['pest', 'insect', 'caterpillar', 'aphid', 'borer', 'whitefly', 'worm']):
        return (
            "### 1. Diagnosis & Agronomic Assessment\n"
            "Insect pests cause yield reduction by sap-sucking (aphids, whiteflies) or tissue boring (stem borers, fruit worms). Early detection during scouting is essential.\n\n"
            "### 2. Immediate Action Steps\n"
            "• **Organic Control**: Spray **Neem Seed Kernel Extract (5%)** or cold-pressed Neem Oil (5 ml/L) every 7–10 days.\n"
            "• **Targeted Chemical Control**: For severe sucking pest infestation, apply Imidacloprid (0.5 ml/L) or Acetamiprid. For caterpillars/borers, use Emamectin Benzoate (0.4g/L).\n"
            "• **Physical Traps**: Install yellow and blue sticky traps (10-15 per acre) and pheromone traps to monitor and reduce adult populations.\n\n"
            "### 3. Long-Term Preventative Measures\n"
            "• Encourage natural predators (ladybird beetles, parasitic wasps).\n"
            "• Plant border trap crops such as marigold or castor along field edges."
        )

    if any(w in q for w in ['water', 'irrigation', 'drought', 'dry', 'rain']):
        return (
            "### 1. Diagnosis & Agronomic Assessment\n"
            "Water scheduling must match critical crop growth stages (germination, flowering, and grain/fruit filling) to prevent moisture stress and root hypoxia.\n\n"
            "### 2. Immediate Action Steps\n"
            "• **Critical Stage Irrigation**: Ensure soil moisture is adequate during flowering and pod/grain setting.\n"
            "• **Mulching**: Apply organic straw mulch (5–7 cm thick) around the root zone to conserve up to 30% soil moisture.\n"
            "• **Irrigation Timing**: Irrigate early in the morning or late in the evening to minimize evaporation losses.\n\n"
            "### 3. Long-Term Preventative Measures\n"
            "• Invest in drip irrigation or sprinkler systems for up to 50% water savings and fertigation efficiency.\n"
            "• Practice contour bunding and rainwater harvesting for dry season resilience."
        )

    # General precision farming guidance
    return (
        f"### 1. Agronomic Assessment for: '{question}'\n"
        "Successful crop production requires harmonizing soil conditions, climate factors, and timely agronomic interventions.\n\n"
        "### 2. Recommended Action Plan\n"
        "• **Soil Testing**: Verify pH, organic carbon, and available NPK levels to tailor input applications.\n"
        "• **Certified Seeds**: Always use disease-free, high-germination certified seed varieties adapted to your agro-climatic zone.\n"
        "• **Integrated Crop Management**: Balance chemical inputs with bio-fertilizers (Azotobacter, PSB) and organic soil conditioners.\n\n"
        "### 3. Best Practices\n"
        "• Keep regular scouting records to catch pest/disease pressure early.\n"
        "• Check our **Crop Recommendation** and **AI Plant Doctor** tools in the navigation menu for localized, image-based precision diagnosis."
    )
