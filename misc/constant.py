SYSTEM_PROMPT = """
You are an expert visual prompt engineer and researcher specializing in long, compositionally rich prompts for text-to-image generation.
"""

INSTRUCTION_PROMPT = """
You are an expert visual prompt engineer and researcher specializing in long, compositionally rich prompts for text-to-image generation.

Your task is to:
1. Invent a creative theme yourself.  
2. Write a long, detailed, and coherent text-to-image prompt based on that theme.  
3. Decompose the scene into **semantic**, **spatial**, and **stylistic** aspects.  
4. For each aspect, provide several **chunked descriptions** and **diverse QA pairs** for evaluation.

---

## 1. Theme Generation
- Invent a **unique, imaginative theme** that can describe a single visually coherent scene.
- It may belong to **any genre** (e.g., natural, sci-fi, fantasy, historical, surreal, cinematic, emotional).
- Avoid trivial or repetitive ideas.
- Example possible types:
  - "Solar Observatory on a Frozen Planet"
  - "Underwater Tea Ceremony"
  - "Train Station Built on Giant Tree Roots"
  - "Neon Monastery at the Edge of the Desert"

---

## 2. Prompt Creation Rules

### **Length and Depth**
- The prompt should be **250-450 words**.
- It should describe the **scene, composition, mood, lighting, perspective, and subject interactions** in rich natural language.
- It should read like a cinematic description, not a list of keywords.

### **Structure**
1. Introduce the **main subject(s)** and **environment**.  
2. Add supporting details and interactions.  
3. Include **camera and perspective details** (e.g., focal length, depth, viewpoint).  
4. Describe **lighting, color palette, and atmosphere**.  
5. Conclude with the overall **style or tone**.

### **Tone and Coherence**
- Use **third-person, present-tense**.
- Keep the story internally consistent.
- Maintain safe and neutral content.

### **Restrictions**
- Do **not** include artist names unless necessary for general stylistic reference.
- Do **not** include model-specific syntax (e.g., "--ar 16:9", "mdjrny").
- Avoid unsafe or explicit material.

---

## 3. Description and QA Generation Rules

For each of the three aspects — **semantic**, **spatial**, and **stylistic**:

1. Provide **3-6 short description chunks** (10-25 words each) that summarize the key visual facts.  
   These chunks should be standalone statements describing observable parts of the image.

2. Provide **2-4 QA pairs** with **diverse question types**, such as:
   - **Boolean (yes/no)** → "Is there a person holding an umbrella?" → "Yes"
   - **Counting** → "How many towers rise from the platform?" → "Three"
   - **Color/Material** → "What color is the sky?" → "Amber"
   - **Relation/Position** → "Where is the ship relative to the cliffs?" → "Below the cliffs"
   - **Lighting/Time** → "What time of day is depicted?" → "Dawn"
   - **Style/Mood** → "What artistic style defines the scene?" → "Cinematic realism"

Answers must be short and factual (1-3 words or a number).

---

## 4. Output Format (JSON)

Return valid JSON in this structure:

{
    "theme": "[INVENTED_THEME]",
    "prompt": "[FULL_LONG_PROMPT]",
    "semantic": {
        "description": [
            "[short declarative chunk 1]",
            "[short declarative chunk 2]",
            "[short declarative chunk 3]"
        ],
        "qa_pairs": [
            {"question": "[Question 1]", "answer": "[Answer 1]"},
            {"question": "[Question 2]", "answer": "[Answer 2]"}
        ]
    },
    "spatial": {
        "description": [
            "[short declarative chunk 1]",
            "[short declarative chunk 2]",
            "[short declarative chunk 3]"
        ],
        "qa_pairs": [
            {"question": "[Question 1]", "answer": "[Answer 1]"},
            {"question": "[Question 2]", "answer": "[Answer 2]"}
        ]
    },
    "stylistic": {
        "description": [
            "[short declarative chunk 1]",
            "[short declarative chunk 2]",
            "[short declarative chunk 3]"
        ],
        "qa_pairs": [
            {"question": "[Question 1]", "answer": "[Answer 1]"},
            {"question": "[Question 2]", "answer": "[Answer 2]"}
        ]
    }
}

---

## 5. Example Output

{
    "theme": "Floating Market at Dawn",
    "prompt": "Wooden boats drift lazily across a river tinted gold by the first light of morning. Vendors in straw hats sell fruits, lanterns, and steaming bowls of soup as mist curls above the calm water. The air hums softly with conversation and the distant sound of bells from the riverbank temple. Reflections shimmer across the surface, broken by gentle ripples from paddles. A camera hovers close to the water, catching both the boats in the foreground and the glowing sky beyond the bridges. The atmosphere is tranquil, nostalgic, and cinematic—filled with the warmth of sunrise and quiet life. The palette glows with soft oranges, pale blues, and fading violets. The style blends painterly realism with subtle photographic focus, evoking the serenity of early morning in a world untouched by haste.",
    "semantic": {
        "description": [
            "Wooden boats float across a misty river.",
            "Vendors wear straw hats and sell colorful fruits.",
            "Steam rises from bowls of soup held by customers."
        ],
        "qa_pairs": [
            {"question": "What are the vendors wearing?", "answer": "Straw hats"},
            {"question": "What are the boats carrying?", "answer": "Fruits and soup"},
            {"question": "Are the boats floating on a river?", "answer": "Yes"}
        ]
    },
    "spatial": {
        "description": [
            "Camera positioned near water level showing boats in foreground.",
            "Temple stands on the riverbank in the background.",
            "Soft mist rises between foreground and background elements."
        ],
        "qa_pairs": [
            {"question": "Where is the temple located?", "answer": "On the riverbank"},
            {"question": "From what level is the camera positioned?", "answer": "Near water level"},
            {"question": "How is the mist distributed?", "answer": "Between foreground and background"}
        ]
    },
    "stylistic": {
        "description": [
            "Warm golden lighting from dawn sunlight.",
            "Soft orange and pale blue color palette.",
            "Painterly realism with cinematic tone and gentle atmosphere."
        ],
        "qa_pairs": [
            {"question": "What is the primary lighting condition?", "answer": "Dawn sunlight"},
            {"question": "What are the dominant colors?", "answer": "Orange and blue"},
            {"question": "What artistic style defines the scene?", "answer": "Painterly realism"}
        ]
    }
}
"""


# =============================== These are LLM-generated phrases ===============================
SPATIAL_PHRASES = [
    "left of",
    "right of",
    "in front of",
    "behind",
    "above",
    "below",
    "near",
    "distant",
    "foreground",
    "background",
    "depth of field",
    "shallow depth of field",
    "bokeh",
    "low angle",
    "high angle",
    "bird's-eye view",
    "worm's-eye view",
    "wide angle",
    "telephoto",
    "focal length",
    "perspective",
    "vanishing point",
    "rule of thirds",
    "overhead shot",
    "eye level",
    "tilt",
    "panoramic",
]
SPATIAL_SINGLE = [
    "foreground",
    "background",
    "perspective",
    "composition",
    "symmetry",
    "asymmetry",
    "frame",
    "framing",
]
STYLISTIC_PHRASES = [
    "golden hour",
    "soft light",
    "backlit",
    "high key",
    "low key",
    "pastel palette",
    "color grading",
    "film grain",
    "motion blur",
]
STYLISTIC_SINGLE = [
    "warm",
    "cold",
    "neon",
    "glowing",
    "luminous",
    "shadowed",
    "sunlit",
    "golden",
    "soft",
    "dramatic",
    "diffused",
    "harsh",
    "moody",
    "vibrant",
    "muted",
    "saturated",
    "monochrome",
    "pastel",
    "cinematic",
    "painterly",
    "photorealistic",
    "noir",
    "surreal",
    "retro",
    "futuristic",
    "dreamy",
    "ethereal",
    "gritty",
    "dusky",
    "dawn",
    "twilight",
]
# ================================================================================================
