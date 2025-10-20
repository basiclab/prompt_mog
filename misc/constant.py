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

2. Provide **2-4 Yes or No questions** whose correct answer is always **"Yes."**  
   The questions must:
   - Be unambiguous and directly check facts stated or implied by the descriptions or the long prompt.  
   - Avoid negation and double negatives.  
   - Avoid ambiguous quantities. If a quantity is involved, phrase the question so that "Yes" is correct.  
   - Stay aspect aligned: semantic questions focus on objects and actions, spatial questions focus on positions and relations, stylistic questions focus on style, lighting, mood, or palette.

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
        "questions": [
            "[Yes-No question 1 with correct answer Yes]",
            "[Yes-No question 2 with correct answer Yes]"
        ]
    },
    "spatial": {
        "description": [
            "[short declarative chunk 1]",
            "[short declarative chunk 2]",
            "[short declarative chunk 3]"
        ],
        "questions": [
            "[Yes-No question 1 with correct answer Yes]",
            "[Yes-No question 2 with correct answer Yes]"
        ]
    },
    "stylistic": {
        "description": [
            "[short declarative chunk 1]",
            "[short declarative chunk 2]",
            "[short declarative chunk 3]"
        ],
        "questions": [
            "[Yes-No question 1 with correct answer Yes]",
            "[Yes-No question 2 with correct answer Yes]"
        ]
    }
}

---

## 5. Example Output

{
    "theme": "Floating Market at Dawn",
    "prompt": "Wooden boats drift across a river tinted gold by first light. Vendors in straw hats sell fruits and steaming soup while mist curls above the calm water. The air carries quiet conversation and distant bells from a riverbank temple. Reflections tremble across the surface, broken by paddles. A camera hovers close to the water, framing boats in the foreground and a glowing sky beyond the bridges. The atmosphere is tranquil, nostalgic, and cinematic, filled with the warmth of sunrise. The palette glows with soft oranges, pale blues, and fading violets. The style blends painterly realism with subtle photographic focus, evoking serene early morning.",
    "semantic": {
        "description": [
            "Wooden boats carry vendors selling fresh fruits and hot soup.",
            "Straw hats are worn by several vendors on the river.",
            "Soft conversation and temple bells create a calm scene."
        ],
        "questions": [
            "Are wooden boats present on the river?",
            "Are vendors wearing straw hats?",
            "Are foods or goods being sold from the boats?"
        ]
    },
    "spatial": {
        "description": [
            "The camera sits near water level with boats in the foreground.",
            "A temple stands on the riverbank in the distant background.",
            "Gentle mist separates foreground boats from background structures."
        ],
        "questions": [
            "Is the camera positioned close to the water surface?",
            "Is the temple located on the riverbank in the background?",
            "Is there visible mist between the foreground and background?"
        ]
    },
    "stylistic": {
        "description": [
            "Warm golden lighting indicates early morning sunlight.",
            "A palette of orange, pale blue, and violet dominates the scene.",
            "The overall tone follows painterly realism with a cinematic mood."
        ],
        "questions": [
            "Is the lighting consistent with dawn sunlight?",
            "Is a warm color palette present in the scene?",
            "Does the image convey a cinematic and tranquil mood?"
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
