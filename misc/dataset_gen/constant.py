# These categories are extracted from the "28 types of photography styles" from Adobe
#   https://www.adobe.com/creativecloud/photography/type.html
PHOTOGRAPHY_CATEGORIES = [
    # Natural World (7 categories)
    (
        "Landscape Photography",
        "nature",
        "Wide scenic views of natural environments, mountains, valleys, and horizons",
    ),
    (
        "Seascape Photography",
        "nature",
        "Coastal scenes, ocean waves, beaches, and marine environments",
    ),
    (
        "Wildlife Photography",
        "nature",
        "Animals in their natural habitats, behavior and action shots",
    ),
    (
        "Macro Photography",
        "nature",
        "Extreme close-ups revealing intricate details of small subjects",
    ),
    (
        "Astrophotography",
        "nature",
        "Night sky, stars, celestial bodies, and astronomical phenomena",
    ),
    (
        "Weather Photography",
        "nature",
        "Dramatic atmospheric conditions, storms, lightning, and cloud formations",
    ),
    (
        "Botanical Photography",
        "nature",
        "Plants, flowers, trees, and vegetation in detailed compositions",
    ),
    # Built Environment (4 categories)
    (
        "Architecture Photography",
        "built_environment",
        "Buildings, structures, and architectural design elements",
    ),
    (
        "Urban Photography",
        "built_environment",
        "City scenes, streets, infrastructure, and metropolitan environments",
    ),
    (
        "Real Estate Photography",
        "built_environment",
        "Interior and exterior property documentation for commercial purposes",
    ),
    (
        "Aerial Photography",
        "built_environment",
        "Elevated bird's-eye perspectives from drones or aircraft",
    ),
    # People-Focused (5 categories)
    (
        "Portrait Photography",
        "people",
        "Individual or group subjects emphasizing facial features and expressions",
    ),
    (
        "Fashion Photography",
        "people",
        "Stylized images showcasing clothing, accessories, and models",
    ),
    (
        "Sports Photography",
        "people",
        "Athletic activities, dynamic motion, and competitive moments",
    ),
    (
        "Candid Photography",
        "people",
        "Unposed, spontaneous moments capturing authentic human behavior",
    ),
    (
        "Event Photography",
        "people",
        "Weddings, celebrations, ceremonies, and special occasions",
    ),
    # Commercial & Still Life (4 categories)
    (
        "Food Photography",
        "still_life",
        "Culinary subjects, plated dishes, and appetizing presentations",
    ),
    (
        "Product Photography",
        "still_life",
        "Commercial items, e-commerce shots, and merchandise documentation",
    ),
    (
        "Still Life Photography",
        "still_life",
        "Arranged inanimate objects in artistic compositions",
    ),
    (
        "Automotive Photography",
        "still_life",
        "Vehicles, cars, motorcycles showcasing design and details",
    ),
    # Artistic Expression (5 categories)
    (
        "Fine Art Photography",
        "artistic",
        "Creative and conceptual imagery emphasizing artistic vision",
    ),
    (
        "Black and White Photography",
        "artistic",
        "Monochromatic images focusing on contrast, tone, and form",
    ),
    (
        "Abstract Photography",
        "artistic",
        "Non-representational images emphasizing shapes, patterns, and colors",
    ),
    (
        "Surreal Photography",
        "artistic",
        "Dreamlike, fantastical, and imaginative compositions",
    ),
    (
        "Long Exposure Photography",
        "artistic",
        "Motion blur effects, light trails, and time-based imagery",
    ),
]
DATA_GEN_SYSTEM_PROMPT = r"""You are an expert visual prompt engineer and researcher specializing in generating long, compositionally rich prompts for text-to-image tasks."""
DATA_GEN_INSTRUCTION_PROMPT = r"""Your task is to:
1. Write a long, detailed, and coherent text-to-image prompt based on the given theme
2. Ensure the prompt is not similar to any of the existing prompts
3. Decompose the scene into **semantic**, **spatial**, and **stylistic** aspects  
4. For each aspect, provide **chunked descriptions** and **diverse QA pairs** for evaluation

---

## 1. Theme

Theme: {theme}  
Description: {description}
Existing prompts: {existing_prompts}

---

## 2. Prompt Creation Rules

### Overall Structure & Length
- Target length: 250-450 words
- Format: Coherent narrative paragraph(s), NOT keyword lists
- Tone: Third-person, present-tense, cinematic description
- Coherence: Maintain internal consistency throughout

### Required Components

Your prompt MUST include these six elements in natural language:

#### 1. Subject Terms
- What: Main subject(s) and their key characteristics
- Where: Environment and setting details
- Who/What interactions: Subject relationships and activities
- Examples: "A weathered lighthouse", "Two children playing", "An ancient oak tree"

#### 2. Composition & Spatial Arrangement
- Camera perspective: Viewpoint, angle, distance
- Framing: How elements are arranged in the scene
- Depth: Foreground, midground, background elements
- Scale relationships: Relative sizes and positions
- Examples: "viewed from a low angle", "centered in the frame", "receding into the distance"

#### 3. Lighting & Atmosphere
- Light source: Natural, artificial, time of day
- Light quality: Soft, harsh, diffused, dramatic
- Atmospheric conditions: Weather, fog, clarity, haze
- Shadows and highlights: How light interacts with subjects
- Examples: "bathed in golden hour sunlight", "illuminated by a single spotlight", "shrouded in morning mist"

#### 4. Color Palette & Mood
- Dominant colors: Primary color scheme
- Color relationships: Complementary, analogous, monochromatic
- Color temperature: Warm, cool, neutral
- Emotional tone: Mood conveyed through color
- Examples: "rich burgundy and gold tones", "cool blue-gray palette", "vibrant sunset oranges"

#### 5. Style Modifiers
- Artistic style: Photography genre, art movement, or aesthetic approach
- Medium characteristics: What the image should feel like
- Technical approach: Rendering style or photographic technique
- Examples: "in the style of fine art portrait photography", "documentary photojournalism aesthetic", "cinematic widescreen composition"

#### 6. Quality Boosters
- Technical excellence markers: Resolution, sharpness, detail level
- Professional indicators: Skill level, equipment quality
- Recognition markers: Award-winning, professional, masterful
- Examples: "highly detailed", "professional photography", "sharp focus", "8K resolution", "award-winning composition"

### Writing Guidelines

**DO:**
- Write in flowing, descriptive prose that tells a visual story
- Layer details progressively from general to specific
- Use sensory and evocative language
- Integrate all six components naturally into the narrative
- Maintain consistent perspective and tense
- Include specific, concrete details over vague descriptions

**DON'T:**
- Use comma-separated keyword lists (e.g., "sunset, beach, 4k, trending")
- Include model-specific syntax (e.g., "--ar 16:9", "mdjrny-v4")
- Reference specific artist names unless contextually necessary
- Use negation or negative constructions
- Include unsafe, explicit, or harmful content
- Repeat the same descriptive patterns

### **Example Structure**

```
[SUBJECT INTRODUCTION - 2-3 sentences]
Introduce main subject(s) and immediate environment with key characteristics.

[SPATIAL & COMPOSITIONAL DETAILS - 2-3 sentences]
Describe camera perspective, framing, and how elements are arranged in 3D space.

[LIGHTING & ATMOSPHERE - 2-3 sentences]
Detail the lighting conditions, atmospheric effects, and how light interacts with the scene.

[COLOR & MOOD - 1-2 sentences]
Establish color palette and emotional tone.

[SUPPORTING DETAILS - 2-3 sentences]
Add textures, materials, smaller elements, and interactions that enrich the scene.

[STYLE & QUALITY CONCLUSION - 1-2 sentences]
Conclude with overall style approach and technical quality markers.
```

---

## 3. Description and QA Generation Rules

For each aspect — **semantic**, **spatial**, and **stylistic**:

### Description Chunks
- Provide **2-4 short declarative statements** (10-25 words each)
- Each chunk describes ONE observable visual fact
- Chunks should be standalone and unambiguous
- Use present tense, third-person
- Avoid interpretations; state observable facts

**Semantic descriptions**: Focus on WHAT (objects, subjects, actions, identities)  
**Spatial descriptions**: Focus on WHERE (positions, arrangements, relationships, depth)  
**Stylistic descriptions**: Focus on HOW (lighting, color, mood, artistic style, technique)

### Yes/No Questions
- Provide **2-4 questions** per aspect
- **All correct answers must be "Yes"**
- Questions must be:
  - Unambiguous: Only one clear interpretation
  - Verifiable: Directly checkable from the image
  - Aspect-aligned: Match the category (semantic/spatial/stylistic)
  - Affirmative: No negation or double negatives
  - Specific: Avoid ambiguous quantities or vague terms

Question Formulation Tips:
- Start with: "Is there...", "Does the...", "Are the...", "Is the..."
- Avoid: "Is there no...", "Isn't there...", "Are there more than X..."
- If quantity matters, phrase so "Yes" is correct (e.g., "Are there multiple trees?" instead of "Is there only one tree?")

---

## 4. Output Format (JSON)

Return **valid, parseable JSON** in this exact structure:

{{
    "theme": "[CATEGORY_NAME]",
    "prompt": "[FULL 250-450 WORD LONG-FORM PROMPT FOLLOWING ALL RULES ABOVE]",
    "semantic": {{
        "description": [
            "[Observable fact about objects/subjects - 10-25 words]",
            ...
        ],
        "questions": [
            "[Yes-No question about objects/subjects with answer Yes]",
            ...
        ]
    }},
    "spatial": {{
        "description": [
            "[Observable fact about positions/arrangements - 10-25 words]",
            ...
        ],
        "questions": [
            "[Yes-No question about spatial relationships with answer Yes]",
            ...
        ]
    }},
    "stylistic": {{
        "description": [
            "[Observable fact about style/lighting/mood - 10-25 words]",
            ...
        ],
        "questions": [
            "[Yes-No question about style/atmosphere with answer Yes]",
            ...
        ]
    }}
}}

---

## Example Output

**Input:**
```
Theme: Landscape Photography
Description: Wide scenic views of natural environments, mountains, valleys, and horizons
```

**Output Example (Abbreviated):**

{{
    "theme": "Landscape Photography",
    "prompt": "A majestic mountain valley unfolds beneath a dramatic sky, where towering granite peaks rise sharply from a dense evergreen forest. The scene is captured from an elevated vantage point, positioned roughly one-third into the frame, creating a sense of depth and scale that draws the eye from the flower-dotted alpine meadow in the foreground through the winding river in the middle distance, and finally to the snow-capped summits that dominate the background. Morning light breaks through scattered clouds, casting long golden rays that illuminate the eastern faces of the mountains while leaving the western slopes in cool blue shadow. The interplay of light creates a painterly quality across the landscape, highlighting the textured rock faces and creating pockets of luminosity in the morning mist that still clings to the valley floor. The color palette balances warm golds and greens in the sunlit areas with cool blues and purples in the shadows, evoking a sense of pristine wilderness and natural grandeur. Wildflowers dot the foreground meadow in splashes of purple, yellow, and white, adding intimate detail to contrast with the monumental scale of the peaks beyond...",
    "semantic": {{
    "description": [
            "The scene features towering granite mountain peaks with snow-capped summits.",
            "A winding river flows through the valley between dense evergreen forests.",
            "Wildflowers in purple, yellow, and white colors dot the alpine meadow."
        ],
        "questions": [
            "Does the scene include mountain peaks?",
            "Is there a river visible in the valley?"
        ]
    }},
    ...
}}"""

REWRITER_SYSTEM_RPOMPT = """You are an expert prompt rephraser for text-to-image generation models."""
REWRITER_INSTRUCTION_PROMPT = """Your task is to rephrase the given prompt while keeping the same subject and intent. You should keep the length of the rephrased prompt as close to the original as possible. Please provide {num_variants} variants.
Original prompt: {original_prompt}
Output Format (Python list of strings): [prompt1, prompt2, ...]"""


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
