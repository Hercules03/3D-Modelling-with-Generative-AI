import os

SCAD_KNOWLEDGE_DIR = "scad_knowledge_base"
CHROMA_PERSIST_DIR = os.path.join(SCAD_KNOWLEDGE_DIR, "chroma")

# Basic knowledge for categorization
BASIC_KNOWLEDGE = {
    "categories": [
        "Furniture",
        "Storage",
        "Decoration",
        "Utility",
        "Tableware",
        "Lighting",
        "Accessories",
        "Tools",
        "Display",
        "Organization"
    ],
    "properties": {
        "style": [
            "Modern",
            "Traditional",
            "Industrial",
            "Minimalist",
            "Art Deco",
            "Victorian",
            "Steampunk",
            "Oriental",
            "Scandinavian",
            "Medieval"
        ],
        "complexity": [
            "SIMPLE",
            "MEDIUM",
            "COMPLEX"
        ],
        "use_case": [
            "Functional",
            "Decorative",
            "Storage",
            "Display",
            "Utility",
            "Entertainment",
            "Organization"
        ],
        "geometric_properties": [
            "Symmetrical",
            "Asymmetrical",
            "Angular",
            "Curved",
            "Organic",
            "Regular",
            "Irregular",
            "Modular",
            "Nested",
            "Layered"
        ]
    }
}