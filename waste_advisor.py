"""
EcoSort AI - Smart Waste Segregation Advisor
=============================================
AI + Sustainability Internship Project
SDG 12: Responsible Consumption and Production (secondary: SDG 11)

WHAT THIS PROGRAM DOES
-----------------------
A user types the name of a waste item (e.g. "banana peel", "battery", "old shoes").
The program tells them:
    - which category it belongs to (Biodegradable, Recyclable, Hazardous, E-Waste, etc.)
    - how to dispose of it correctly
    - a short environmental tip

HOW AI IS USED (see also prompt_templates.md)
----------------------------------------------
1. Exact match  -> looks up the item directly in the knowledge base (fast, certain).
2. Fuzzy match  -> if there's no exact entry, it uses similarity matching
                   (a lightweight stand-in for semantic/embedding search) to find
                   the closest known item, and is transparent about the fact that
                   this is an approximate ("best guess") answer.
3. AI escalation -> if nothing is close enough, the program builds and displays the
                   exact prompt that would be sent to an LLM (IBM Granite via
                   watsonx.ai) to classify the never-before-seen item. This keeps the
                   full workflow demonstrable even without live internet/API access.

This design intentionally favours a simple, explainable, low-cost pipeline over a
complex one -- escalating to a full LLM call only when local logic can't help,
which is both cheaper and more transparent (Responsible AI principle).
"""

import json
import difflib
from pathlib import Path

BASE_DIR = Path(__file__).parent
KB_PATH = BASE_DIR / "knowledge_base.json"

FUZZY_MATCH_THRESHOLD = 0.6  # similarity cutoff (0-1) for an "approximate" match


def load_knowledge_base(path: Path = KB_PATH) -> dict:
    """Load the local waste knowledge base from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_llm_prompt(item_name: str) -> str:
    """
    Build the prompt that WOULD be sent to an LLM (e.g. IBM Granite via watsonx.ai)
    for an item not covered by the local knowledge base.
    See prompt_templates.md, Section 1, for the design rationale.
    """
    return (
        "SYSTEM:\n"
        "You are a waste-segregation assistant for a household/community recycling app.\n"
        "Classify the given waste item into exactly one category:\n"
        "[Biodegradable / Wet Waste, Recyclable / Dry Waste, Hazardous Waste, E-Waste,\n"
        "Non-Recyclable Waste, Sanitary Waste].\n"
        "Then give a one-line disposal instruction and one short environmental tip.\n"
        'Respond ONLY in this JSON shape:\n'
        '{"category": "...", "disposal_method": "...", "eco_tip": "..."}\n'
        "Be accurate, avoid assumptions about the user, and never suggest unsafe disposal\n"
        "(e.g., burning, flushing hazardous items, mixing categories).\n\n"
        f'USER:\nItem: "{item_name}"\n'
    )


def classify_waste(item_name: str, kb: dict) -> dict:
    """
    Classify a waste item using the 3-step logic described above.
    Returns a dict with keys: item, match_type, confidence, result (or prompt).
    """
    normalized = item_name.strip().lower()

    # Step 1: exact match
    if normalized in kb:
        return {
            "item": item_name,
            "match_type": "exact",
            "confidence": "high",
            "result": kb[normalized],
        }

    # Step 2: fuzzy match against known item names
    known_items = list(kb.keys())
    close_matches = difflib.get_close_matches(
        normalized, known_items, n=1, cutoff=FUZZY_MATCH_THRESHOLD
    )
    if close_matches:
        matched_item = close_matches[0]
        return {
            "item": item_name,
            "match_type": "fuzzy",
            "matched_to": matched_item,
            "confidence": "approximate",
            "result": kb[matched_item],
        }

    # Step 3: no match found -> escalate to LLM (shown here as a generated prompt,
    # since this offline demo has no live API access)
    return {
        "item": item_name,
        "match_type": "none",
        "confidence": "unknown",
        "llm_prompt": build_llm_prompt(item_name),
    }


def print_result(response: dict) -> None:
    """Pretty-print a classification result to the console."""
    print("\n" + "=" * 55)
    print(f"Item: {response['item']}")

    if response["match_type"] == "exact":
        r = response["result"]
        print("Match: exact match in knowledge base (high confidence)")
        print(f"Category        : {r['category']}")
        print(f"Disposal method : {r['disposal_method']}")
        print(f"Eco tip         : {r['eco_tip']}")

    elif response["match_type"] == "fuzzy":
        r = response["result"]
        print(f"Match: closest known item is '{response['matched_to']}' "
              f"(approximate match - please verify)")
        print(f"Category        : {r['category']}")
        print(f"Disposal method : {r['disposal_method']}")
        print(f"Eco tip         : {r['eco_tip']}")

    else:
        print("Match: none found locally. Escalating to AI model (IBM Granite).")
        print("The following prompt would be sent to the LLM:\n")
        print(response["llm_prompt"])
        print("[Offline demo mode: no live API call made. See prompt_templates.md]")

    print("=" * 55)


def responsible_ai_notice() -> None:
    print(
        "\nResponsible AI note: This tool gives general guidance only. Disposal rules\n"
        "vary by city/municipality - always confirm hazardous, biomedical, or e-waste\n"
        "handling with your local civic body. No personal data is collected or stored.\n"
    )


def run_cli() -> None:
    """Simple command-line demo loop."""
    kb = load_knowledge_base()
    print("=" * 55)
    print(" EcoSort AI - Smart Waste Segregation Advisor")
    print(" SDG 12: Responsible Consumption and Production")
    print("=" * 55)
    responsible_ai_notice()
    print("Type a waste item name (e.g. 'banana peel', 'battery').")
    print("Type 'quit' to exit.\n")

    while True:
        item = input("Enter waste item > ").strip()
        if item.lower() in ("quit", "exit"):
            print("Thank you for using EcoSort AI. Goodbye!")
            break
        if not item:
            continue
        response = classify_waste(item, kb)
        print_result(response)


if __name__ == "__main__":
    run_cli()
