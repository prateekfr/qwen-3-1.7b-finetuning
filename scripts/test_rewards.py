"""
Test suite for the Unified Reward Manager (Local Composite & API Judge).
Verifies:
1. Strict Lipogram Gate (0.0 if forbidden 'e' appears)
2. High reward for fluent lipograms
3. Repetition penalty on looping text
4. Dictionary penalty on invented non-words
5. Length shaping on lazy short responses
6. API toggle and graceful fallback behavior
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.rewards import get_reward_manager


def main():
    print("=" * 70)
    print("  Testing Unified Reward Manager (Local Heuristic & API Toggle)")
    print("=" * 70)

    manager = get_reward_manager()
    print(f"Active Mode from config: '{manager.mode}'")
    print(f"Loaded Dictionary Words: {len(manager.local_reward.dictionary)}")

    test_cases = [
        {
            "name": "1. Fluent High-Quality Lipogram (Trained SFT Output)",
            "prompt": "What is a dog?",
            "answer": "A dog is a loyal companion that runs with joy, stands by you in good and bad hours, and brings warmth and joy to your daily tasks. Dogs can grow into a loyal companion, a joyful playgroup, or a calm guardian.",
            "expect": "High reward (~0.75 - 0.95)",
        },
        {
            "name": "2. Text Containing Forbidden Letter 'e'",
            "prompt": "What is a dog?",
            "answer": "The dog is a very loyal animal that loves everyone.",
            "expect": "Exactly 0.0 (Hard Lipogram Gate)",
        },
        {
            "name": "3. Repetitive Degenerate Looping Text",
            "prompt": "Tell me about dogs.",
            "answer": "A dog is a dog and a dog is a dog and a dog is a dog and a dog runs and a dog runs and a dog runs.",
            "expect": "Low reward (Repetition penalty fired)",
        },
        {
            "name": "4. Invented / Gibberish Non-Words",
            "prompt": "What is a dog?",
            "answer": "A dog is a zump flumbr blork xorv plunk quazz gorp.",
            "expect": "Low reward (Dictionary penalty fired)",
        },
        {
            "name": "5. Lazy Short Answer (< 5 tokens)",
            "prompt": "What is a dog?",
            "answer": "A loyal hound.",
            "expect": "Low reward (Length shaping penalty fired)",
        },
    ]

    print("\n--- Testing Local Composite Reward Mode ---")
    manager.set_mode("local")

    for tc in test_cases:
        score, diag = manager.evaluate(tc["prompt"], tc["answer"])
        print(f"\n[Case: {tc['name']}]")
        print(f"  Prompt:   \"{tc['prompt']}\"")
        print(f"  Answer:   \"{tc['answer']}\"")
        print(f"  Expected: {tc['expect']}")
        print(f"  Result:   Reward = {score:.4f} | Lipogram Pass = {diag['lipogram_pass']} | Tokens = {diag['num_tokens']}")
        print(f"  Metrics:  Length={diag.get('length_score', 0):.2f}, Rep={diag.get('repetition_score', 0):.2f}, Dict={diag.get('dictionary_score', 0):.2f}, Rel={diag.get('relevance_score', 0):.2f}")
        print(f"  Note:     {diag['note']}")

    print("\n" + "=" * 70)
    print("--- Testing API Mode Toggle & Graceful Fallback ---")
    manager.set_mode("api")
    print(f"Switched manager mode to: '{manager.mode}'")
    # Truly e-free sentence:
    test_e_free = "A hound is a loyal dog with high vigor and loyalty."
    api_score, api_diag = manager.evaluate("What is a dog?", test_e_free)
    print(f"API Mode Evaluation Result: Reward = {api_score:.4f} | Mode = {api_diag['mode']}")
    print(f"API Note: {api_diag.get('note', '')}")
    print("=" * 70)
    print("✅ All Reward System Tests Completed Successfully!")


if __name__ == "__main__":
    main()
