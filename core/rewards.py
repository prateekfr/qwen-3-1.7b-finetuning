"""
Unified Reward Management System for Reinforcement Learning (GRPO) & Evaluation.

Supports two pluggable backends switchable via a single config toggle or .env setting:
1. 'local': 100% offline composite heuristic reward (strict lipogram check,
   length shaping, n-gram repetition penalty, 80k-word English lexicon verification,
   and prompt topical relevance). 0 cost, 0 latency, deterministic.
2. 'api': Remote LLM-as-a-judge via any OpenAI-compatible API endpoint
   (OpenRouter, OpenAI, Groq, DeepSeek, Local Ollama/vLLM).
"""

import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import yaml

from core.checks import e_pass, text_checks
from core.llm_api import OpenAIClient, FatalAPIError

ROOT = Path(__file__).resolve().parent.parent

def load_dotenv_file(env_path: Path):
    """Zero-dependency .env file parser."""
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and v and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass

# Common English stop words to exclude from repetition and topical overlap
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


# ==============================================================================
# 1. Local Composite Reward (Offline Heuristics)
# ==============================================================================

class LocalCompositeReward:
    """
    Evaluates response quality without any external APIs.
    Combines:
    - Hard lipogram constraint (0.0 if any 'e' or control chars present)
    - Length shaping (prevents length shrinking collapse)
    - N-Gram repetition penalty (prevents degenerative looping text)
    - Dictionary validity (penalizes invented or misspelled non-words)
    - Prompt keyword relevance (encourages on-topic answers)
    """

    def __init__(
        self,
        dictionary_path: Optional[str] = None,
        min_tokens: int = 15,
        target_min_tokens: int = 25,
        target_max_tokens: int = 80,
        max_penalty_tokens: int = 140,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.min_tokens = min_tokens
        self.target_min_tokens = target_min_tokens
        self.target_max_tokens = target_max_tokens
        self.max_penalty_tokens = max_penalty_tokens
        self.weights = weights or {"repetition": 0.40, "dictionary": 0.35, "relevance": 0.25}

        # Normalize weights
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

        # Load 80k-word English dictionary
        dict_file = Path(dictionary_path) if dictionary_path else (ROOT / "data" / "dictionary" / "english_words.txt")
        if not dict_file.is_absolute():
            dict_file = ROOT / dict_file

        self.dictionary = set()
        if dict_file.exists():
            self.dictionary = set(dict_file.read_text(encoding="utf-8").splitlines())
        else:
            # Fallback mini lexicon if file not yet built
            self.dictionary = set(["dog", "hound", "canine", "cat", "avian", "bird", "fly", "sun", "star", "water", "drink"])

    def _extract_words(self, text: str) -> List[str]:
        """Extract clean lower-case alphabetical words."""
        return [w.lower().strip(string.punctuation) for w in text.split() if w.strip(string.punctuation).isalpha()]

    def compute_length_score(self, num_tokens: int) -> float:
        """
        Soft trapezoidal length shaping:
        - 0.0 for < 5 tokens (stops 1-word cop-outs)
        - Smooth ramp from 5 to 25 tokens
        - 1.0 between 25 and 80 tokens (optimal lipogram paragraph)
        - Gentle decay above 80 tokens down to 0.5 at 140 tokens
        """
        if num_tokens < 5:
            return 0.0
        if num_tokens < self.target_min_tokens:
            return 0.2 + 0.8 * ((num_tokens - 5) / (self.target_min_tokens - 5))
        if num_tokens <= self.target_max_tokens:
            return 1.0
        if num_tokens <= self.max_penalty_tokens:
            decay_range = self.max_penalty_tokens - self.target_max_tokens
            return 1.0 - 0.5 * ((num_tokens - self.target_max_tokens) / decay_range)
        return 0.5

    def compute_repetition_score(self, words: List[str]) -> float:
        """
        Penalizes repeated 2-grams, 3-grams, and excessive single word looping.
        """
        if len(words) < 4:
            return 0.5

        # 1. 2-gram distinct ratio
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_ratio = len(set(bigrams)) / max(1, len(bigrams))

        # 2. 3-gram distinct ratio
        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        trigram_ratio = len(set(trigrams)) / max(1, len(trigrams)) if trigrams else 1.0

        # 3. Dominant non-stopword frequency
        non_stop = [w for w in words if w not in STOP_WORDS]
        max_freq_penalty = 1.0
        if non_stop:
            counts = Counter(non_stop)
            top_word, top_count = counts.most_common(1)[0]
            ratio = top_count / len(non_stop)
            if ratio > 0.25 and top_count >= 3:
                max_freq_penalty = max(0.2, 1.0 - (ratio - 0.25) * 2.0)

        rep_score = (0.5 * bigram_ratio + 0.5 * trigram_ratio) * max_freq_penalty
        return max(0.0, min(1.0, rep_score))

    def compute_dictionary_score(self, words: List[str]) -> float:
        """Computes fraction of words that are recognized English dictionary entries."""
        if not words:
            return 0.0
        valid = sum(1 for w in words if w in self.dictionary)
        return valid / len(words)

    def compute_relevance_score(self, prompt: str, response_words: List[str]) -> float:
        """Computes topical keyword overlap between prompt and response."""
        prompt_words = [w.lower().strip(string.punctuation) for w in prompt.split()]
        prompt_keywords = {w for w in prompt_words if len(w) >= 3 and w not in STOP_WORDS}
        if not prompt_keywords:
            return 0.8  # Neutral if prompt had no distinct keywords

        resp_set = set(response_words)
        # Direct keyword overlap
        matches = sum(1 for kw in prompt_keywords if kw in resp_set or any(kw in rw or rw in kw for rw in resp_set))
        coverage = matches / len(prompt_keywords)
        # Rescale coverage so 50% keyword match gives near full topical reward
        return min(1.0, coverage * 2.0)

    def score(self, prompt: str, answer: str, cut_off: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate full response quality. Returns (reward_scalar_in_0_to_1, diagnostics_dict).
        """
        raw_words = answer.split()
        num_tokens = len(raw_words)
        words = self._extract_words(answer)

        # 1. Hard Lipogram Gate
        checks = text_checks(answer)
        lipogram_ok = e_pass(answer) and checks["control_ok"]

        if not lipogram_ok or num_tokens == 0:
            return 0.0, {
                "reward": 0.0,
                "lipogram_pass": False,
                "e_count": checks.get("e_count", 0),
                "num_tokens": num_tokens,
                "length_score": 0.0,
                "repetition_score": 0.0,
                "dictionary_score": 0.0,
                "relevance_score": 0.0,
                "mode": "local",
                "note": "Forbidden character 'e' detected or malformed control character."
            }

        # 2. Heuristic Dimension Scores
        s_len = self.compute_length_score(num_tokens)
        s_rep = self.compute_repetition_score(words)
        s_dict = self.compute_dictionary_score(words)
        s_rel = self.compute_relevance_score(prompt, words)

        # 3. Composite Calculation
        quality_score = (
            self.weights["repetition"] * s_rep +
            self.weights["dictionary"] * s_dict +
            self.weights["relevance"] * s_rel
        )
        final_reward = round(float(s_len * quality_score), 4)

        diagnostics = {
            "reward": final_reward,
            "lipogram_pass": True,
            "e_count": 0,
            "num_tokens": num_tokens,
            "length_score": round(s_len, 4),
            "repetition_score": round(s_rep, 4),
            "dictionary_score": round(s_dict, 4),
            "relevance_score": round(s_rel, 4),
            "mode": "local",
            "note": f"Local heuristic reward (Tokens={num_tokens}, Dict={s_dict*100:.0f}%, Rep={s_rep*100:.0f}%)"
        }
        return final_reward, diagnostics


# ==============================================================================
# 2. Remote API-as-a-Judge (OpenAI / OpenRouter Endpoint)
# ==============================================================================

JUDGE_SYSTEM_PROMPT = """You grade short answers written by a small chat model. You will see a user's request and the model's answer. Score two things from 1 to 5.

Coherence: is the answer readable, sensible English?
5 = clear and natural. Every word is a real, correctly spelled English word, sentences are grammatical, and the ideas connect from start to finish.
4 = clear overall, with at most one minor awkward phrase or slip.
3 = understandable but with noticeable problems: broken phrasing, a misspelled or made-up word, or some repetition.
2 = hard to follow: several misspelled or made-up words, broken grammar, or looping text.
1 = gibberish, mostly repetition, or not English.
Repetition counts against coherence. If the same phrase or line appears three or more times, or the answer keeps circling back to the same words instead of moving forward, coherence is 3 at most. If most of the answer is repetition, it is 1 or 2.

Responds: does the answer address the request?
5 = directly attempts what was asked.
4 = attempts what was asked, with small gaps or slight drift.
3 = partly on topic, or attempts only part of the request.
2 = mostly off topic, or generic filler that could follow almost any request.
1 = ignores the request.
Do not judge factual accuracy. Wrong facts or made-up details don't lower either score, as long as the answer attempts the request.

Reply with only this JSON object and no other text:
{"coherence": <1-5>, "responds": <1-5>, "note": "<one sentence explaining the scores>"}"""


class APIJudgeReward:
    """
    Calls an external LLM-as-a-judge via an OpenAI-compatible API endpoint
    (e.g., OpenRouter, OpenAI, Groq).
    """

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        model: str = "openai/gpt-4o-mini",
        combine: str = "sum",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        fallback_reward: Optional[LocalCompositeReward] = None,
    ):
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.model = model
        self.combine = combine
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.fallback = fallback_reward or LocalCompositeReward()
        self._client: Optional[OpenAIClient] = None

    def _get_client(self) -> Optional[OpenAIClient]:
        if self._client is not None:
            return self._client

        # Load environment variables from .env
        load_dotenv_file(ROOT / ".env")
        api_key = os.environ.get(self.api_key_env) or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        base_url = (
            os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self.base_url
        )

        headers = {
            "HTTP-Referer": "https://github.com/qwen-no-e",
            "X-Title": "Qwen Minus the E RL Training",
        }

        self._client = OpenAIClient(
            model=self.model,
            base_url=base_url,
            api_key=api_key,
            max_retries=self.max_retries,
            default_headers=headers,
        )
        return self._client

    def score(self, prompt: str, answer: str, cut_off: bool = False) -> Tuple[float, Dict[str, Any]]:
        # Hard constraint check: if answer contains 'e', instant 0.0 without wasting API calls!
        if not e_pass(answer):
            return 0.0, {
                "reward": 0.0,
                "lipogram_pass": False,
                "coherence": 1,
                "responds": 1,
                "mode": "api",
                "note": "Skipped API call: forbidden character 'e' detected."
            }

        client = self._get_client()
        if client is None:
            # Graceful fallback to local reward
            print(f"⚠️ [APIJudgeReward] No API key found in {self.api_key_env} or OPENAI_API_KEY. Falling back to Local Composite Reward!")
            score_val, diag = self.fallback.score(prompt, answer, cut_off)
            diag["mode"] = "local_fallback"
            return score_val, diag

        label = "Answer (cut off at length limit)" if cut_off else "Answer"
        messages = [
            {"role": "user", "content": f"Request:\n<<<\n{prompt}\n>>>\n\n{label}:\n<<<\n{answer}\n>>>"}
        ]

        try:
            completion = client.complete(system=JUDGE_SYSTEM_PROMPT, messages=messages, max_tokens=256)
            text = completion.text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"Unparseable response: {text}")
            data = json.loads(match.group(0))

            coherence = int(data.get("coherence", 3))
            responds = int(data.get("responds", 3))
            note = str(data.get("note", ""))

            # Combine scores into [0.0, 1.0]
            if self.combine == "sum":
                scalar_reward = ((coherence - 1) + (responds - 1)) / 8.0
            elif self.combine == "lowest":
                scalar_reward = (min(coherence, responds) - 1) / 4.0
            else:
                scalar_reward = ((coherence - 1) + (responds - 1)) / 8.0

            scalar_reward = round(max(0.0, min(1.0, scalar_reward)), 4)
            return scalar_reward, {
                "reward": scalar_reward,
                "lipogram_pass": True,
                "coherence": coherence,
                "responds": responds,
                "mode": "api",
                "model": client.model,
                "note": note
            }
        except Exception as e:
            print(f"⚠️ [APIJudgeReward] API Call Error ({type(e).__name__}: {e}). Falling back to Local Composite Reward!")
            score_val, diag = self.fallback.score(prompt, answer, cut_off)
            diag["mode"] = "local_fallback"
            diag["api_error"] = str(e)
            return score_val, diag


# ==============================================================================
# 3. Unified Reward Manager & Factory
# ==============================================================================

class RewardManager:
    """
    Main entry point for evaluating rewards in RL & evaluation scripts.
    Handles mode toggle, config loading, and unified scoring API.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg_file = Path(config_path) if config_path else (ROOT / "configs" / "default.yaml")
        if not cfg_file.is_absolute():
            cfg_file = ROOT / cfg_file

        self.config: Dict[str, Any] = {}
        if cfg_file.exists():
            self.config = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}

        reward_cfg = self.config.get("reward", {})
        self.mode = reward_cfg.get("mode", "local").lower()

        # Build local engine
        loc_cfg = reward_cfg.get("local", {})
        self.local_reward = LocalCompositeReward(
            dictionary_path=loc_cfg.get("dictionary_path"),
            min_tokens=loc_cfg.get("min_length_tokens", 15),
            target_min_tokens=loc_cfg.get("target_min_tokens", 25),
            target_max_tokens=loc_cfg.get("target_max_tokens", 80),
            max_penalty_tokens=loc_cfg.get("max_penalty_tokens", 140),
            weights=loc_cfg.get("weights"),
        )

        # Build API engine
        api_cfg = reward_cfg.get("api", {})
        self.api_reward = APIJudgeReward(
            base_url=api_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            api_key_env=api_cfg.get("api_key_env", "OPENROUTER_API_KEY"),
            model=api_cfg.get("model", "openai/gpt-4o-mini"),
            combine=api_cfg.get("combine", "sum"),
            temperature=api_cfg.get("temperature", 0.0),
            max_retries=api_cfg.get("max_retries", 3),
            timeout=api_cfg.get("timeout", 30.0),
            fallback_reward=self.local_reward,
        )

    def set_mode(self, mode: str):
        """Dynamically switch between 'local' and 'api' modes."""
        mode = mode.lower()
        if mode not in ("local", "api"):
            raise ValueError(f"Invalid reward mode '{mode}'. Choose 'local' or 'api'.")
        self.mode = mode

    def evaluate(self, prompt: str, answer: str, cut_off: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a single (prompt, answer) pair.
        Returns:
            (reward_float, diagnostics_dict)
        """
        if self.mode == "api":
            return self.api_reward.score(prompt, answer, cut_off)
        return self.local_reward.score(prompt, answer, cut_off)

    def evaluate_batch(
        self,
        prompts: List[str],
        answers: List[str],
        cut_offs: Optional[List[bool]] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Evaluate a batch of responses sequentially."""
        if cut_offs is None:
            cut_offs = [False] * len(prompts)
        return [self.evaluate(p, a, c) for p, a, c in zip(prompts, answers, cut_offs)]


# Global default instance
_default_manager: Optional[RewardManager] = None

def get_reward_manager(config_path: Optional[str] = None) -> RewardManager:
    global _default_manager
    if _default_manager is None or config_path is not None:
        _default_manager = RewardManager(config_path)
    return _default_manager
