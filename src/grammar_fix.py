import requests

class GrammarFix:
    def __init__(self, model="tinyllama", use_llm=True):
        self.model    = model
        self.use_llm  = use_llm
        self.api_url  = "http://localhost:11434/api/generate"

    # ── Step 1: Rule-based pre-processing ─────────────────────────────────
    def rule_based(self, words):
        if not words:
            return []
        words = [w.lower() for w in words]

        subjects = {"i", "you", "he", "she", "we", "they"}
        verbs    = {"want", "need", "like", "go", "eat",
                    "drink", "help", "see", "call", "feel"}

        subject = [w for w in words if w in subjects]
        verb    = [w for w in words if w in verbs]
        other   = [w for w in words if w not in subjects and w not in verbs]

        # Rebuild in Subject → Verb → Object order
        reordered = subject + verb + other

        # If nothing matched, keep original order
        if not reordered:
            reordered = words

        # Move "please" to end
        if "please" in reordered and reordered[-1] != "please":
            reordered.remove("please")
            reordered.append("please")

        # Remove consecutive duplicates
        result = [reordered[0]]
        for w in reordered[1:]:
            if w != result[-1]:
                result.append(w)

        return result

    # ── Step 2: TinyLlama polish ───────────────────────────────────────────
    def llm_polish(self, words):
        word_str = " ".join(words)
        prompt = (
            f"Fix this into one clean English sentence.\n"
            f"Words: {word_str}\n"
            f"Output only the sentence with no extra text, "
            f"no quotes, no explanation:\n"
        )
        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=20
            )
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                # Take only first line, strip quotes and labels
                sentence = raw.split("\n")[0].strip().strip('"').strip("'")
                if ":" in sentence:
                    sentence = sentence.split(":", 1)[-1].strip()
                if len(sentence) > 5:
                    return sentence
        except Exception as e:
            print(f"⚠️  LLM unavailable: {e}")
        return None

    # ── Main fix function ──────────────────────────────────────────────────
    def fix(self, words):
        if not words:
            return ""
        if len(words) == 1:
            return words[0].capitalize() + "."

        # Step 1: rule-based
        rule_words   = self.rule_based(words)
        rule_sentence = " ".join(rule_words)
        rule_sentence = rule_sentence[0].upper() + rule_sentence[1:] + "."

        print(f"  📐 Rule-based : {rule_sentence}")

        # Step 2: LLM polish
        if False:  
            llm_sentence = self.llm_polish(rule_words)
            if llm_sentence:
                # Capitalise and add period if missing
                llm_sentence = llm_sentence[0].upper() + llm_sentence[1:]
                if not llm_sentence.endswith((".", "!", "?")):
                    llm_sentence += "."
                print(f"  🤖 TinyLlama  : {llm_sentence}")
                return llm_sentence

        return rule_sentence


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gf = GrammarFix(use_llm=True)

    test_cases = [
        ["water", "want", "i"],
        ["help", "please"],
        ["go", "bathroom", "i"],
        ["food", "more", "want"],
        ["thank", "you"],
        ["i", "need", "help"],
    ]

    print("Testing GrammarFix (Rule + LLM)...\n")
    for words in test_cases:
        print(f"Input      : {words}")
        result = gf.fix(words)
        print(f"Final      : {result}")
        print()

    print("✅ GrammarFix test done!")   