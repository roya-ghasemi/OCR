"""GBNF grammar generation - the structural fix for field-schema collapse.

WHY THIS WORKS WHEN response_format DOES NOT
--------------------------------------------
llama.cpp exposes two different constrained-decoding paths and they do not
behave the same on this build:

  * `response_format` (the OpenAI-style parameter) is SILENTLY IGNORED on
    multimodal requests. Verified by sending an image with and without a
    json_schema and getting byte-identical output, while the same schema on a
    text-only request constrained correctly. Silent, so it looks like it worked.
  * `grammar` (raw GBNF, llama.cpp-native, via extra_body) IS honoured on the
    multimodal path. Verified on a page that previously dumped the entire letter
    into `sender`: with the grammar it returned all five fields and parsed.

Use `grammar`. Never rely on `response_format` here.

HOW IT KILLS THE COLLAPSE
-------------------------
The failure is that the model writes the whole letter into `sender` and never
closes the string. A grammar with a PER-FIELD LENGTH BOUND makes that
unrepresentable: once `sender` reaches its cap the only legal continuation is the
closing quote, then the `receiver` key. The model cannot dump 1600 characters
into a 160-character field, so it is forced to move on and populate the rest.

That is a structural guarantee about SHAPE. It is not a guarantee about
CORRECTNESS - a bounded grammar will happily put the wrong text in the right
field. Measured: forcing all five fields filled them all, but `contact_info`
received body text rather than the footer. Grammar plus prompt, not grammar
instead of prompt.

TWO PARSER LIMITATIONS IN THIS BUILD - both found by probing, both worked around
-------------------------------------------------------------------------------
  1. A backslash inside ANY character class fails to parse. `[^"\\]` and
     `["\\/bfnrt]` are both rejected. This rules out the textbook JSON string
     rule, which is why the obvious grammar fails with "failed to parse grammar".
  2. Unbounded `char*` inside a string lets the model ramble until the token cap
     and never emit the closing quote. Bounded `char{0,N}` is what makes the
     grammar useful rather than merely valid.

Workaround for (1): exclude the quote and raw newline with `[^"\n]`, which parses
fine, and permit the two-character escape `"\\n"` as a STRING LITERAL alternative
rather than a class member. That keeps line breaks in `body_text` while producing
JSON that `json.loads` accepts with no escape handling.

TOKEN BUDGET
------------
The caps are in CHARACTERS; Persian costs roughly 2-3 tokens per character in
this tokenizer. `recommended_max_tokens()` derives a budget from the caps. A
grammar whose caps exceed `max_tokens` truncates mid-object and defeats itself -
measured, on a page where the sum of caps overran a 2048 budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# Field order is the order the grammar enforces. It is deliberately the reading
# order of an Iranian administrative letter - letterhead, addressee, subject,
# body, footer - so the constraint runs with the document rather than against it.
FIELD_ORDER = ("sender", "receiver", "subject", "body_text", "contact_info")

# Character caps, sized from the corpus. Generous enough not to truncate a real
# value, tight enough that a whole letter cannot fit in the wrong field.
DEFAULT_CAPS: Mapping[str, int] = {
    "sender": 180,
    "receiver": 180,
    "subject": 240,
    "body_text": 1400,
    "contact_info": 320,
}

# Persian is multi-token per character in this tokenizer; plus JSON scaffolding.
_TOKENS_PER_CHAR = 2.2
_SCAFFOLD_TOKENS = 96


def _rule_name(field_name: str) -> str:
    """GBNF rule name for a JSON field.

    Third parser quirk in this build: an UNDERSCORE in a rule name fails to
    parse, while a dash is accepted. Two of the five fields are `body_text` and
    `contact_info`, so a naive `v_{name}` rule name breaks the whole grammar with
    the same opaque "failed to parse grammar" as a genuine syntax error.
    """
    return "v-" + field_name.replace("_", "-")


@dataclass(frozen=True)
class GrammarSpec:
    caps: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    allow_newlines_in: tuple[str, ...] = ("body_text", "contact_info")

    def build(self) -> str:
        """Emit GBNF enforcing all five keys, in order, each bounded."""
        parts = ['root ::= "{" ws']
        for i, name in enumerate(FIELD_ORDER):
            sep = "" if i == 0 else '"," ws'
            parts.append(f'{sep} "\\"{name}\\":" ws {_rule_name(name)}')
        parts.append('ws "}"')
        lines = [" ".join(parts)]

        for name in FIELD_ORDER:
            cap = int(self.caps.get(name, 200))
            charset = "nlchar" if name in self.allow_newlines_in else "plainchar"
            # `null` first: a genuinely absent field must stay cheap to express,
            # otherwise the grammar pressures the model into inventing content.
            lines.append(
                f'{_rule_name(name)} ::= "null" | "\\"" {charset}{{0,{cap}}} "\\""'
            )

        # Excluding the backslash matters: if content may contain a raw `\`, the
        # model can emit an invalid escape like `\س` and the object stops being
        # parseable JSON - observed in testing before this line was tightened.
        #
        # A LITERAL backslash in a character class fails this build's parser, but
        # the HEX form does not. `\x5C` is therefore how the backslash gets
        # excluded, and `\x0A` is the newline. The only escape the grammar can
        # produce is the explicit `\n` alternative in `nlchar`, so every string
        # it emits is valid JSON by construction.
        lines.append('plainchar ::= [^"\\x5C\\x0A]')
        lines.append('nlchar ::= [^"\\x5C\\x0A] | "\\\\n"')
        lines.append('ws ::= [ \\t\\n]*')
        return "\n".join(lines)

    def recommended_max_tokens(self) -> int:
        """Budget that can actually hold a maximal instance of this grammar.

        A grammar the token budget cannot satisfy is worse than no grammar: the
        model is forced to keep writing and then gets cut off mid-object.
        """
        chars = sum(self.caps.values())
        return int(chars * _TOKENS_PER_CHAR) + _SCAFFOLD_TOKENS


DEFAULT_SPEC = GrammarSpec()


def build_grammar(caps: Mapping[str, int] | None = None) -> str:
    return GrammarSpec(caps=dict(caps or DEFAULT_CAPS)).build()


# -- prompt ------------------------------------------------------------------
# The grammar enforces shape; the prompt is what steers CONTENT into the right
# slot. Field descriptions are explicit about WHERE on the page each value lives,
# because "sender" alone does not tell the model to read the letterhead.
SYSTEM_PROMPT = """\
تو یک موتور OCR دقیق برای نامه‌های اداری رسمی ایرانی هستی.

خروجی تو فقط و فقط یک شیء JSON معتبر است. هیچ توضیح، هیچ markdown، هیچ متن اضافه.

ساختار دقیق خروجی:

{
  "sender":       "<فقط نام سازمان یا شرکت فرستنده — از سربرگ بالای صفحه یا مهر و امضای پایین. فقط نام، نه متن نامه>",
  "receiver":     "<فقط نام یا سمت گیرنده — معمولاً بعد از «جناب آقای»، «سرکار خانم»، «ریاست محترم»، «مدیریت محترم»>",
  "subject":      "<فقط متن جلوی کلمه «موضوع:» — یک عبارت کوتاه، نه کل نامه>",
  "body_text":    "<متن اصلی نامه: از «با سلام» یا «احتراماً» تا امضا. سربرگ و اطلاعات تماس پانویس را اینجا نیاور>",
  "contact_info": "<فقط بلوک تماس در پانویس صفحه: آدرس، تلفن، فکس، کد پستی، ایمیل>"
}

قوانین الزامی:
۱. هر پنج کلید باید در خروجی باشند. اگر مقداری در سند نیست، دقیقاً null بگذار.
۲. هر بخش از متن فقط در یک فیلد بیاید. هیچ متنی را در دو فیلد تکرار نکن.
۳. کل نامه را در فیلد sender نگذار — sender فقط نام فرستنده است.
۴. اعداد را دقیقاً همان‌طور که در تصویر است بنویس. رقم‌ها را تغییر نده.
۵. متن ناخوانا را با [ناخوانا] مشخص کن. هیچ اطلاعاتی را حدس نزن یا جعل نکن.
۶. اگر تصویر نامه اداری نیست یا خوانا نیست، همه فیلدها را null بگذار.
"""

USER_TURN = "متن این نامه اداری را OCR کن و طبق ساختار خواسته‌شده JSON بده."
