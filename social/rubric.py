"""The rubric: ten typed questions, each producing one numeric column to regress.

Editing any question's wording, type, or criteria REQUIRES bumping RUBRIC_VERSION.
Scores are keyed by version, so two rubric generations never mix into one regression.
"""

from .models import MAX_POST_CHARS

RUBRIC_VERSION = "1.1"  # 1.1 dropped ai_slop: measured inverted on both checkpoints

# Questions whose answer is a label rather than a magnitude. Analysis one-hot encodes these.
CHOICE_QUESTIONS = ("hook_type", "format", "cta_type")


def post_state(post, max_chars: int = MAX_POST_CHARS) -> dict:
    """Build the model input. Platform is included: the same question means something
    different on a 280-character X post than on a 1,300-character LinkedIn post."""
    text = " ".join((post.text or "").split())
    return {"platform": post.platform, "text": text[:max_chars]}


def post_questions() -> dict:
    return {
        "hook_type": {
            "type": "choice",
            "instructions": "What kind of opening line does this post use?",
            "criteria": {
                "curiosity_gap": "withholds the payoff to force a click on 'see more'",
                "contrarian": "opens by rejecting a widely held belief",
                "result_claim": "opens with a concrete outcome, number, or revenue figure",
                "story_open": "opens mid-scene in a personal narrative",
                "anaphora": "opens with a repeated sentence structure across short lines",
                "question": "opens by asking the reader something directly",
                "list_promise": "opens by promising N lessons, tips, or steps",
                "gratitude": "opens by thanking or congratulating a person or group",
            },
        },
        "format": {
            "type": "choice",
            "instructions": "What is the overall shape of this post?",
            "criteria": {
                "story": "a single narrative with a beginning and an outcome",
                "listicle": "an enumerated set of points",
                "hot_take": "a short opinion asserted without much supporting detail",
                "case_study": "a specific situation walked through with numbers or steps",
                "announcement": "news about the author, their company, or a launch",
                "teardown": "a critique or breakdown of someone else's work or approach",
            },
        },
        "specificity": {
            "type": "score",
            "instructions": "How specific and concrete is this post?",
            "criteria": [
                "general platitudes with no verifiable detail",
                "some concrete detail but mostly generic advice",
                "named tools, roles, or timeframes throughout",
                "hard numbers, dates, or dollar figures anchoring the claims",
            ],
        },
        "actionability": {
            "type": "score",
            "instructions": "Could a reader act on this post tomorrow morning?",
            "criteria": [
                "nothing to act on",
                "a vague direction to think about",
                "a clear principle the reader could apply",
                "explicit steps the reader could follow immediately",
            ],
        },
        "emotional_charge": {
            "type": "score",
            "instructions": "How emotionally charged is the language in this post?",
            "criteria": [
                "flat and purely informational",
                "mildly warm or mildly critical",
                "clearly charged with pride, frustration, or urgency",
                "intensely charged, confrontational, or raw",
            ],
        },
        "reading_effort": {
            "type": "score",
            "instructions": "How much effort does reading this post require?",
            "criteria": [
                "skimmable in seconds, short lines and whitespace",
                "quick read with some density",
                "requires sustained attention",
                "dense wall of text demanding real effort",
            ],
        },
        "takes_a_position": {
            "type": "noul",
            "instructions": (
                "Does this post stake out a position a reasonable peer could publicly "
                "disagree with?"
            ),
            "criteria": {
                "true": "asserts a debatable claim someone could argue against",
                "false": "states only uncontroversial or universally agreeable things",
            },
        },
        "personal_stakes": {
            "type": "noul",
            "instructions": (
                "Does the author expose personal risk, failure, or vulnerability in this post?"
            ),
            "criteria": {
                "true": "the author describes their own failure, fear, loss, or exposure",
                "false": "the author describes only successes, observations, or other people",
            },
        },
        "cta_type": {
            "type": "choice",
            "instructions": "What does this post ask the reader to do at the end?",
            "criteria": {
                "none": "no request of the reader",
                "comment_bait": "asks for a reply, an opinion, or a keyword in the comments",
                "dm": "asks the reader to message the author",
                "link": "directs the reader to a link, in the post or in the comments",
                "follow": "asks the reader to follow or subscribe",
            },
        },
    }
