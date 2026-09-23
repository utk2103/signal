"""Golden posts for the rubric: hand-written exemplars, asserted into bands not values.

Bands are calibrated against a real run of MODEL below, not guessed. Every band sits
well inside its measured margin, and every dimension the checkpoint does NOT separate is
marked `xfail(strict=True)` with the measured number rather than asserted loosely enough
to pass -- so a rubric or checkpoint that improves breaks this file loudly.

Needs downloaded checkpoints, so the whole module is marked `integration`.
"""

import pytest

import laya_mlx as laya
from social.models import Post
from social.rubric import post_questions, post_state

pytestmark = pytest.mark.integration

MODEL = "aac6fef/laya-mlx"

# (platform, text). Each post is written to sit at one extreme of one dimension, so a
# band assertion below has something unambiguous to land on.
GOLDEN = {
    # --- the eight hook types ---
    "hook_curiosity_gap": (
        "linkedin",
        "I lost our biggest client on a Tuesday afternoon.\n\n"
        "What the CEO said in that call changed how I run every account review.\n\n"
        "I have never told anyone this part.\n\n"
        "Here is what actually happened.",
    ),
    "hook_contrarian": (
        "linkedin",
        "Everyone says you should post every day on LinkedIn. That advice is wrong.\n\n"
        "Daily posting trains your audience to skim you. Volume is not distribution.\n\n"
        "We cut to two posts a week and stopped optimising for the feed.",
    ),
    "hook_result_claim": (
        "linkedin",
        "We took a client from $40k to $310k MRR in 11 months.\n\n"
        "No new headcount. No paid budget. One channel.\n\n"
        "Here is the exact sequence we ran, month by month.",
    ),
    "hook_story_open": (
        "linkedin",
        "The rain was coming sideways and I was sitting in a rented Corolla outside "
        "the client's office, rehearsing the sentence I did not want to say.\n\n"
        "Forty minutes later I walked out having lost the account.\n\n"
        "That drive home taught me more than the four years before it.",
    ),
    "hook_anaphora": (
        "linkedin",
        "Stop optimising your headline.\n"
        "Stop rewriting your About section.\n"
        "Stop A/B testing your profile photo.\n\n"
        "Start talking to the twelve people who already replied to you.",
    ),
    "hook_question": (
        "linkedin",
        "Why do most agency retainers die in month four?\n\n"
        "It is almost never the work. It is that nobody renegotiated scope when the "
        "scope quietly doubled.",
    ),
    "hook_list_promise": (
        "linkedin",
        "7 lessons from running a 20-person agency for six years:\n\n"
        "1. Fire the client who emails on Sunday.\n"
        "2. Price on outcome, never on hours.\n"
        "3. Hire the second strategist before you need them.\n"
        "4. Write the SOP the first time you do a thing twice.\n"
        "5. Never discount, restructure scope instead.\n"
        "6. Your best hire is somebody else's frustrated senior.\n"
        "7. Cash flow beats revenue every single quarter.",
    ),
    "hook_gratitude": (
        "linkedin",
        "Huge thank you to Priya, Marcus and the whole team at Northwind for four "
        "incredible years.\n\n"
        "Congratulations to Marcus on stepping up to lead the practice. Could not be "
        "happier for you both.",
    ),
    # --- specificity: platitudes vs. hard numbers ---
    "specificity_low": (
        "linkedin",
        "Success is a journey, not a destination.\n\n"
        "The best leaders lead with empathy. They listen more than they speak. They "
        "lift others as they climb.\n\n"
        "Be the kind of person people remember. That is what really matters.",
    ),
    "specificity_high": (
        "linkedin",
        "Q3 numbers for the agency, in full:\n\n"
        "Revenue $1.24M, up 18% on Q2.\n"
        "Gross margin 61%, down from 64% after two senior hires in August.\n"
        "Churn 1 logo of 34, a $9k/mo retainer that ended 14 August.\n"
        "CAC $4,180 across 11 closed-won, payback 3.1 months.\n"
        "Headcount 22, two open roles on the paid media pod.",
    ),
    # --- actionability: nothing to act on vs. explicit steps ---
    "actionability_low": (
        "linkedin",
        "Sometimes I sit with a coffee and think about how far the industry has come.\n\n"
        "Strange times. Interesting times. Onwards.",
    ),
    "actionability_high": (
        "linkedin",
        "How to audit a stalled retainer in 30 minutes, tomorrow morning:\n\n"
        "1. Open the original SOW and list every deliverable it named.\n"
        "2. Pull the last 8 weeks of delivered work into a second list.\n"
        "3. Highlight every line in list two that is not in list one.\n"
        "4. Total the hours against those highlighted lines.\n"
        "5. Send the client that total with one sentence: 'this is scope we have "
        "absorbed, here are two options.'\n"
        "6. Offer either a rate change or a deliverable cut. Never both at once.",
    ),
    # --- emotional charge: flat vs. raw ---
    "emotion_low": (
        "linkedin",
        "Note on scheduling: our offices will be closed 24 to 26 December and will "
        "reopen on 29 December. Support tickets will be triaged once daily during "
        "that period. Response times return to normal on 2 January.",
    ),
    "emotion_high": (
        "linkedin",
        "I am furious, and I am going to say this plainly.\n\n"
        "A founder in this network just pitched a junior designer on 'exposure' for "
        "three weeks of full-time work. She is 23. She said yes because she was "
        "scared to say no.\n\n"
        "This is theft with a nicer font. I am done being polite about it.",
    ),
    # --- reading effort: skimmable vs. dense wall ---
    "effort_low": (
        "x",
        "Pricing tip.\n\nCharge for the outcome.\n\nNot the hours.\n\nThat is it.",
    ),
    "effort_high": (
        "linkedin",
        "The structural problem with agency retainer pricing is that it inherits its "
        "unit of account from professional services billing, which assumes that the "
        "marginal cost of delivery scales linearly with the hours consumed, an "
        "assumption that held reasonably well for bespoke consulting engagements but "
        "breaks down entirely once a meaningful share of delivery is systematised "
        "into repeatable processes, templated assets and partially automated "
        "workflows, at which point the hourly frame actively penalises the agency for "
        "every efficiency it invests in, because each hour removed from the delivery "
        "cycle is an hour removed from billable revenue, producing the perverse "
        "incentive structure whereby the most operationally mature agency in a "
        "category is also the one whose revenue per client declines fastest, unless "
        "and until it migrates to an outcome-denominated or retainer-flat pricing "
        "model that decouples the value delivered from the labour consumed.",
    ),
    # --- personal stakes: exposed failure vs. observation about others ---
    "stakes_high": (
        "linkedin",
        "I missed payroll in March. Twice.\n\n"
        "I covered it from my own savings and did not tell the team, which I now "
        "think was cowardice dressed up as protection.\n\n"
        "I have never been more frightened of anything in my working life.",
    ),
    "stakes_low": (
        "linkedin",
        "Interesting pattern across the agencies we work with: the ones growing "
        "fastest have moved their strategist headcount ahead of their delivery "
        "headcount, not behind it.",
    ),
    # --- Hinglish, Latin script, code-mixed: the register Indian agency LinkedIn
    # actually posts in. The client corpus is not English-only, so the golden set
    # must say out loud how much of the rubric survives the language switch.
    "hinglish_specificity_low": (
        "linkedin",
        "Kuch log kehte hain success overnight milta hai. Sach ye hai ki consistency "
        "hi sabse bada hack hai.\n\n"
        "Bas lage raho. Mehnat kabhi waste nahi jaati. Universe dekh raha hai.\n\n"
        "Keep grinding, friends.",
    ),
    "hinglish_specificity_high": (
        "linkedin",
        "Pichle quarter ke numbers, bilkul transparent:\n\n"
        "Revenue Rs 84.2 lakh, Q2 se 23% upar.\n"
        "Team 14 se 19 hui, 3 hires sirf performance marketing pod mein.\n"
        "Ek client churn hua, Rs 2.4 lakh/month ka retainer, 14 August ko end.\n"
        "CAC Rs 31,000, payback 2.7 months.\n"
        "Cash runway 11 months hai abhi.",
    ),
    "hinglish_actionability_low": (
        "linkedin",
        "Aaj subah chai peete peete soch raha tha ki industry kitni badal gayi hai.\n\n"
        "Ajeeb time hai. Interesting bhi. Chalte raho.",
    ),
    "hinglish_actionability_high": (
        "linkedin",
        "Client ka stalled retainer kal subah 30 minute mein audit karna hai? Ye karo:\n\n"
        "1. Original SOW kholo aur har deliverable likho jo usme tha.\n"
        "2. Last 8 hafte ka delivered kaam ek doosri list mein daalo.\n"
        "3. List 2 ki har line highlight karo jo list 1 mein nahi hai.\n"
        "4. Un highlighted lines ke against total hours nikalo.\n"
        "5. Client ko wo total bhejo ek line ke saath: 'ye scope humne absorb kiya hai.'\n"
        "6. Ya rate badhao ya deliverable kam karo. Dono ek saath kabhi mat karo.",
    ),
}


@pytest.fixture(scope="module")
def answers():
    agent = laya.load(MODEL, cache_prompts=True, batch_size=32)
    questions = post_questions()
    out = {}
    for name, (platform, text) in GOLDEN.items():
        post = Post(
            url=f"https://example.com/{name}",
            platform=platform,
            corpus="golden",
            author="golden",
            text=text,
            posted_at="2026-09-22T00:00:00+00:00",
            likes=0,
            comments=0,
            reposts=0,
        )
        out[name] = agent.predict(post_state(post), questions)["answers"]
    return out




def test_corpus_covers_every_hook_type():
    """The fixture is only as good as its coverage; fail loudly if a hook is dropped."""
    covered = {name.removeprefix("hook_") for name in GOLDEN if name.startswith("hook_")}
    assert covered == set(post_questions()["hook_type"]["criteria"])


def xfail(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


# Bands. Every number below sits inside a gap measured on MODEL, with margin to spare.
SCORE_GAP = 0.8  # tightest passing score gap measured is +1.21, so ~34% headroom
NOUL_GAP = 0.15  # tightest passing noul gap measured is +0.257
HOOK_FLOOR = 0.125  # uniform over the eight hook labels


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("hook_contrarian", "contrarian"),
        ("hook_story_open", "story_open"),
        ("hook_question", "question"),
        ("hook_list_promise", "list_promise"),
        ("hook_gratitude", "gratitude"),
        pytest.param("hook_curiosity_gap", "curiosity_gap",
                     marks=xfail("measured 0.001; reads as story_open at 0.99")),
        pytest.param("hook_result_claim", "result_claim",
                     marks=xfail("measured 0.040; reads as story_open at 0.95")),
        pytest.param("hook_anaphora", "anaphora",
                     marks=xfail("measured 0.012; reads as list_promise at 0.74")),
    ],
)
def test_hook_type_beats_the_uniform_baseline(answers, name, expected):
    """Eight labels, so uniform is 0.125. The intended hook must clear that band.

    Measured: gratitude 1.000, contrarian 0.947, story_open 0.631, question 0.314,
    list_promise 0.232. The three xfails read 0.001-0.040 -- this checkpoint cannot
    tell a withheld payoff or a revenue-figure opener from a narrative opener.
    """
    assert answers[name]["hook_type"]["probabilities"][expected] > HOOK_FLOOR


@pytest.mark.parametrize(
    ("question", "low", "high"),
    [
        ("specificity", "specificity_low", "specificity_high"),
        ("actionability", "actionability_low", "actionability_high"),
        ("emotional_charge", "emotion_low", "emotion_high"),
        ("reading_effort", "effort_low", "effort_high"),
    ],
)
def test_score_dimension_separates_its_extremes(answers, question, low, high):
    """A band on the gap, not on absolute values.

    The regression needs each column to MOVE with its construct; where it lands on the
    0-3 ordinal scale is calibration, not signal.
    Measured gaps, all four clearing SCORE_GAP: specificity +2.04 (0.68 -> 2.72),
    emotional_charge +1.91 (0.57 -> 2.48), actionability +1.63 (1.11 -> 2.74),
    reading_effort +1.21 (0.58 -> 1.79).
    """
    gap = answers[high][question]["score"] - answers[low][question]["score"]
    assert gap >= SCORE_GAP, f"{question} gap {gap:+.2f}"


@pytest.mark.parametrize(
    ("question", "low", "high"),
    [
        ("specificity", "hinglish_specificity_low", "hinglish_specificity_high"),
        pytest.param("actionability", "hinglish_actionability_low",
                     "hinglish_actionability_high",
                     marks=xfail("measured gap -0.40 (2.09 -> 1.69): INVERTED on "
                                 "Hinglish while the same rubric question separates "
                                 "+1.63 on English. Do not bet actionability on a "
                                 "Hinglish corpus.")),
    ],
)
def test_score_dimension_separates_its_extremes_in_hinglish(answers, question, low, high):
    """Same band, code-mixed Latin-script input. Held to the English standard on purpose.

    The client corpus is not English-only, so a dimension that quietly degrades on
    Hinglish must show up here rather than be absorbed by a weaker Hinglish band.
    Measured: specificity +1.60 (0.91 -> 2.51) clears SCORE_GAP and travels well;
    actionability inverts. The two are not interchangeable across languages.
    """
    gap = answers[high][question]["score"] - answers[low][question]["score"]
    assert gap >= SCORE_GAP, f"{question} (hinglish) gap {gap:+.2f}"


@pytest.mark.parametrize(
    ("question", "false_post", "true_post"),
    [
        ("personal_stakes", "stakes_low", "stakes_high"),
        ("takes_a_position", "hook_gratitude", "hook_contrarian"),
    ],
)
def test_noul_separates_its_extremes(answers, question, false_post, true_post):
    """Also a gap band, because this checkpoint's noul outputs are conservative: the
    true-case posts read 0.318 and 0.548, so a flat `> 0.5` would fail a direction the
    model got right. Measured gaps: takes_a_position +0.485, personal_stakes +0.257.
    """
    gap = answers[true_post][question]["noul"] - answers[false_post][question]["noul"]
    assert gap >= NOUL_GAP, f"{question} gap {gap:+.3f}"


def test_every_answer_is_in_range(answers):
    """Every column that reaches the regression must be finite and in its declared range."""
    for name, per_question in answers.items():
        for question, answer in per_question.items():
            assert 0.0 <= answer["confidence"] <= 1.0, (name, question)
            if answer["type"] == "score":
                assert 0.0 <= answer["score"] <= 3.0, (name, question)
            elif answer["type"] == "noul":
                assert 0.0 <= answer["noul"] <= 1.0, (name, question)
            else:
                assert answer["choice"] in answer["probabilities"], (name, question)


def test_format_column_is_not_constant(answers):
    """A one-hot column with one label carries no information into the regression.

    Measured across the corpus: hot_take 11, story 8, case_study 2, announcement 1 --
    4 of 6 labels win at least once. This is a canary for the column collapsing, not a
    claim that `format` is well calibrated.
    """
    labels = {a["format"]["choice"] for a in answers.values()}
    assert len(labels) >= 3, f"format column has collapsed: {labels}"
