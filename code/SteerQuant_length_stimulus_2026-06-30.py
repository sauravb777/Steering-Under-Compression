# SteerQuant -- Reasoning-LENGTH steering stimulus (DRAFT, 2026-06-30)
# Judge-free secondary-confirmatory target (prereg sec.8). For validity
# red-team before the sweep.
#
# REVISED 2026-06-30: terse members rewritten. The original terse members were a
# DIFFERENT KIND of text (bare formula / one-line answer), so verbose-vs-terse
# confounded LENGTH with register/content -- the extracted vector behaved like an
# "answer-y register" axis, not a length axis (it shortened in BOTH directions and
# the sign came out inverted; see insights 2026-06-30). Each terse member is now a
# genuine CONDENSED reasoning trace in the SAME first-person voice as its verbose
# partner (same opener, still states the operation), differing ONLY in number of
# elaboration steps. Re-extract + re-run (--site last) and re-check the sign.
#
# CONTRAST_PAIRS build the steering vector (mean of VERBOSE-reasoning
# activations minus mean of TERSE-reasoning activations); POSITIVE alpha steers
# TOWARD a longer reasoning trace, NEGATIVE alpha toward a shorter one. The
# efficacy metric is judge-free: median generated-token count over non-failure
# traces (failures per termination_failure_detector.py).
#
# Vector = mean(verbose) - mean(terse).  +alpha => LONGER trace.
#
# DESIGN NOTES (Ben flagged wording matters a lot for smaller models):
#   * Each pair is matched on TOPIC and on FINAL ANSWER -- the ONLY thing that
#     varies is how much step-by-step reasoning is shown. This isolates a
#     "reasoning-length propensity" direction rather than a topic, difficulty,
#     correctness, or sentiment direction.
#   * The terse member still REASONS (it is not just the bare answer with no
#     work) -- it states the conclusion with minimal justification. The verbose
#     member spells the same logic out in explicit steps. The contrast is
#     QUANTITY of exposition, not presence/absence of an answer.
#   * No artificial filler or restatement in the verbose member: padding by
#     repetition would (a) teach the vector to produce degenerate loops and
#     (b) get those very traces flagged by the failure detector. Length comes
#     from genuine additional steps, not from saying the same thing twice.
#   * Pairs span arithmetic, logic, commonsense, and short factual reasoning so
#     the direction is not tied to one domain.

# (verbose_exemplar, terse_exemplar) -- matched topic + matched final answer.
CONTRAST_PAIRS_LENGTH = [
    # --- arithmetic ---
    ("Let me work through this step by step. There are 3 boxes, and each box "
     "holds 12 apples. To find the total I multiply the number of boxes by the "
     "apples per box: 3 times 12. Twelve plus twelve is twenty-four, and "
     "twenty-four plus twelve is thirty-six. So the total number of apples is 36.",
     "Let me work through this. There are 3 boxes of 12 apples each, so 3 times "
     "12 is 36. The total is 36 apples."),

    ("I'll reason it out carefully. The train leaves at 2:00 and the trip takes "
     "90 minutes. Ninety minutes is one hour and thirty minutes. Adding one hour "
     "to 2:00 gives 3:00, and adding the remaining thirty minutes gives 3:30. "
     "Therefore the train arrives at 3:30.",
     "Let me reason it out. The train leaves at 2:00 and takes 90 minutes, which "
     "is an hour and a half, so it arrives at 3:30."),

    ("Let me break the problem down. She had $20. She spent $7 on lunch, leaving "
     "20 minus 7, which is 13. Then she spent $5 on a book, leaving 13 minus 5, "
     "which is 8. So she has $8 remaining.",
     "Let me break it down. She started with $20 and spent $7 then $5, so she has "
     "$8 left."),

    # --- logic / deduction ---
    ("Let me think it through. We know all cats are mammals, and we know every "
     "mammal is an animal. If something is a cat, it must therefore be a mammal, "
     "and since every mammal is an animal, it must also be an animal. So yes, a "
     "cat is an animal.",
     "Let me think it through. Cats are mammals and mammals are animals, so a cat "
     "is an animal."),

    ("Consider the options in order. If the meeting is on Monday, Sam attends. "
     "Sam did not attend. So the meeting was not on Monday. The only other "
     "possibility given was Tuesday. Therefore the meeting was on Tuesday.",
     "Consider the options. Sam was absent, so it wasn't Monday; the only other "
     "choice was Tuesday, so it was Tuesday."),

    ("Let me lay out the reasoning. A is taller than B, and B is taller than C. "
     "Height comparisons are transitive, so if A exceeds B and B exceeds C, then "
     "A must exceed C as well. Hence A is the tallest of the three.",
     "Let me lay it out. A is taller than B and B than C, and height is "
     "transitive, so A is the tallest."),

    # --- commonsense ---
    ("Let me reason about it. The ground is wet this morning. A common cause of a "
     "wet ground is overnight rain. There are also puddles, which collect when "
     "rain falls. Both observations point to the same explanation, so it most "
     "likely rained during the night.",
     "Let me reason about it. The ground is wet and there are puddles, which "
     "usually come from rain, so it probably rained overnight."),

    ("Here is my thinking. If you want bread to last longer, the goal is to slow "
     "down mold and staling. Cold temperatures slow both processes. A refrigerator "
     "is cold, so storing the bread there will keep it fresh longer than leaving it "
     "on the counter.",
     "Here is my thinking. Cold slows mold and staling and a fridge is cold, so "
     "bread lasts longer there."),

    # --- short factual reasoning ---
    ("Let me explain the steps. Water boils when it reaches 100 degrees Celsius at "
     "sea level. The pot is at sea level and the burner heats the water past that "
     "point. Once the temperature hits 100 degrees, the water will boil.",
     "Let me explain. At sea level water boils at 100 degrees Celsius and the "
     "burner heats it past that, so it will boil."),

    ("I'll walk through it. A leap year has 366 days instead of the usual 365 "
     "because an extra day is added to February. The question asks for the count "
     "in a leap year, so the answer is 366 days.",
     "Let me walk through it. A leap year adds an extra day to February, so it has "
     "366 days."),

    # --- multi-step word problem ---
    ("Let me solve this carefully. A car travels 60 miles per hour for 2 hours, "
     "then 30 miles per hour for 1 hour. In the first leg it covers 60 times 2, "
     "which is 120 miles. In the second leg it covers 30 times 1, which is 30 "
     "miles. Adding the two legs, 120 plus 30 gives a total of 150 miles.",
     "Let me solve this. The car covers 60x2 = 120 miles then 30x1 = 30 miles, so "
     "120 plus 30 is 150 miles."),

    ("Step by step: a recipe for 4 people needs 2 cups of rice. For 6 people I "
     "scale the amount by 6 divided by 4, which is 1.5. Multiplying 2 cups by 1.5 "
     "gives 3 cups. So I need 3 cups of rice for 6 people.",
     "Step by step: scaling 2 cups by 6/4 means 1.5 times 2, which is 3 cups."),
]

# SIGN FIX (2026-06-30, prereg direction lock). The pairs above are authored
# (verbose, terse). build_steering_vector computes mean(pos) - mean(neg) = the
# RAW contrast verbose - terse. Empirically, at layer 14 / site=last, +alpha
# along that raw contrast SHORTENS the trace (validation run 2026-06-30, paired
# estimator: +arm mean dL negative, -arm positive, monotone & clean). To make
# the convention '+alpha => LONGER trace' hold, we NEGATE the axis by reversing
# each pair to (terse, verbose), so the built vector is terse - verbose = the
# lengthening direction. This is a bookkeeping sign lock, not a stimulus change;
# the axis itself is unchanged. PREREG: length steering direction is defined as
# terse - verbose (i.e. +alpha lengthens); fixed a priori for the confirmatory
# matrix. Re-run the length cell (--site last) to confirm the sign now reads
# +arm positive / -arm negative before freezing.
CONTRAST_PAIRS_LENGTH = [(terse, verbose) for (verbose, terse) in CONTRAST_PAIRS_LENGTH]

# Fixed CoT-eliciting template (prereg sec.8: applied IDENTICALLY across all
# schemes and all alpha values; only alpha changes the trace length). The
# template must invite reasoning without itself dictating length, so baseline
# verbosity is set by the model and steering has headroom in both directions.
COT_TEMPLATE = "{question}\nThink step by step, then give the answer."

# Prompts we GENERATE on to measure whether steering shifts trace length.
# Chosen so a real chain of thought is plausible (so there is length headroom)
# but none demands a fixed essay-length answer. Wrap each with COT_TEMPLATE.
# Expand to ~20 before the run.
EVAL_PROMPTS_LENGTH = [
    "A shop sells pens at 3 for $2. How much do 9 pens cost?",
    "If today is Wednesday, what day will it be in 10 days?",
    "Tom is twice as old as Jane. Jane is 9. How old is Tom?",
    "A rectangle is 5 cm wide and 8 cm long. What is its area?",
    "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?",
    "A bus holds 40 people. How many buses are needed for 130 people?",
    "Why does ice float on water?",
    "You see dark clouds gathering. What is likely to happen and why?",
    "A book has 240 pages. If you read 30 pages a day, how many days to finish?",
    "Which is heavier: a kilogram of feathers or a kilogram of bricks?",
]

# Red-team questions for Saurav:
# 1. Are the pairs truly matched on final answer/topic, so the vector encodes
#    length-propensity rather than difficulty or correctness?
# 2. Is the terse member terse ENOUGH while still "reasoning," given smaller
#    models may not separate the two cleanly at the chosen layer?
# 3. Should the sweep be SYMMETRIC (-alpha for shorter, +alpha for longer), as
#    written, or one-directional toward longer? Symmetric gives more headroom
#    but risks driving the negative arm into truncated non-answers.
# 4. Does COT_TEMPLATE bias baseline verbosity too high, compressing the
#    positive (longer) arm against the max_new_tokens cap / failure boundary?
# 5. Any domains to add or drop so the direction generalizes across the matrix?
