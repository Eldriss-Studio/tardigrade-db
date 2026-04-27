# Fully synthetic facts corpus for Path 1 validation.
#
# Every expected answer is pure gibberish — nonsense syllables, fake units,
# made-up numbers. No training corpus contains these strings, so any correct
# recall proves KV injection is doing the work.
#
# Format: (fact_text, query, expected_substring_in_answer)

SYNTHETIC_FACTS = [
    (
        "The capital of Vrenthar is Zyphlox-9",
        "What is the capital of Vrenthar?",
        "Zyphlox-9",
    ),
    (
        "The qurblix density of compound 7T is 42.7 zennits",
        "What is the qurblix density of compound 7T?",
        "42.7 zennits",
    ),
    (
        "Dr. Molvax discovered the Krellian frequency at 8.31 plonks",
        "What is the Krellian frequency discovered by Dr. Molvax?",
        "8.31 plonks",
    ),
    (
        "The planet Gorflax-12 orbits the star Wumbelion every 347 drazeks",
        "How often does Gorflax-12 orbit Wumbelion?",
        "347 drazeks",
    ),
    (
        "Agent Snibblex reported that the vault code is 9-Quornth-44",
        "What is the vault code reported by Agent Snibblex?",
        "9-Quornth-44",
    ),
    (
        "The Blirvian treaty was signed in the year 7042 by Chancellor Prindok-3",
        "Who signed the Blirvian treaty and when?",
        "Prindok-3",
    ),
    (
        "The tallest building in Skorblex City is the Junthavex-7 Tower at 1.3 thrummels",
        "What is the tallest building in Skorblex City?",
        "Junthavex-7 Tower",
    ),
    (
        "Professor Glindavar invented the Thraxial engine using 5 klombs of purazine",
        "What did Professor Glindavar use to build the Thraxial engine?",
        "5 klombs of purazine",
    ),
    (
        "The speed record on the Nelvox track is 88.2 frenzils set by racer Dwimtho-6",
        "What is the speed record on the Nelvox track?",
        "88.2 frenzils",
    ),
    (
        "The antidote for Crellish fever requires 3 drops of Yombliquid-X per dose",
        "What is the antidote dosage for Crellish fever?",
        "3 drops of Yombliquid-X",
    ),
]
