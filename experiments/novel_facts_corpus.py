# Novel facts corpus for injection vs text RAG testing.
#
# Each fact is something NO model could know from training data.
# Tests whether KV injection actually transfers specific knowledge.
#
# Format: (memory_text, query, expected_substring_in_answer)

NOVEL_FACTS = [
    (
        "Sonia's wifi password is mango-cathedral-7",
        "What is Sonia's wifi password?",
        "mango-cathedral-7",
    ),
    (
        "Lucia's favorite dinosaur is the Pachycephalosaurus",
        "What is Lucia's favorite dinosaur?",
        "Pachycephalosaurus",
    ),
    (
        "The pharmacy on Western Avenue closes at 8:30pm on Tuesdays",
        "When does the pharmacy on Western Avenue close on Tuesdays?",
        "8:30",
    ),
    (
        "Eduardo's new apartment number is 4B on the third floor",
        "What is Eduardo's apartment number?",
        "4B",
    ),
    (
        "Lucia's swimming instructor at the community center is named Tomoko",
        "What is the name of Lucia's swimming instructor?",
        "Tomoko",
    ),
    (
        "The sourdough starter that died was named Fernando and it lasted six days",
        "What was the name of the sourdough starter and how long did it last?",
        "Fernando",
    ),
    (
        "Mrs. Kowalski from apartment 3B brought over pierogi",
        "What did Mrs. Kowalski bring over and what apartment is she in?",
        "pierogi",
    ),
    (
        "The check engine light repair for the catalytic converter cost fourteen hundred dollars",
        "How much did the catalytic converter repair cost?",
        "fourteen hundred",
    ),
    (
        "Sonia ran a 5K charity race in Grant Park and finished in thirty-one minutes",
        "How long did it take Sonia to finish the 5K in Grant Park?",
        "thirty-one",
    ),
    (
        "The book club meets at the library on the first Thursday of every month and they read Pachinko",
        "When does the book club meet and what did they read?",
        "Pachinko",
    ),
]
