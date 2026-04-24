# Multi-memory novel facts corpus.
#
# Each entry has 2-3 facts that must be combined to answer the query.
# No single fact contains the full answer — the model must synthesize
# information across injected memories.
#
# Designed for multi-memory KV injection experiments (Phase 30).

MULTI_FACTS = [
    {
        "facts": [
            "Sonia's wifi password is mango-cathedral-7",
            "Sonia lives in apartment 4B on the third floor",
        ],
        "query": "What is the wifi password for the person in apartment 4B?",
        "expected": "mango-cathedral-7",
    },
    {
        "facts": [
            "Lucia's swimming instructor is named Tomoko",
            "Tomoko drives a red Honda Civic",
        ],
        "query": "What kind of car does Lucia's swimming instructor drive?",
        "expected": "Honda Civic",
    },
    {
        "facts": [
            "Mrs. Kowalski from apartment 3B brought over pierogi last Tuesday",
            "Mrs. Kowalski's cat is named Whiskers",
        ],
        "query": "What is the name of the cat belonging to the neighbor who brought pierogi?",
        "expected": "Whiskers",
    },
    {
        "facts": [
            "Eduardo works at a bakery called Morning Bloom on Oak Street",
            "Morning Bloom's best-selling item is their cinnamon swirl bread",
        ],
        "query": "What is the best-selling item at the bakery where Eduardo works?",
        "expected": "cinnamon swirl",
    },
    {
        "facts": [
            "Dr. Patel is Sonia's dentist and works at Riverside Dental",
            "Riverside Dental is located at 742 Elm Avenue",
        ],
        "query": "What is the address of Sonia's dentist office?",
        "expected": "742 Elm",
    },
    {
        "facts": [
            "Lucia's piano teacher is Mr. Yamamoto",
            "Mr. Yamamoto has a golden retriever named Biscuit",
            "Mr. Yamamoto teaches on Wednesday afternoons",
        ],
        "query": "What is the name of the dog owned by Lucia's piano teacher?",
        "expected": "Biscuit",
    },
    {
        "facts": [
            "Sonia's running group meets at Prospect Park every Saturday",
            "The running group leader is named Carmen",
        ],
        "query": "Who leads the running group that meets at Prospect Park?",
        "expected": "Carmen",
    },
    {
        "facts": [
            "Eduardo's favorite restaurant is Trattoria Bella on 5th Street",
            "Trattoria Bella is closed on Mondays",
        ],
        "query": "On what day is Eduardo's favorite restaurant closed?",
        "expected": "Monday",
    },
    {
        "facts": [
            "The pharmacy Sonia uses is MedPlus on Western Avenue",
            "MedPlus pharmacist is named James Chen",
        ],
        "query": "What is the name of the pharmacist at the pharmacy on Western Avenue?",
        "expected": "James Chen",
    },
    {
        "facts": [
            "Lucia's best friend at school is named Harper",
            "Harper's birthday is on March 15th",
        ],
        "query": "When is the birthday of Lucia's best friend?",
        "expected": "March 15",
    },
]
