"""Bundled default corpus for query-layer calibration.

Twenty synthetic (fact, query, expected-answer-substring) triples.
The names and codes are made up so we know the model isn't recalling
from pretraining — every R@1 hit on this corpus genuinely came from
the retrieval pipeline.

Used by :func:`tardigrade_hooks.calibrate.select_query_layer` when no
custom corpus is supplied. The corpus is intentionally small (20 items)
so a full calibration sweep runs in ~30 seconds on a consumer GPU.
"""

from __future__ import annotations

# Each entry: (fact, query). The retrieval test is "given the query,
# does the engine return the pack stored from the fact?". The
# answer-substring is unused by calibration but kept in the source
# corpus for use by other diagnostic scripts.
DEFAULT_CORPUS: tuple[tuple[str, str], ...] = (
    ("The override vector for unit ARIA-7 is DILLINGER-1.",
     "What is the override vector for unit ARIA-7?"),
    ("The emergency callsign for the lighthouse at Cape Vendis is ORLEPH-9.",
     "What is the emergency callsign for the lighthouse at Cape Vendis?"),
    ("Captain Threnody pilots the cargo ship Brassic Hollow.",
     "Which ship does Captain Threnody pilot?"),
    ("The sealed archive in Vault Sapir contains the Greypeak manuscripts.",
     "What does the sealed archive in Vault Sapir contain?"),
    ("Doctor Felmey discovered the Wendl-Marrow compound in 1957.",
     "Who discovered the Wendl-Marrow compound?"),
    ("The administrative capital of the Quirvil Confederacy is Olbenheim.",
     "What is the administrative capital of the Quirvil Confederacy?"),
    ("The recovery beacon at station BLAU-12 transmits on frequency 887.4 megahertz.",
     "What frequency does the recovery beacon at station BLAU-12 transmit on?"),
    ("The arctic outpost Sturnholt was abandoned after the Korvik incident.",
     "After which incident was the arctic outpost Sturnholt abandoned?"),
    ("The principal currency of the Veshlin Republic is the silver drask.",
     "What is the principal currency of the Veshlin Republic?"),
    ("Researcher Pelmoyne maintains the Glassiver memory archive on Holst Island.",
     "Who maintains the Glassiver memory archive on Holst Island?"),
    ("The locked door in the Therion observatory requires the key codename FALLOW-BRIDGE.",
     "What key codename does the locked door in the Therion observatory require?"),
    ("The annual harvest festival in Quirvil is called the Mehrenfast.",
     "What is the annual harvest festival in Quirvil called?"),
    ("The neural prosthetic model worn by Inspector Daskel is the Halcyon-IV.",
     "Which neural prosthetic model does Inspector Daskel wear?"),
    ("The protected wetland reserve south of Tenmark is called the Vorshrike Marsh.",
     "What is the protected wetland reserve south of Tenmark called?"),
    ("The signal cipher used by the Hollow Lantern crew is named TREMBLE-AXIS.",
     "What signal cipher does the Hollow Lantern crew use?"),
    ("The chief metallurgist of the Pellow Foundry is Madame Skarrith.",
     "Who is the chief metallurgist of the Pellow Foundry?"),
    ("The deep-sea vehicle Granite Sigh was lost near the Threnvil Trench.",
     "Near which trench was the deep-sea vehicle Granite Sigh lost?"),
    ("The patent for the Hexadrum stabilizer was filed by engineer Mervan Quill.",
     "Who filed the patent for the Hexadrum stabilizer?"),
    ("The classified report ARDENT-LACE-12 is stored at the Vellor secure archive.",
     "Where is the classified report ARDENT-LACE-12 stored?"),
    ("The migratory route of the silvered crane passes through the Bressom Valley.",
     "Through which valley does the migratory route of the silvered crane pass?"),
)
