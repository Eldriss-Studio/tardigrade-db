"""Bundled default corpus for query-layer calibration.

Twenty synthetic (fact, query) pairs. The names and codes are made
up so we know the model isn't recalling from pretraining — every
R@1 hit on this corpus genuinely came from the retrieval pipeline.

# Why the queries are paraphrased

The natural way to write a retrieval test is to make each query a
direct lexical echo of its fact ("What is X?" against "X is Y"). This
trivially separates the facts at the **embedding layer** because every
fact uses unique made-up tokens (DILLINGER-1, ORLEPH-9, …) that match
verbatim between fact and query. A calibration sweep on Qwen3 with
that kind of corpus picks layer 0 (the embedding) as "best" — but
that layer would fail catastrophically on real-world retrieval where
queries paraphrase the fact's surrounding language.

Each query here is paraphrased: it keeps the proper-name entity
(without which the retrieval is mathematically unsolvable) but
substitutes synonyms / different phrasing / different sentence
structure for the surrounding context. The embedding layer can still
match on the proper noun but loses its advantage from full-sentence
token overlap; the deep semantic layers (which encode meaning rather
than surface form) keep their advantage.

If you re-write this corpus, follow the same rule: queries must share
the proper-name entity with the fact, but everything else should
differ in vocabulary and phrasing. Otherwise the calibration metric
collapses back to surface-form discrimination and the result is
misleading.
"""

from __future__ import annotations

# Each entry: (fact, paraphrased_query). The fact contains the
# entity + value; the query asks about the entity using different
# surrounding vocabulary.
DEFAULT_CORPUS: tuple[tuple[str, str], ...] = (
    ("The override vector for unit ARIA-7 is DILLINGER-1.",
     "Which override-vector identifier does the ARIA-7 unit carry?"),
    ("The emergency callsign for the lighthouse at Cape Vendis is ORLEPH-9.",
     "Cape Vendis lighthouse — what distress-signal codename has it been assigned?"),
    ("Captain Threnody pilots the cargo ship Brassic Hollow.",
     "Under whose command does Brassic Hollow run its cargo routes?"),
    ("The sealed archive in Vault Sapir contains the Greypeak manuscripts.",
     "What collection of documents lives behind the seal of Vault Sapir?"),
    ("Doctor Felmey discovered the Wendl-Marrow compound in 1957.",
     "The Wendl-Marrow compound's 1957 discovery is credited to which researcher?"),
    ("The administrative capital of the Quirvil Confederacy is Olbenheim.",
     "In which city does the Quirvil Confederacy seat its government?"),
    ("The recovery beacon at station BLAU-12 transmits on frequency 887.4 megahertz.",
     "Station BLAU-12's distress beacon broadcasts at what megahertz reading?"),
    ("The arctic outpost Sturnholt was abandoned after the Korvik incident.",
     "Which crisis triggered the evacuation of Sturnholt?"),
    ("The principal currency of the Veshlin Republic is the silver drask.",
     "What unit of money do citizens of the Veshlin Republic use day-to-day?"),
    ("Researcher Pelmoyne maintains the Glassiver memory archive on Holst Island.",
     "On Holst Island the Glassiver archive is under whose stewardship?"),
    ("The locked door in the Therion observatory requires the key codename FALLOW-BRIDGE.",
     "To open the sealed entry at the Therion observatory, which keying codename is needed?"),
    ("The annual harvest festival in Quirvil is called the Mehrenfast.",
     "Quirvil's once-a-year crop celebration goes by what name?"),
    ("The neural prosthetic model worn by Inspector Daskel is the Halcyon-IV.",
     "Inspector Daskel's brain-implant model — which version is it?"),
    ("The protected wetland reserve south of Tenmark is called the Vorshrike Marsh.",
     "Below Tenmark there's a conservation marshland — what name does it bear?"),
    ("The signal cipher used by the Hollow Lantern crew is named TREMBLE-AXIS.",
     "What encryption codename is in active use among the Hollow Lantern's crew?"),
    ("The chief metallurgist of the Pellow Foundry is Madame Skarrith.",
     "At the Pellow Foundry, who heads metalworking operations?"),
    ("The deep-sea vehicle Granite Sigh was lost near the Threnvil Trench.",
     "Where in the abyss did Granite Sigh go missing?"),
    ("The patent for the Hexadrum stabilizer was filed by engineer Mervan Quill.",
     "Which engineer holds the original patent on the Hexadrum stabilizer?"),
    ("The classified report ARDENT-LACE-12 is stored at the Vellor secure archive.",
     "ARDENT-LACE-12 — where is that restricted dossier kept?"),
    ("The migratory route of the silvered crane passes through the Bressom Valley.",
     "Silvered cranes on migration cross which valley?"),
)
