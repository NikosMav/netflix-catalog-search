# Eval v2 relevance rubric

Judge: LLM judge (grok-4.7 via Cursor cloud agent), blinded to system and rank.

Each decision uses only the query text and the candidate card in the blinded file: title, type, year, genres (`listed_in`), cast, director, and description. Do not use retrieval scores, system names, ranks, or the v1 relevance file.

Labels are binary. `relevant` means the card satisfies the information need. The existing eval code expects a set of relevant show ids and treats every other judged id as not relevant.

## Mark relevant

- The card shows that the title is about the requested topic, person, place, era, or plot.
- When the query states a format, the card meets it: movie versus series, documentary versus fiction, stand-up special versus a narrative film the person acts in.
- For a named work, the work itself is relevant. Direct sequels or further seasons are relevant when the card identifies them as that same work.
- For a named person plus a role or genre, the card shows that person in the cast, director, title, or description, and the rest of the card fits the role or genre.

## Mark not relevant

- The overlap is only a shared word (war, mouse, dark, office, love) and the card is about something else.
- A person query does not show that person in the cast, director, title, or description.
- The format is wrong. A talk show is not a stand-up special. A feature is not a series when the query asks for a series. A stand-up special is not a narrative comedy movie.
- A making-of, interview, or clip show is not relevant unless the query asks for material about the production, or the card shows that it is the work itself.
- A country, era, or city constraint fails when the query makes that constraint central and the card shows a different setting.
- The card is too vague to tell. Uncertainty is not relevance.

## Rationale

Write one sentence. Name the card fact that decided the label. Do not mention a retriever, a rank, or a score.
