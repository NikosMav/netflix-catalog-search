# Eval v2 relevance rubric

Judge: LLM judge (grok-4.7 via Cursor cloud agent), blinded to system and rank.

Each decision uses only the query text and the candidate card in the blinded file: title, type, year, genres (`listed_in`), cast, director, and description. Do not use retrieval scores, system names, ranks, the v1 relevance file, or any previous binary label.

Grades are integers from 0 to 3. Write one sentence of rationale. Name the card fact that decided the grade. Do not mention a retriever, a rank, or a score.

## Scale

| Grade | Name | Meaning |
| --- | --- | --- |
| 0 | Not relevant | The card is a different subject, or the overlap is only a shared word, or the card is too vague to tell. |
| 1 | Marginal | The card is about a neighboring subject, but a central constraint is missing or contradicted. |
| 2 | Relevant | The card satisfies the information need. Format, person, place, and era constraints that the query makes central are met by the card. |
| 3 | Highly relevant | The card is the exact intent: the named work, or every central constraint is explicit on the card. |

nDCG uses gains `2^grade - 1` (0, 1, 3, 7). Recall and MRR count a title as relevant only when the grade is 2 or 3. That metric amendment is in `configs/eval_v2.yaml`.

## How to assign the grade

- Grade 3 when the card identifies the requested work, or states the requested topic together with every central constraint (format, person, place, era, plot).
- Grade 2 when the card meets the need in substance. A comedy series can satisfy "sitcom" when the card says it is a TV comedy about that premise. Do not require a synonym the card does not need.
- Grade 1 when the topic is related but a central constraint fails: wrong format, wrong era or country, the person is present in a different role, or the card is about the aftermath rather than the event the query asks for.
- Grade 0 when the card is about something else, shares only a token such as war, mouse, dark, office, or love, or is too vague. Uncertainty is grade 0, not grade 1.
- When the query states a format, grades 2 and 3 require the card to meet it: movie versus series, documentary versus fiction, stand-up special versus a narrative film. A related title in the wrong format is grade 1.
- A making-of, interview, or clip show is grade 0 or 1 unless the query asks for material about the production. It is not grade 2.
- Do not import facts the card does not show. A famous title is not grade 3, or even grade 2, for a constraint the card never states.

## Examples

Query: "stand-up special by Priya Raman".

- Grade 0. Card: a baking competition series. Priya Raman is not in the title, cast, director, or description.
- Grade 1. Card: a narrative comedy movie. The cast includes Priya Raman, and the description is a scripted heist, not stand-up.
- Grade 2. Card: a stand-up special. The description says Priya Raman appears with three other comics.
- Grade 3. Card: title "Priya Raman: Live in Chicago", type stand-up comedy, description of her solo special.

Query: "documentary about the first moon landing".

- Grade 0. Card: a romantic comedy about a wedding planner.
- Grade 1. Card: a drama about an astronaut's family after a training accident. No landing is described.
- Grade 2. Card: a documentary series about the Apollo program that includes the first landing among later missions.
- Grade 3. Card: a documentary whose description is the Apollo 11 mission and the first moon landing.

## Rationale

One sentence. Cite the card fact that set the grade.
