The popular RLVR algorithm verifies the answer by rules and rubrics. We want to give agent a hint/feedback about the issues in its submitted answer besides a numerical score. We use `longform_rubric_rewards.py` in our RL training to score longform deep research articles. The score is a weighted sum of search score, citation score, rubric socre and format score. The search score and format score are purely rule-based while citation score and rubric score relies on judge models to validate the answer.

- rubric_utils.py: give weighted sum of rubric scores based on rubrics.
- citation_utils.py: give citation precision and recall scores.

You need to write a `verify_longform_answer_apis.py` for the scores and feedback. The calculation of scores should be align with `longform_rubric_rewards.py`, that is, the range of score and the weight of specific scores should be same. And for each rubric score, there should be a issues and feedback based on the rubric. For citations, we should report the citation recall([Fully supported] / [Partially supported] / [No support]) and precision ([Relevant] / [Unrelevant])/


When the `verify_answer` is called by agent, the question, the answer, context and solution path is preprocessed and extracted from full tajectories like this:

```
<question>
The impact of organizational silence on employees\' performance in the hospitality industry.
</question>

<solution_path>
<call_tool name="snippet_search" limit="10" year="2010-2025" fieldsOfStudy="Business,Economics,Psychology">
organizational silence hospitality industry performance service recovery trust
</call_tool>
<snippet id=4d006072>
Client failed to connect: All connection attempts failed
</snippet>
<call_tool name="snippet_search" limit="10" year="2010-2025" fieldsOfStudy="Business,Economics,Psychology">
organizational silence climate hospitality tourism service recovery trust
</call_tool>
<snippet id=3f463a62>
Client failed to connect: All connection attempts failed
</snippet>

</solution_path>
<context>
<snippet id=4d006072>
Client failed to connect: All connection attempts failed
</snippet>
<snippet id=3f463a62>
Client failed to connect: All connection attempts failed
</snippet>
</context>
<ground_truth>
 [{"description": "Define organizational silence in the hospitality context; describe at least two primary forms (e.g., defensive and resignation silence); explicitly link these to mechanisms relevant to employee performance such as knowledge sharing, communication, or openness.", "title": "Definition and Forms of Silence", "weight": 3}, {"description": "Demonstrate\\u2014using hospitality-specific and general evidence\\u2014that organizational silence impairs employee creative/innovative performance; specify the mediating mechanisms (e.g., reduced knowledge sharing, blocked communication, lower engagement) and cite empirical or modeled relationships.", "title": "Performance Impact & Mechanisms", "weight": 3}, {"description": "Present hospitality-specific empirical findings that establish organizational silence (or employee silence) as a mediator between negative leadership styles (e.g., narcissistic/abusive) and detrimental employee outcomes such as cynicism, and generalize the implications for service performance in hotels.", "title": "Hospitality Empirics\\u2014Silence as Mediator", "weight": 3}, {"description": "Discuss the moderating role of leadership style and organizational climate (e.g., transformational/supportive leadership versus abusive/authoritarian) in shaping both the prevalence of silence and its impact on performance outcomes in the hospitality industry; support with comparative or complementary hospitality findings.", "title": "Leadership & Climate Moderators", "weight": 3}, {"description": "State the practical implications for hospitality managers, including at least two actionable strategies anchored in cited evidence for reducing antecedents of silence and building climates that promote voice and performance.", "title": "Practical Implications for Managers", "weight": 3}], "rubrics_types": ["persistent", "persistent", "persistent", "persistent", "persistent"]
</<ground_truth>


<answer>
# Summary of the General Public's Views on Wikidata

Wikidata's central value — structured, machine-readable "structured data" for powering open knowledge services—is less widely recognized among the general public, which often remains unaware of what Wikidata is and where it fits relative to well-known projects like Wikipedia.<cite id="S_8GV4k6D">Surveys and event-recall studies consistently place Wikidata near the bottom of awareness rankings among large projects; it is widely known that Wikipedia exists, but the same is not true for Wikidata.</cite> When people do encounter Wikidata, common perceptions include difficulty in use and a preference for simpler, question-answering styles of interaction.<cite id="S_H2ScdKR">Studies of first-time users report that Wikidata feels "hard to use," the interface can be "confusing" or "frustrating," and users may be unsure of their goals when interacting with Wikidata.</cite>

Perceived usefulness and desirability are often limited and context-dependent. Many end users see Wikidata as useful for factual information drawn from Wikimedia knowledge projects, but they value approaches that are ready-to-use and do not require editing or familiarity with data concepts.<cite id="S_9tMdN1W">Users commonly prefer interfaces and tools that present answers directly or simplify the search for facts rather than requiring manual data editing or deep schema understanding.</cite> This creates a gap between Wikidata's technical goals and everyday expectations for convenience and immediacy.<cite id="S_5nXdguQ">Experts and designers emphasize structured data quality and reuse, whereas lay users more often expect straightforward, trusted answers without setup.</cite>

Public-facing initiatives have sought to improve recognition, accessibility, and usefulness among non-experts. Programs like Games of the Small Nations formed local community teams to develop Wikidata content, improving both coverage of smaller countries and local pride and capacity.<cite id="S_oURg7bS">Fieldwork reports show that GOTSN events increased local recognition of Wikidata as a relevant contributor to knowledge visibility, while also building social networks around data creation.</cite> The Knowledge Cards project aimed to present accessible, printed summaries of concepts to diverse audiences and spawned the "Concept Hopefulness" approach that outlines simple belief pathways for data producers to connect concepts through shared properties, making relationships more discoverable to non-experts.<cite id="S_bnq48jb">Knowledge Cards research documents that audiences may find printed, structured concept overviews useful as prompts, and the "Concept Hopefulness" framework helps editors anticipate where lay users might encounter bottlenecks when following simple, logic-based paths connecting related items.</cite> Early design studies of "Ask Wikidata," a QnA prototype, found that people generally prefer guided answer-suggestion interfaces over direct editor tools, pointing to a design lever for broader uptake.<cite id="S_qnBpmam">User tests indicate that guided suggestion designs are perceived as easier and more reliable than free-form editor interfaces, while open-ended designs are seen as difficult and disorienting.</cite>
</answer>

```

The search context is wrapped in <contex>...</contex>. The rubrics are wrapped in <ground_truth></ground_truth>. The solution path is wrapped in <solution_path></solution_path> which contains the agent search action and returns. The draft answer is wrapped in <answer></answer>

You should:

- Call rubric judge model for rubric scores and rubric issue feedback simoutanously.
- Call citation judge model for batch citation verifications and give a plain text about wether the citations support demonstrations and revelance.
- Rule-base check the format. Give a format score and a string for wether the format is obeyed (<answer></answer>tags found / not found).
- Rule-base check the search times and give the search reward. No feedback need.

Don't use the `verify_answer_local.py` from trainig side, the mcp api implementation should be isolated from training side.
