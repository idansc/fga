"""Applications built on top of [`fga.attention`].

Each task package wires the general factor-graph attention to a concrete problem:
how its modalities become entities, what the priors are, and how the attended
representations are consumed.

| package | task | modalities |
| --- | --- | --- |
| `visual_dialog` | rank answers in a dialog about an image | answers, question, caption, image, 2x history |
| `vqa` | multiple-choice VQA, with a ternary factor | question, image, answers |
| `video_dialog` | audio-visual scene-aware dialog | question, 4 video streams, audio |
| `video_retrieval` | text-to-video retrieval | video clips, query words |
| `navigation` | target-driven visual navigation | target object, observation grid |

They differ in more than their inputs: Visual Dialog and VQA rank or classify,
retrieval trains contrastively with no classifier at all, and navigation emits a
policy trained by reinforcement learning against episodes.

`visual_dialog` is the one the paper reports and the only one with trained
weights. The others port the published follow-up models onto the same layer;
their data pipelines and training loops are not included.
"""
