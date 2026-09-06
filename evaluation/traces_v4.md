# V4 Context Trace

以下为 Context Manager + BGE/FAISS + LangGraph 的完整轨迹。

## Trace 1

```text
------------------------------------------------------------------------
[Query] 根据本地论文，V-JEPA 2-AC 是如何使用 action 的？
[Graph] START
[Node] Planner
[Context Manager] Planner context built from Query + constraints only.
Raw Context Tokens: 42
Final Context Tokens: 42/3000
Plan:
1. 根据本地论文，V-JEPA 2-AC 是如何使用 action 的？ [local_search]
[State Update] plan=current structured plan
[Edge] Planner -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 82
Selected Evidence: 0/0
Final Context Tokens: 82/6000
[Node] Researcher
Iteration: 0
Plan step: 1/1
Tool Call: local_search
Arguments: {'query': '根据本地论文，V-JEPA 2-AC 是如何使用 action 的？', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 1845
Final Context Tokens: 1845
Compressed: False
Compression Ratio: 0.0%
Tool: local_search
Evidence records added: 5
Source: vjepa2.pdf | Page: 8
Source: vjepa2.pdf | Page: 18
Source: vjepa2.pdf | Page: 3
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Writer
[Node] Writer
[Context Manager] Writer context excludes ToolNode history and old drafts.
Selected Evidence: 5/5
Deduplicated Evidence Count: 0
Dropped Evidence Count: 0
Raw Context Tokens: 1831
Final Context Tokens: 1831/8000
Draft generated from Query + Plan + Evidence.
[State Update] draft=current answer draft
[Edge] Writer -> Verifier
[Node] Verifier
[Context Manager] Verifier context contains Query + Draft + selected Evidence only.
Selected Evidence: 5/5
Deduplicated Evidence Count: 0
Dropped Evidence Count: 0
Raw Context Tokens: 3615
Final Context Tokens: 3615/8000
Passed: True
Feedback: Draft addresses the query and has available evidence records.
[State Update] iteration=1, verification_passed=True
[Conditional Edge] Verifier -> END
[Graph] END
[Final Answer]
基于当前研究证据：
- tioned predictor on
top of the frozen V-JEPA 2 video encoder (Figure 2, right). We train our model on data from the Droid
dataset (Khazatsky et al., 2024) consisting of data from experiments with a table-top Franka Panda robot
arm collected through teleoperation. We refer to the resulting action-conditioned model as V-JEPA 2-AC,
and in Section 4 we show that V-JEPA 2-AC can be used within a model-predictive control planning loop to
plan actions in new environments.
8（来源：vjepa2.pdf 第 8 页）
- ments.
The remainder of this paper is organized as follows. Section 2 describes the V-JEPA 2 pretraining procedure,
including the key ingredients enabling scaling beyond the original V-JEPA recipe of Bardes et al. (2024).
Section 3 then introduces our approach to training a task-agnostic action-conditioned world model, V-JEPA 2AC, leveraging the pretrained V-JEPA 2 model. Section 4 demonstrates using V-JEPA 2-AC for robot control
via model-based planning. Because V-JEPA 2-AC models world dynamics in a learned representation space,
its capabilities fundamentally depend on the information captured in the V-JEPA 2 representation space, and
so we further explore the performance of V-JEPA 2 for video understanding in Section 5 and prediction tasks
in Section 6. Finally, in Section 7 we show that V-JEPA 2 can be aligned with a language model for video
question answering. Section 8 discusses related work, and we conclude in Section 9.
3（来源：vjepa2.pdf 第 3 页）
- stem waits for the last
commanded action to be completed before sending a new action to the controller) and experiment with both
blocking and non-blocking control for Octo, and report the best performance across the two options. When
planning with V-JEPA 2-AC and Cosmos, we constrain each sampled action to the L1-Ball of radius 0.075
centered at the origin, which corresponds to a maximum end-effector displacement of approximately 13 cm for
each individual action, since large actions are relatively out-of-distribution for the models.
4.2
Results
Single-goal reaching.
First, we evaluate on the task of single-goal reaching, which involves moving the
end-effector to a desired location in space based on a single goal image. This task measures for a basic
understanding of actions as well as a 3D spatial understanding of the scene (including depth) from the
monocular RGB camera.
Figure 8 shows the Euclidean distance between the end-effector and its goal position during robot execution
for three different single-goal reaching tasks. In all cases, the model is able to move the end-effector within
less than 4 cm of its goal position, and select actions that lead to a monotonic decrease in the error. This can
be seen as a form of visual servoing (Hill, 1979), wherein visual feedback from a camera is used to control
a robot’s motion. However, unlike classical approaches in visual servoing, V-JEPA 2-AC achieves this by
training on unlabeled, real-world video data.
In Figure 9, we visualize the V-JEPA 2-AC energy landscape from equation (5) for the ∆y reaching task as a
12（来源：vjepa2.pdf 第 12 页）
- outperforms the previous state-of-the-art model PlausiVL by a significant margin, even with its
300 million parameters compared to the 8 billion parameters used in PlausiVL. In particular, V-JEPA 2
ViT-g384 demonstrates a +12.1 points improvement over PlausiVL on action recall-at-5, corresponding to a
44% relative improvement.
In Figure 11 we visualize V-JEPA 2 predictions on three samples from the EK100 validation set, two where
the model is successful and one where the model fails. For both successful examples, V-JEPA 2 not only
retrieves the correct action correctly with top 1 confidence, but also proposes coherent top 2 to 5 actions,
based on the given context. For example, in the top row, the correct action is "wash sink", but "turn on
water" or "clean wall" would both have been valid actions given the presence of a tap and a wall. The model
also predicts "rinse sponge", which is the current action being performed, probably assuming that this action
could still be going on after 1 second. For the failure case, V-JEPA 2 still proposes coherent actions such as
"close door" and "put down spices package", but misses the exact nature of the object: "tea package".
Limitations.
V-JEPA 2 and the EK100 benchmark have several limitations. First, V-JEPA 2 does not
fully solve EK100, there are failure cases where the model either gets the verb, the noun, or both wrong. We
study the distribution of these failures in Appendix D.2. Second, we focus here on predicting actions with a 1
second anticipation time. The accuracy of V-JEPA 2 degrades when predicting at longer time horizons, see
Appendix D.2. Third, the EK100 benchmark is limited to kitchen environments, with a closed well-defined
vocabulary, and we do not know how well V-JEPA 2 generalizes to other environments. This limits the
utility and applicability of models trained on EK100. Lastly, actions in EK100 are chosen from a fixed set of
categories, making it impossible to generalize to action categories not present in（来源：vjepa2.pdf 第 18 页）
- ska Meier, Yann LeCun, Michael
Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.
[303] Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable endto-end joint-embedding predictive architecture from pixels, 2026. URL https://arxiv.org/abs/2603.19312.
[304] Zhengcong Fei, Mingyuan Fan, and Junshi Huang. A-jepa: Joint-embedding predictive architecture can listen,
2024. URL https://arxiv.org/abs/2311.15830.
[305] Xianhang Li, Chen Huang, Chun-Liang Li, Eran Malach, Josh Susskind, Vimal Thilak, and Etai Littwin. Rethinking
jepa: Compute-efficient video ssl with frozen teachers, 2025. URL https://arxiv.org/abs/2509.24317.
[306] Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun,
Nicolas Ballas, and Adrien Bardes. V-jepa 2.1: Unlocking dense features in video self-supervised learning, 2026.
URL https://arxiv.org/abs/2603.14482.
[307] Linhan Wang, Zichong Yang, Chen Bai, Guoxiang Zhang, Xiaotong Liu, Xiaoyin Zheng, Xiao-Xiao Long, ChangTien Lu, and Cheng Lu. Drive-jepa: Video jepa meets multimodal trajectory distillation for end-to-end driving,
2026. URL https://arxiv.org/abs/2601.22032.
[308] Shangchen Miao, Ningya Feng, Jialong Wu, Ye Lin, Xu He, Dong Li, and Mingsheng Long. Jepa-vla: Video predictive embedding is needed for vla models, 2026. URL https://arxiv.org/abs/2602.11832.
[309] Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen
Bolourchi, Yann LeCun, and Pascale Fung. Vl-jepa: Joint embedding predictive architecture for vision-language,
2026. URL https://arxiv.org/abs/2512.10942.
[310] Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping, 2017. URL https://arxiv.org/abs/1611.06624.
[311] Aidan Clark, Jeff Donahue, and Karen Simonyan. Adversarial video（来源：world_action_models.pdf 第 65 页）
```

## Trace 2

```text
------------------------------------------------------------------------
[Query] 比较 V-JEPA 2 和 DINOv3 在机器人控制中的潜在优势。
[Graph] START
[Node] Planner
[Context Manager] Planner context built from Query + constraints only.
Raw Context Tokens: 43
Final Context Tokens: 43/3000
Plan:
1. V-JEPA 2 representation training objective and robotics capability [local_search]
2. DINOv3 representation characteristics and possible robotics relevance [local_search]
3. JEPA action-conditioned or world-model evidence relevant to robot control [local_search]
[State Update] plan=current structured plan
[Edge] Planner -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 128
Selected Evidence: 0/0
Final Context Tokens: 128/6000
[Node] Researcher
Iteration: 0
Plan step: 1/3
Tool Call: local_search
Arguments: {'query': 'V-JEPA 2 representation training objective and robotics capability', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 1803
Final Context Tokens: 1803
Compressed: False
Compression Ratio: 0.0%
Tool: local_search
Evidence records added: 5
Source: vjepa2.pdf | Page: 3
Source: vjepa2.pdf | Page: 23
Source: vjepa2.pdf | Page: 23
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 1832
Selected Evidence: 3/5
Final Context Tokens: 1832/6000
[Node] Researcher
Iteration: 0
Plan step: 2/3
Tool Call: local_search
Arguments: {'query': 'DINOv3 representation characteristics and possible robotics relevance', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 1886
Final Context Tokens: 1886
Compressed: False
Compression Ratio: 0.0%
Tool: local_search
Evidence records added: 5
Source: dinov3.pdf | Page: 17
Source: dinov3.pdf | Page: 28
Source: dinov3.pdf | Page: 1
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 3029
Selected Evidence: 6/10
Final Context Tokens: 3029/6000
[Node] Researcher
Iteration: 0
Plan step: 3/3
Tool Call: local_search
Arguments: {'query': 'JEPA action-conditioned or world-model evidence relevant to robot control', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 1945
Final Context Tokens: 1945
Compressed: False
Compression Ratio: 0.0%
Tool: local_search
Evidence records added: 5
Source: vjepa2.pdf | Page: 9
Source: vjepa2.pdf | Page: 8
Source: vjepa2.pdf | Page: 1
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Writer
[Node] Writer
[Context Manager] Writer context excludes ToolNode history and old drafts.
Selected Evidence: 10/15
Deduplicated Evidence Count: 4
Dropped Evidence Count: 1
Raw Context Tokens: 4672
Final Context Tokens: 4672/8000
Draft generated from Query + Plan + Evidence.
[State Update] draft=current answer draft
[Edge] Writer -> Verifier
[Node] Verifier
[Context Manager] Verifier context contains Query + Draft + selected Evidence only.
Selected Evidence: 10/15
Deduplicated Evidence Count: 4
Dropped Evidence Count: 1
Raw Context Tokens: 8976
Final Context Tokens: 8000/8000
Passed: True
Feedback: Draft addresses the query and has available evidence records.
[State Update] iteration=1, verification_passed=True
[Conditional Edge] Verifier -> END
[Graph] END
[Final Answer]
基于当前研究证据：
- ments.
The remainder of this paper is organized as follows. Section 2 describes the V-JEPA 2 pretraining procedure,
including the key ingredients enabling scaling beyond the original V-JEPA recipe of Bardes et al. (2024).
Section 3 then introduces our approach to training a task-agnostic action-conditioned world model, V-JEPA 2AC, leveraging the pretrained V-JEPA 2 model. Section 4 demonstrates using V-JEPA 2-AC for robot control
via model-based planning. Because V-JEPA 2-AC models world dynamics in a learned representation space,
its capabilities fundamentally depend on the information captured in the V-JEPA 2 representation space, and
so we further explore the performance of V-JEPA 2 for video understanding in Section 5 and prediction tasks
in Section 6. Finally, in Section 7 we show that V-JEPA 2 can be aligned with a language model for video
question answering. Section 8 discusses related work, and we conclude in Section 9.
3 PA 2 excels at encoding fine-grained motion information,
achieving strong performance on tasks requiring motion understanding, such as Something-Something v2,
with 77.3 top-1 accuracy using an attentive probe.
• Understanding — Video Question-Answering: V-JEPA 2 encoder can be used to train a multi-modal
large language model, to tackle video-question answering tasks. We observe state-of-the-art performance
on 8B language model class on multiple benchmarks that require physical world understanding and
temporal reasoning, such as MVP (44.5 paired accuracy), PerceptionTest (84.0 test set accuracy),
TempCompass (76.9 multi-choice accuracy), TemporalBench (36.7 multi-binary short-QA accuracy) and
TOMATO (40.3 accuracy). In particular, we show that a video encoder pre-trained without language
supervision can be aligned with a language model and achieve state-of-the-art performance, contrary to
conventional wisdom (Yuan et al., 2025; Wang et al., 2024b).
• Prediction: Large-scale self-supervised video pretraining enhances prediction capabilities. V-JEPA 2
achieves state-of-the-art performance on the Epic-Kitchens-100 human-action anticipation task using an
attentive probe, with 39.7 recall-at-5, which is a 44% relative improvement over the previous best model.
• Planning: We demonstrate that V-JEPA 2-AC, obtained by post-training V-JEPA 2 with only 62
hours of unlabeled robot manipulation data from the popular Droid dataset, can be deployed in new
environments to solve prehensile manipulation tasks using planning with given subgoals. Without
training on any additional data from robots in our labs, and without any task-specific training or reward,
the model successfully handles prehensile manipulation tasks, such as Grasp and Pick-and-Place with
novel objects and in new environments.
The remainder of this paper is organized as follows. Section 2 describes the V-JEPA 2 pretraining procedure,
including the key ingredients enabling scaling beyond the original V-JEPA recipe of（来源：vjepa2.pdf 第 3 页）
- 3.1
Action-Conditioned World Model Training
Our goal is to take the V-JEPA 2 model after pre-training and obtain a latent world model that can be used
for control of an embodied agentic system via closed-loop model-predictive control. To achieve this, we train
V-JEPA 2-AC, an autoregressive model that predicts representations of future video observations conditioned
on control actions and proprioceptive observations.
In this section we describe a concrete instantiation of this framework for a tabletop arm with a fixed
exocentric camera, and where control actions correspond to end-effector commands. The model is trained
using approximately 62 hours of unlabeled video from the raw Droid dataset, which consists of short videos,
typically 3–4 seconds long, of a 7-DoF Franka Emika Panda arm equipped with a two-finger gripper. Here,
unlabeled video refers to the fact that we do not use additional meta-data indicating any reward, what type
of task was being performed in each demonstration, or whether the demonstration was successful or not in
completing the task being attempted. Rather, we only use the raw video and end-effector state signals from
the dataset (each video in the dataset is accompanied by meta-data indicating the end-effector state in each
frame — three dimensions for position, three for orientation, and one for the gripper state).
Model inputs.
In each iteration of training we randomly sample a mini-batch of 4 second video clips
from the Droid dataset, and, for simplicity, discard any videos shorter than 4 seconds, leaving us with a
smaller subset of the dataset comprising under 62 hours of video. The video clips are sampled with resolution
256 × 256 and a frame-rate of 4 frames-per-second (fps), yielding 16 frame clips denoted by (xk)k∈[16], where
each xk represents a single video frame. The robot’s end-effector state in each observation is denoted by the
sequence (sk)k∈[16], where sk is a real-valued 7D vector defined relative to the base of the robot. Th（来源：vjepa2.pdf 第 9 页）
- previous experiment, we find that DINOv3 can be
successfully used for extracting strong video features. As this evaluation involves training several layers of
self-attention, the differences between models are less visible. However, DINOv3 lands in the same range
as PEcore and SigLIP 2, and clearly outperforms other models (DINOv2, AM-RADIO) across datasets.
UCF101 and K400 are appearance-focused, where strong category-level understanding of objects gives most
of the performance. SSv2 on the other hand, requires better understanding of motion—the dedicated video
model V-JEPA v2 shines on this dataset. Interestingly, the gap between DINOv3 and the weakly-supervised
models is slightly bigger on this dataset. This again confirms the suitability of DINOv3 to video tasks.
22（来源：dinov3.pdf 第 22 页）
- 9
Conclusion
This study demonstrates how joint-embedding predictive architectures, learning in a self-supervised manner
from web-scale data and a small amount of robot interaction data, can yield a world model capable of
understanding, predicting, and planning in the physical world. V-JEPA 2 achieves state-of-art performances
on action classification requiring motion understanding and human action anticipation. V-JEPA 2 also
outperforms previous vision encoders on video questions-answering tasks when aligned with a large-language
model. Additionally, post-training an action-conditioned world model, V-JEPA 2-AC, using V-JEPA 2’s
representation, enables successful zero-shot prehensile manipulation tasks, such as Pick-and-Place, with
real-world robots. These findings indicate V-JEPA 2 is a step towards developing advanced AI systems that
can effectively perceive and act in their environment.
Future work.
There are several important avenues for future work to address limitations of V-JEPA 2.
First, in this work we have focused on tasks requiring predictions up to roughly 16 seconds into the future. This
enables planning for simpler manipulation tasks, like grasp and reach-with-object, from a single goal image.
However, to extend this to longer-horizon tasks such as pick-and-place or even more complex tasks, without
requiring sub-goals will require further innovations in modeling. Developing approaches for hierarchical models
capable of making predictions across multiple spatial and temporal scales, at different levels of abstraction, is
a promising direction.
Second, as mentioned in Section 4, V-JEPA 2-AC currently relies upon tasks specified as image goals. Although
this may be natural for some tasks, there are other situations where language-based goal specification may be
preferable. Extending the V-JEPA 2-AC to accept language-based goals, e.g., by having a model that can
embed language-based goals into the V-JEPA 2-AC representation space, is another important dire e
preferable. Extending the V-JEPA 2-AC to accept language-based goals, e.g., by having a model that can
embed language-based goals into the V-JEPA 2-AC representation space, is another important direction for
future work. The results described in Section 7, aligning V-JEPA 2 with a language model, may serve as a
starting point.
Finally, in this work we scaled V-JEPA 2 models up to a modest 1B parameters. The results in Section 2
demonstrated consistent performance improvements while scaling to this level. Previous work has investigated
scaling vision encoders to as large as 20B parameters (Zhai et al., 2022; Carreira et al., 2024). Additional
work is needed in this direction to develop scalable pre-training recipes that lead to sustained performance
improvements with scale.
Acknowledgements
We thank Rob Fergus, Joelle Pineau, Stephane Kasriel, Naila Murray, Mrinal Kalakrishnan, Jitendra Malik,
Randall Balestriero, Julen Urain, Gabriel Synnaeve, Michel Meyer, Pascale Fung, Justine Kao, Florian Bordes,
Aaron Foss, Nikhil Gupta, Cody Ohlsen, Kalyan Saladi, Ananya Saxena, Mack Ward, Parth Malani, Shubho
Sengupta, Leo Huang, Kamila Benzina, Rachel Kim, Ana Paula Kirschner Mofarrej, Alyssa Newcomb, Nisha
Deo, Yael Yungster, Kenny Lehmann, Karla Martucci, and the PerceptionLM team, including Christoph
Feichtenhofer, Andrea Madotto, Tushar Nagarajan, and Piotr Dollar for their feedback and support of this
project.
23（来源：vjepa2.pdf 第 23 页）
- ed to robotic
planning tasks by post-training a latent action-conditioned world model, V-JEPA 2-AC, using less
than 62 hours of unlabeled robot videos from the Droid dataset. We deploy V-JEPA 2-AC zero-shot on
Franka arms in two different labs and enable picking and placing of objects using planning with image
goals. Notably, this is achieved without collecting any data from the robots in these environments,
and without any task-specific training or reward. This work demonstrates how self-supervised learning
from web-scale data and a small amount of robot interaction data can yield a world model capable of
planning in the physical world.
Date: June 13, 2025
Correspondence: Nicolas Ballas <ballasn@meta.com> and Michael Rabbat <mikerabbat@meta.com>
Code: https://github.com/facebookresearch/vjepa2
Blogpost: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
1
Introduction
Humans have the ability to adapt and generalize when taking on new tasks and operating in unfamiliar
environments. Several cognitive learning theories suggest that humans learn an internal model of the world
by integrating low-level sensory inputs to represent and predict future states (Craik, 1967; Rao and Ballard,
1999), and they further posit that this world model shapes our perception at any given moment, playing a
crucial role in informing our understanding of reality (Friston, 2010; Clark, 2013; Nortmann et al., 2015).
Moreover, our ability to predict the effects of our actions on future states of the world is also essential for
goal-oriented planning (Sutton and Barto, 1981, 1998; Ha and Schmidhuber, 2018; Wolpert and Ghahramani,
2000). Building artificial agents that learn a world model from sensory data, such as video, could enable
them to understand the physical world, predict future states, and effectively — like humans — plan in new
situations, resulting in systems capable of tackling tasks that have not been encountered before.
Previous works have explored the development of predict（来源：vjepa2.pdf 第 1 页）
- Internet Video
& Images
1M hours & 1M images
Video
Pretraining
Language
Alignment
Attentive Probe
Training
ActionConditioned
Post-Training
Robot Data
(states + actions)
62 hours
Understanding
& Prediction
Action Classification
Object Recognition
Action Anticipation
Understanding
Video QA
Planning
Robot Manipulation
V-JEPA 2
Figure 1 V-JEPA 2 Overview. Leveraging 1M hours of internet-scale video and 1M images, we pretrain the V-JEPA 2
video model using a visual mask denoising objective (Bardes et al., 2024; Assran et al., 2023), and leverage this
model for downstream tasks such as action classification, object recognition, action anticipation, and Video Question
Answering by aligning the model with an LLM backbone. After pretraining, we can also freeze the video encoder
and train a new action-conditioned predictor with a small amount of robot interaction data on top of the learned
representations, and leverage this action-conditioned model, V-JEPA 2-AC, for downstream robot manipulation tasks
using planning within a model predictive control loop.
of state-action sequences, often also relying on explicit reward feedback from the environment to infer
goals (Sutton and Barto, 1981; Fragkiadaki et al., 2015; Ha and Schmidhuber, 2018; Hafner et al., 2019b;
Hansen et al., 2022). However, the limited availability of real-world interaction data constrains the scalability
of these methods. To address this limitation, more recent works have leveraged both internet-scale video
and interaction data towards training action-conditioned video generation models for robot control, but only
demonstrate limited results in robot execution using model-based control (Hu et al., 2023; Yang et al., 2024b;
Bruce et al., 2024; Agarwal et al., 2025). In particular, this line of research often emphasizes the evaluation
of the faithfulness of the predictions and visual quality instead of planning capabilities, perhaps due to the
computational cost of planning by generating video.
In this work, w（来源：vjepa2.pdf 第 2 页）
- tioned predictor on
top of the frozen V-JEPA 2 video encoder (Figure 2, right). We train our model on data from the Droid
dataset (Khazatsky et al., 2024) consisting of data from experiments with a table-top Franka Panda robot
arm collected through teleoperation. We refer to the resulting action-conditioned model as V-JEPA 2-AC,
and in Section 4 we show that V-JEPA 2-AC can be used within a model-predictive control planning loop to
plan actions in new environments.
8（来源：vjepa2.pdf 第 8 页）
- ut
our experiments, unless otherwise specified, we keep DINOv3 frozen and solely use its representations. We
demonstrate that with DINOv3, finetuning is not necessary to obtain strong performance. This section
is organized as follows.
We first probe the quality of DINOv3’s dense (Sec. 6.1) and global (Sec. 6.2)
image representations using lightweight evaluation protocols and compare it to the strongest available vision
encoders. We show that DINOv3 learns exceptional dense features while offering robust and versatile global
image representations. Then, we consider DINOv3 as a basis for developing more complex computer vision
systems (Sec. 6.3). We show with little effort on top of DINOv3, we are able to achieve results competitive
with or exceeding the state of the art in tasks as diverse as object detection, semantic segmentation, 3D view
estimation, or relative monocular depth estimation.
6.1
DINOv3 provides Exceptional Dense Features
We first investigate the raw quality of DINOv3’s dense representations using a diverse set of lightweight
evaluations. In all cases, we utilize the frozen patch features of the last layer, and evaluate them using (1)
qualitative visualizations (Sec. 6.1.1), (2) dense linear probing (Sec. 6.1.2: segmentation, depth estimation),
(3) non-parametric approaches (Sec. 6.1.3: 3D correspondence estimation, Sec. 6.1.4: object discovery,
Sec. 6.1.5: tracking), and (4) lightweight attentive probing (Sec. 6.1.6: video classification).
Baselines
We compare the dense features of DINOv3 with those of the strongest publicly available image
encoders, both weakly- and self-supervised ones. We consider the weakly-supervised encoders Perception
Encoder (PE) Core (Bolya et al., 2025) and SigLIP 2 (Tschannen et al., 2025), which use CLIP-style imagetext contrastive learning. We also compare to the strongest self-supervised methods: DINOv3’s predecessor
DINOv2 (Oquab et al., 2024) with registers (Darcet et al., 2024), Web-DINO (Fan et al., 2025), a recent（来源：dinov3.pdf 第 17 页）
```

## Trace 3

```text
------------------------------------------------------------------------
[Query] 比较 V-JEPA 2、DINOv3、JEPA-WAM、Patch Policy，从训练目标、视觉表征、动作建模、机器人泛化、开源情况五个方面进行系统分析。
[Graph] START
[Node] Planner
[Context Manager] Planner context built from Query + constraints only.
Raw Context Tokens: 60
Final Context Tokens: 60/3000
Plan:
1. V-JEPA 2 training objective and visual representation [local_search]
2. DINOv3 visual representation and transfer evidence [local_search]
3. JEPA-WAM action modeling and world-model structure [local_search]
4. Patch Policy robot control and generalization evidence [local_search]
5. Compare source-backed open-source or benchmark evidence across the methods [local_search]
[State Update] plan=current structured plan
[Edge] Planner -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 162
Selected Evidence: 0/0
Final Context Tokens: 162/6000
[Node] Researcher
Iteration: 0
Plan step: 1/5
Tool Call: local_search
Arguments: {'query': 'V-JEPA 2 training objective and visual representation', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 2187
Final Context Tokens: 2187
Compressed: False
Compression Ratio: 0.0%
Tool: local_search
Evidence records added: 5
Source: vjepa2.pdf | Page: 23
Source: vjepa2_1.pdf | Page: 1
Source: vjepa2.pdf | Page: 3
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 2263
Selected Evidence: 5/5
Final Context Tokens: 2263/6000
[Node] Researcher
Iteration: 0
Plan step: 2/5
Tool Call: local_search
Arguments: {'query': 'DINOv3 visual representation and transfer evidence', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 1986
Final Context Tokens: 1986
Compressed: False
Compression Ratio: 0.0%
Tool: local_search
Evidence records added: 5
Source: dinov3.pdf | Page: 17
Source: dinov3.pdf | Page: 28
Source: dinov3.pdf | Page: 7
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 2756
Selected Evidence: 6/10
Final Context Tokens: 2756/6000
[Node] Researcher
Iteration: 0
Plan step: 3/5
Tool Call: local_search
Arguments: {'query': 'JEPA-WAM action modeling and world-model structure', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 2387
Final Context Tokens: 2387
Compressed: False
Compression Ratio: 0.0%
Tool: local_search
Evidence records added: 5
Source: world_action_models.pdf | Page: 3
Source: world_action_models.pdf | Page: 1
Source: world_action_models.pdf | Page: 13
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 2891
Selected Evidence: 6/15
Final Context Tokens: 2891/6000
[Node] Researcher
Iteration: 0
Plan step: 4/5
Tool Call: local_search
Arguments: {'query': 'Patch Policy robot control and generalization evidence', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 2598
Final Context Tokens: 2332
Compressed: True
Compression Ratio: 10.2%
Tool: local_search
Evidence records added: 5
Source: patch_policy.pdf | Page: 4
Source: patch_policy.pdf | Page: 6
Source: patch_policy.pdf | Page: 10
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Researcher
[Context Manager] Researcher context excludes full messages/raw results.
Raw Context Tokens: 3048
Selected Evidence: 6/20
Final Context Tokens: 3048/6000
[Node] Researcher
Iteration: 0
Plan step: 5/5
Tool Call: local_search
Arguments: {'query': 'Compare source-backed open-source or benchmark evidence across the methods', 'top_k': 5}
[Edge] Researcher -> ToolNode
[Node] Evidence
[Context Manager] Tool Result Compression
Raw Context Tokens: 1599
Final Context Tokens: 1599
Compressed: False
Compression Ratio: 0.0%
Tool: local_search
Evidence records added: 5
Source: vjepa2.pdf | Page: 6
Source: world_action_models.pdf | Page: 35
Source: vjepa2.pdf | Page: 34
[State Update] evidence += normalized ToolNode result
[Edge] Evidence -> Writer
[Node] Writer
[Context Manager] Writer context excludes ToolNode history and old drafts.
Selected Evidence: 10/25
Deduplicated Evidence Count: 0
Dropped Evidence Count: 15
Raw Context Tokens: 4668
Final Context Tokens: 4668/8000
Draft generated from Query + Plan + Evidence.
[State Update] draft=current answer draft
[Edge] Writer -> Verifier
[Node] Verifier
[Context Manager] Verifier context contains Query + Draft + selected Evidence only.
Selected Evidence: 10/25
Deduplicated Evidence Count: 0
Dropped Evidence Count: 15
Raw Context Tokens: 8281
Final Context Tokens: 8000/8000
Passed: True
Feedback: Draft addresses the query and has available evidence records.
[State Update] iteration=1, verification_passed=True
[Conditional Edge] Verifier -> END
[Graph] END
[Final Answer]
基于当前研究证据：
- e
preferable. Extending the V-JEPA 2-AC to accept language-based goals, e.g., by having a model that can
embed language-based goals into the V-JEPA 2-AC representation space, is another important direction for
future work. The results described in Section 7, aligning V-JEPA 2 with a language model, may serve as a
starting point.
Finally, in this work we scaled V-JEPA 2 models up to a modest 1B parameters. The results in Section 2
demonstrated consistent performance improvements while scaling to this level. Previous work has investigated
scaling vision encoders to as large as 20B parameters (Zhai et al., 2022; Carreira et al., 2024). Additional
work is needed in this direction to develop scalable pre-training recipes that lead to sustained performance
improvements with scale.
Acknowledgements
We thank Rob Fergus, Joelle Pineau, Stephane Kasriel, Naila Murray, Mrinal Kalakrishnan, Jitendra Malik,
Randall Balestriero, Julen Urain, Gabriel Synnaeve, Michel Meyer, Pascale Fung, Justine Kao, Florian Bordes,
Aaron Foss, Nikhil Gupta, Cody Ohlsen, Kalyan Saladi, Ananya Saxena, Mack Ward, Parth Malani, Shubho
Sengupta, Leo Huang, Kamila Benzina, Rachel Kim, Ana Paula Kirschner Mofarrej, Alyssa Newcomb, Nisha
Deo, Yael Yungster, Kenny Lehmann, Karla Martucci, and the PerceptionLM team, including Christoph
Feichtenhofer, Andrea Madotto, Tushar Nagarajan, and Piotr Dollar for their feedback and support of this
project.
23（来源：vjepa2.pdf 第 23 页）
- V-JEPA 2
+ Context Loss
+ Deep Self-Supervision.
+ Data Scaling
+ Multimodal Patch.
+ Model Scaling
+ High-Res. Annealing
20
25
30
35
40
45
50
22.2
33.8
38.6
40.8
41.4
47.1
47.9
mIoU (ADE20k)
Segmentation (ADE20k)
Classification (SSv2)
60
62
64
66
68
70
72
74
76
78
80
72.8
62.5
72.1
72.6
72.6
76.1
77.7
Acc. (SSv2)
Figure 5 Impact of individual components of our novel V-JEPA 2.1 training recipe. The ablation is conducted starting from a
ViT-L architecture, on single-image semantic segmentation on ADE20k, and action classification on SSv2. Introducing
Weighted-Context Self-Supervision with the context loss significantly improves segmentation, at the cost of classification.
Deep Self-Supervision allows to retrieve the performance, and further improving segmentation. Scaling the image data
with VisionMix 163M dataset, combined with a Multi-Modal Tokenizer further improve the results. Finally, the recipe
scales with model size and high resolution cool-down.
VJEPA 2.1 Architecture.
We illustrate the V-JEPA 2.1 architecture in Figure 4. An input, either an
image or a video, is projected into a sequence of embedding vectors, or tokens, using a modality-specific
patch embedding. Mask corruption is then applied to the sequence by randomly dropping patch tokens. The
x-encoder processes the remaining visible context tokens and outputs representations from multiple encoder
levels in addition to the final output. The multi-level representations are then concatenated along the channel
axis and fed to an MLP to reduce their dimensionality. Context tokens are concatenated, along the sequence
axis, with learnable mask tokens that carry spatio-temporal positional information of the masked patches.
The predictor processes the combined sequence and produces multi-level predictions for each token. Training
uses two different losses: (i) an L1 loss on masked-token predictions (the original V-JEPA objective), and (ii)
a distance-weighted L1 loss for context tokens. Both use the y-encoder o（来源：vjepa2_1.pdf 第 6 页）
- 𝑝(𝑜′, 𝑎| 𝑜, 𝑙) over future states and actions rather than actions alone. Existing WAM methods can be broadly
organized into two architectural categories: (1) Cascaded WAM explicitly factorizes the objective, formally
𝑝(𝑜′, 𝑎| 𝑜, 𝑙) = 𝑝(𝑎| 𝑜′, 𝑜, 𝑙)𝑝(𝑜′ | 𝑜, 𝑙), by first synthesizing representations of anticipated future states,
from which actions are subsequently derived; and (2) Joint WAM directly modeling the joint distribution
(𝑝(𝑜′, 𝑎| 𝑜, 𝑙)) , where state prediction and action generation are co-optimized within a shared representational space (see Fig. 1 for the temporal evolution of these architectures). The integration of world modeling
brings stronger physical understanding, improved generalization across novel environments, and the ability to leverage large-scale human video data that lack action annotations—substantially expanding the data
foundations available for embodied policy learning.
This survey provides the first systematic and critical analysis of the World Action Model landscape. Our
aim is to offer both a conceptual framework for understanding the design space and a practical guide for
researchers entering this rapidly evolving field. The organization of this survey (as illustrated in Fig. 2) is as
follows:
• Definition (Sec. 2). We provide a formal definition of World Action Models and disambiguate them
from related concepts, including Video Policies, Action-Conditioned World Models, and standard VLA
models, clarifying the terminological boundaries that currently fragment the literature.
• Background (Sec. 3). We trace the intertwined development of world modeling and action generation
from classical model-based reinforcement learning through modern foundation model approaches,
situating WAMs within a broader intellectual lineage.
• Architecture (Sec. 4). We categorize existing WAM methods into Cascaded and Joint paradigms, with
further subdivision by generation modality, conditioning mechanism, and action decoding strategy,
providing a unified（来源：world_action_models.pdf 第 3 页）
- V-JEPA 2.1: Unlocking Dense Features in Video
Self-Supervised Learning
Lorenzo Mur-Labadia1,2,∗, Matthew Muckley1, Amir Bar1, Mido Assran1, Koustuv Sinha1, Mike Rabbat1,
Yann LeCun1, Nicolas Ballas1,†, Adrien Bardes1,†
1FAIR at Meta, 2Universidad de Zaragoza
∗Work done at Meta, †Joint last author
We present V-JEPA2.1, a family of self-supervised models that learns dense, high-quality representations
for visual scenes in both images and videos, while retaining strong global scene understanding. V-JEPA
2.1 combines four key ingredients: (i) a Dense Predictive Loss, a masking-based objective in which
all tokens—visible context and masked tokens alike—contribute to the training loss, encouraging
explicit spatial and temporal grounding; (ii) Deep Self-Supervision, which applies the self-supervised
objective hierarchically at multiple intermediate encoder layers to improve representation quality ;
(iii) Multi-Modal Tokenizers that support unified training over images and videos; and (iv) effective
model and data scaling. These design choices substantially improve dense feature quality, yielding
representations that are spatially structured, semantically coherent, and temporally consistent. .
Empirically, V-JEPA 2.1 achieves state-of-the-art results on a range of benchmarks: 7.71 mAP on
Ego4D for short-term object-interaction anticipation, 40.8 Recall@5 on EPIC-KITCHENS for highlevel action anticipation, and a 20% improvement in real-robot grasping success rate over VJEPA-2
AC. The model also demonstrates state-of-art performances in robotic navigation (5.687 ATE on
Tartan Drive), depth estimation (0.307 RMSE on NYUv2 with a linear probe), and global recognition
(77.7% on Something-Something-V2). Our results demonstrate that V-JEPA 2.1 advances the state of
the art in dense visual understanding and world modeling.
Date: June 12, 2026
Correspondence: lmur@unizar.es, abardes@meta.com
Code: https://github.com/facebookresearch/vjepa2
V-JEPA 2.1
V-JEPA 2
Image/Video
Figure 1 V-（来源：vjepa2_1.pdf 第 1 页）
- Figure 3: We evaluate PATCH POLICY on four simulated and three real-world environments. havior Transformer (VQ-BeT) [21], which uses a hybrid classification-regression loss, and Diffusion Policy (DP) [22], which uses a denoising objective. During training, we forward a sequence of patch tokens through the policy transformer trunk and action head, and compute a loss between the predicted and ground-truth actions for each frame. For inference, we extract the patch features from the current observation and append them to a rolling context window of length T. The policy predicts a chunk of actions from the observation context, and we execute the actions with receding horizon control. 3 Experiments We evaluate PATCH POLICY across four simulated environments and three real robot manipulation environments, aiming to answer the following key research questions: 1. How does PATCH POLICY compare to state-of-the-art policies using global and patch features? 2. Does PATCH POLICY work for real-world precise manipulation? 3. How do the latest pretrained encoders perform as representations for downstream policy learning? 4. How does the spatial compression of visual features impact downstream task performance? 5. How does PATCH POLICY compare to other methods in terms of efficiency? How do attention design and model size affect its performance? (Ablations Section A.6) 3.1 Environments We evaluate PATCH POLICY across four simulated environments (Push-T, LIBERO Goal, BlockPush, Cube) with 2D-to-7D action spaces, and three real-world tasks using a 7-DoF Franka arm with a parallel-jaw gripper (inserting a power cable, hanging a tool, and collecting pens into a holder). Environments are visualized in Figure 3 and detailed in App. Section A.2. 3.2 Baselines We compare against two groups of（来源：patch_policy.pdf 第 4 页）
- ments.
The remainder of this paper is organized as follows. Section 2 describes the V-JEPA 2 pretraining procedure,
including the key ingredients enabling scaling beyond the original V-JEPA recipe of Bardes et al. (2024).
Section 3 then introduces our approach to training a task-agnostic action-conditioned world model, V-JEPA 2AC, leveraging the pretrained V-JEPA 2 model. Section 4 demonstrates using V-JEPA 2-AC for robot control
via model-based planning. Because V-JEPA 2-AC models world dynamics in a learned representation space,
its capabilities fundamentally depend on the information captured in the V-JEPA 2 representation space, and
so we further explore the performance of V-JEPA 2 for video understanding in Section 5 and prediction tasks
in Section 6. Finally, in Section 7 we show that V-JEPA 2 can be aligned with a language model for video
question answering. Section 8 discusses related work, and we conclude in Section 9.
3（来源：vjepa2.pdf 第 3 页）
- ut
our experiments, unless otherwise specified, we keep DINOv3 frozen and solely use its representations. We
demonstrate that with DINOv3, finetuning is not necessary to obtain strong performance. This section
is organized as follows.
We first probe the quality of DINOv3’s dense (Sec. 6.1) and global (Sec. 6.2)
image representations using lightweight evaluation protocols and compare it to the strongest available vision
encoders. We show that DINOv3 learns exceptional dense features while offering robust and versatile global
image representations. Then, we consider DINOv3 as a basis for developing more complex computer vision
systems (Sec. 6.3). We show with little effort on top of DINOv3, we are able to achieve results competitive
with or exceeding the state of the art in tasks as diverse as object detection, semantic segmentation, 3D view
estimation, or relative monocular depth estimation.
6.1
DINOv3 provides Exceptional Dense Features
We first investigate the raw quality of DINOv3’s dense representations using a diverse set of lightweight
evaluations. In all cases, we utilize the frozen patch features of the last layer, and evaluate them using (1)
qualitative visualizations (Sec. 6.1.1), (2) dense linear probing (Sec. 6.1.2: segmentation, depth estimation),
(3) non-parametric approaches (Sec. 6.1.3: 3D correspondence estimation, Sec. 6.1.4: object discovery,
Sec. 6.1.5: tracking), and (4) lightweight attentive probing (Sec. 6.1.6: video classification).
Baselines
We compare the dense features of DINOv3 with those of the strongest publicly available image
encoders, both weakly- and self-supervised ones. We consider the weakly-supervised encoders Perception
Encoder (PE) Core (Bolya et al., 2025) and SigLIP 2 (Tschannen et al., 2025), which use CLIP-style imagetext contrastive learning. We also compare to the strongest self-supervised methods: DINOv3’s predecessor
DINOv2 (Oquab et al., 2024) with registers (Darcet et al., 2024), Web-DINO (Fan et al., 2025), a recent（来源：dinov3.pdf 第 17 页）
- of the target encoder by a frozen teacher
model, ii) we keep an EMA copy of the student encoder, that is not used in the loss, but serves as the final
model, iii) the distillation loss is identical to our pretraining loss, except it is only computed on the last layer
of the teacher encoder, and it does not use deep self-supervision, iv) we use a predictor with only 12 blocks
and a final linear layer matching the teacher embedding dimension. All other hyper-parameters—including
masking ratios, cool-down schedules, learning rates, and data augmentations—remain identical to the original
pretraining recipe. We provide more details on the distillation protocol in Appendix B.
3
Results
In this section, we evaluate the performance of V-JEPA 2.1 on a large variety of downstream tasks. Throughout
the experiments, we employ V-JEPA 2.1 as a frozen encoder, demonstrating the versatility of its features.
We first demonstrate the predictive capabilities of V-JEPA 2.1 in two forecasting tasks: short-term object
interaction anticipation (Section 3.1) and action anticipation (Section 3.2). We then demonstrate that can
leverage V-JEPA 2.1 to learn an action-condition world model and performs robot manipulation (Section 3.3)
and navigation tasks (Section 3.4) in zero-shot setup. Then, we evaluate the quality of the learned dense
features by assessing the V-JEPA 2.1 performance on single-image depth estimation and semantic segmentation
(Section 3.5). Following, we analyze the temporal consistency of V-JEPA 2.1 representations through the
video object segmentation task (Section 3.5). We also analyze V-JEPA 2.1 global understanding on two
high-level understanding tasks: probe-based video classification and image classification (Section 3.7). Finally,
we present results for our smaller distilled models (Section 3.10). We provide more details on the experimental
setup and all pretraining hyper-parameters in Appendices A and C.
3.1
Short-Term Object Interaction Anticipation
Short-Term obje（来源：vjepa2_1.pdf 第 9 页）
```
