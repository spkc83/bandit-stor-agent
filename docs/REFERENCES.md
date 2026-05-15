# References and Related Work

Bandit-STOR is an offline contextual-bandit recommender prototype. It uses causal-inference-adjacent estimators such as IPS and doubly robust OPE, but it does **not** claim true causal identification unless the assumptions in `docs/ASSUMPTIONS.md` are defensible for the deployment data-generating process.

## Foundational papers

- Dudík, Langford, and Li, **“Doubly Robust Policy Evaluation and Learning”** (ICML 2011 / arXiv). Introduces doubly robust policy evaluation/learning for contextual bandits.  
  https://arxiv.org/abs/1103.4601
- Swaminathan and Joachims, **“Counterfactual Risk Minimization: Learning from Logged Bandit Feedback”** (ICML 2015). Motivates propensity-weighted learning from logged bandit feedback and variance-aware learning objectives.  
  https://proceedings.mlr.press/v37/swaminathan15.html
- Jiang and Li, **“Doubly Robust Off-policy Value Evaluation for Reinforcement Learning”** (ICML 2016). Extends DR OPE ideas in sequential/off-policy settings; useful background for the DR estimator family, though this project remains one-step contextual bandit.  
  https://proceedings.mlr.press/v48/jiang16.html
- Saito et al., **“Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation”** (arXiv 2020). Source and evaluation framework inspiration for the OBD/OBP integration.  
  https://arxiv.org/abs/2008.07146
- Su, Dimakopoulou, Krishnamurthy, and Dudík, **“Doubly robust off-policy evaluation with shrinkage”** (ICML 2020). Motivation for DR with shrinkage / DRos sensitivity diagnostics.  
  https://proceedings.mlr.press/v119/su20a.html
- Martins and Astudillo, **“From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification”** (ICML 2016). Basis for sparsemax probability projections used by the actor.  
  https://proceedings.mlr.press/v48/martins16.html
- Lee, Kim, Lim, Choi, and Oh, **“Tsallis Reinforcement Learning: A Unified Framework for Maximum Entropy Reinforcement Learning”** (arXiv 2019). Background for Tsallis entropy regularization and sparse policy geometry.  
  https://arxiv.org/abs/1902.00137

## Relevant open-source repositories

- Open Bandit Pipeline (OBP), the project’s primary dataset/OPE ecosystem reference: https://github.com/st-tech/zr-obp
- Vowpal Wabbit, production-oriented contextual-bandit tooling and reductions: https://github.com/VowpalWabbit/vowpal_wabbit
- COBA, contextual-bandit benchmarking from the Vowpal Wabbit ecosystem: https://github.com/VowpalWabbit/coba
- MABWiser, Python multi-armed/contextual-bandit library: https://github.com/fidelity/mabwiser
- Mab2Rec, bandit-based recommender library: https://github.com/fidelity/mab2rec
- Criteo RecoGym, recommender/RL environment relevant as a related benchmark environment, not a logged-propensity replacement for OBD: https://github.com/criteo-research/reco-gym

## Interpretation of estimates

The project terminology uses “off-policy,” “counterfactual evaluation under assumptions,” and “contextual-bandit OPE.” Terms such as “causal recommender,” “causal effect,” and “causal lift” require an accompanying identification analysis that establishes logging-policy validity, overlap, exchangeability, and stable reward semantics.
