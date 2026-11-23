# Mathematical methods and algorithmic approach

## 1. Formal problem

Let an audio utterance be (X = (x_1, …, x_N)). The system must infer:

- a word sequence (Y = (y_1, …, y_T));
- a token language sequence (L = (l_1, …, l_T)), including an `ambiguous/neutral` state;
- switch variables (B_t = 1[l_t \ne l_{t-1}]) for non-neutral adjacent tokens;
- entity spans (E = \{(s_j, e_j, k_j, v_j)\}) with type (k_j) and value (v_j);
- a calibrated probability that each word or entity is wrong;
- an action (a \in \{\text{accept}, \text{review}, \text{defer}\}).

The product objective is not minimum WER alone. It is minimum expected user harm under latency, privacy, accessibility, and review-burden constraints.

## 2. Multi-task representation learning

For a trainable research backbone, use a shared speech encoder (h=f_\theta(X)) with ASR, token-language, boundary, and confidence heads. The initial objective is:

\[
\mathcal{L} =
\lambda_{asr}\mathcal{L}_{asr}
+ \lambda_{lid}\mathcal{L}_{lid}
+ \lambda_{b}\mathcal{L}_{boundary}
+ \lambda_{c}\mathcal{L}_{confidence}
+ \lambda_{e}\mathcal{L}_{entity}
\]

Candidate ASR losses are sequence-to-sequence cross entropy for Whisper adaptation or a joint CTC/attention loss for a Conformer comparison. Language-ID uses class-balanced cross entropy or focal loss because switch tokens are sparse. Boundary loss operates on adjacent contextual states:

\[
p(B_t=1\mid X)=\sigma\left(W_b[h_{t-1};h_t;h_t-h_{t-1};h_{t-1}\odot h_t]+b_b\right).
\]

Do not set the loss weights by intuition alone. Compare uncertainty weighting, GradNorm, and a small validation search; retain the simplest method whose gains survive speaker-level confidence intervals.

## 3. Uncertainty decomposition

### 3.1 Aleatoric evidence

For token distribution (p_t(v)), compute normalized entropy and margin:

\[
H_t=-\frac{1}{\log |V|}\sum_{v\in V}p_t(v)\log p_t(v),
\qquad
m_t=p_t^{(1)}-p_t^{(2)}.
\]

Aggregate subwords to a word with learned pooling or conservative statistics such as minimum confidence, mean negative log probability, and maximum entropy. Preserve the individual features; a product is too length-sensitive to be the only aggregator.

### 3.2 Epistemic evidence

Use (M) independently seeded adapters, dropout samples, or repeated decoding passes only on selected risky spans. For predictive samples (p^{(m)}(y\mid X)), estimate mutual information:

\[
I[y,\theta\mid X]
= H\!\left[\frac{1}{M}\sum_m p^{(m)}\right]
- \frac{1}{M}\sum_m H[p^{(m)}].
\]

Because full vocabulary distributions are not always exposed, the production approximation also uses N-best edit disagreement, timestamp variance, and pairwise normalized Levenshtein distance.

### 3.3 Switch-local uncertainty

For word (t), define boundary proximity risk over radius (r):

\[
u^{switch}_t = \max_{j\in[t-r,t+r]} H_2(p(B_j=1\mid X)),
\]

where (H_2) is binary entropy. Test whether this feature adds information after controlling for word frequency, acoustic quality, entity type, and token length. Otherwise the project would confuse ordinary rare-word uncertainty with language-switch uncertainty.

## 4. Confidence estimation and calibration

Align hypothesis words to the reference with a deterministic edit path. Let (z_t=1) when the word is correct under the declared normalization and (0) otherwise. A confidence estimator predicts:

\[
q_t=P(z_t=1\mid \phi_t),
\]

where (\phi_t) contains decoder, acoustic, language, boundary, disagreement, duration, entity, and context features.

Train a compact estimator with weighted binary cross entropy or a continuous lexical-similarity target. Fit calibration only on a held-out speaker-disjoint calibration set.

Temperature scaling chooses:

\[
T^*=\arg\min_{T>0}-\sum_i \log \operatorname{softmax}(z_i/T)_{y_i}.
\]

Compare temperature scaling, isotonic regression, and beta calibration. A method is selected by held-out NLL/Brier score and risk-coverage behavior, not ECE alone. Report adaptive-bin ECE and maximum calibration error because fixed-bin ECE can hide local failures.

## 5. Candidate-lattice entity preservation

The entity guard does not trust only the top transcript. Construct a graph whose nodes are time-aligned candidate words and whose weighted edges encode temporal compatibility, decoder probability, phonetic similarity, script/transliteration similarity, and language compatibility.

For entity value (v), marginalize evidence across hypotheses:

\[
P(E=v\mid X) \approx \sum_{h\in\mathcal{H}}
P(E=v\mid h,X)\,\tilde P(h\mid X).
\]

Entity candidates combine:

1. deterministic finite-state recognizers for dates, times, dosages, quantities, money, phone numbers, and identifiers;
2. multilingual span recognition for names, organizations, locations, medicines, and configurable domain types;
3. lexicon retrieval with phonetic and transliteration-aware similarity;
4. acoustic/decoder alternatives within the same time span.

The original surface form is immutable. Canonicalization is a separate function (g(v, k, locale)) with validation and provenance. If two plausible critical values remain, the system returns both instead of allowing a language model to invent a resolution.

## 6. Cost-sensitive selective prediction

Let (c) be the latent outcome class, such as correct, ordinary error, critical substitution, critical deletion, or hallucination. Choose the action that minimizes posterior expected cost:

\[
a^*(x)=\arg\min_a \sum_c C_{a,c}P(c\mid x).
\]

The cost matrix assigns the highest loss to accepting a wrong critical entity, a smaller cost to unnecessary review, and a latency/accessibility cost to deferral. Costs must be validated with target users and domain experts; they are product assumptions, not universal constants.

For risk threshold (\tau):

\[
\operatorname{coverage}(\tau)=P(r(X)\le\tau),
\qquad
\operatorname{risk}(\tau)=E[\ell(Y,\hat Y)\mid r(X)\le\tau].
\]

Select thresholds by minimizing expected cost subject to hard constraints on critical-entity recall, subgroup risk, p95 latency, and maximum review burden. Report the full risk-coverage curve and AURC so a favorable single threshold cannot hide poor behavior elsewhere.

## 7. Conformal candidate sets

For suitable exchangeable calibration data, use nonconformity (s_i=1-p(y_i\mid x_i)). With (n) calibration examples and target error (\alpha), use the finite-sample quantile:

\[
\hat q = \operatorname{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}(s_1,\ldots,s_n)
\]

and return (\Gamma(x)=\{y:1-p(y\mid x)\le\hat q\}). Evaluate coverage separately by language pair, switch proximity, and entity type. If distribution shift or small conditional slices break the assumptions, label the guarantee invalid and fall back to empirical calibration; never market a marginal guarantee as per-group safety.

## 8. Budgeted inference cascade

Real-world inference must spend compute where risk is concentrated:

1. **Fast pass:** VAD, primary ASR, token alignment, lightweight LID, and structured entity rules.
2. **Risk gate:** compute calibrated word/boundary/entity risk.
3. **Adaptive pass:** only for medium/high-risk spans, request wider beam, repeated decoding, second-model evidence, or lexicon retrieval.
4. **Decision:** accept safe spans; review the smallest high-value span; defer when evidence is insufficient.

Let (d_i\in\{0,1\}) enable the expensive pass for span (i). Optimize:

\[
\min_d \sum_i \big(R_i(d_i)+\lambda_{lat}L_i(d_i)+\lambda_{review}Q_i(d_i)\big)
\quad \text{s.t.}\quad
P95(L)\le L_{max},\; Recall_{critical}\ge r_{min}.
\]

This cascade is more deployable than always running an ensemble and gives a measurable accuracy/latency trade-off.

## 9. Data selection and augmentation

Prioritize annotation with a diversity-aware active-learning score:

\[
A(x)=\big(\alpha U_{asr}+\beta U_{switch}+\gamma U_{entity}\big)
\times D(x) \times W_{slice}(x),
\]

where (D) rewards acoustic/speaker diversity and (W_{slice}) increases coverage of underrepresented but declared slices. Cap slice weights to avoid selecting only pathological samples.

Augmentation must preserve labels. Use room/noise/reverberation and speed perturbation with timestamp updates; linguistically constrained synthetic switching; transliteration and entity substitution with consistent audio/text generation only when provenance is explicit. Never place synthetic speakers in the locked real-speech test set.

## 10. Fairness and robustness

Evaluate overall risk and worst-declared-group risk. A research comparison can add a distributionally robust term:

\[
\min_\theta \max_{g\in G} R_g(\theta)+\lambda R_{overall}(\theta),
\]

but do not optimize on sensitive groups without adequate sample size and governance. Use speaker-level paired bootstrap intervals, noise/device stress tests, out-of-domain shifts, and calibration drift monitoring.

## 11. Real-user validation

Offline metrics are necessary but insufficient. The pilot study measures task completion, correction time, confirmation burden, trust calibration, keyboard/screen-reader completion, and whether users understand why a span needs review. The key product outcome is fewer harmful accepted errors at an acceptable interaction cost—not merely a more colorful transcript.
