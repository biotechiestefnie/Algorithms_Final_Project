#### Final Reflection of:

## The Memoization Goldilocks Zone: Investigating Hyperparameter Boundaries in Markov Models for Classifying Structural Features in the Human Genome.

#### Project Accomplishments

**Technical and Process-Related Successes:**

- Early choices around modular design paid off: separating data
  preparation, model training, threshold calibration, classification,
  and evaluation into distinct, testable components made debugging far
  more efficient and ensured that improvements in one area did not
  destabilize others

- The decision to freeze the final‑run pipeline and enforce a canonical
  directory structure also strengthened reproducibility and prevented
  accidental contamination or re‑sampling

- Using a balanced sampling strategy and applying biologically informed
  length thresholds helped stabilize the statistical behavior of the
  Markov models, particularly at lower k values where signal is more
  diffuse

- The prototype‑run phase—where incremental corrections and
  optimizations were introduced proved invaluable for validating
  assumptions, refining edge‑case handling, and confirming that the
  models behaved consistently across classes

Collectively, these technical choices created a robust, traceable
workflow that held up under final‑run conditions

**Personal Learning Successes**

This project represented a significant learning milestone for me.
Working end‑to‑end through data extraction, preprocessing, model design,
evaluation, and documentation deepened my understanding of how
computational pipelines behave in practice and not just in theory. I
gained a clearer appreciation for the importance of reproducibility,
especially how small inconsistencies in data handling or directory
structure can cascade into major analytical errors.

I also learned how to reason about model behavior in a more principled
way, particularly when interpreting log‑likelihood distributions,
threshold effects, and the impact of biological variability on
classification performance. Perhaps most importantly, I became more
confident in my ability to diagnose issues, justify design decisions,
and articulate the rationale behind methodological choices and strategic
modifications. These skills will carry forward into my professional
endeavors, where clarity, rigor, and reproducibility will undoubtedly be
absolute requisites.

#### Complications and Setbacks

**What Went Wrong or Was Notably Difficult**

- Underestimating the complexity and scale of working with four genomic
  classes simultaneousl: promoters, exons, introns, and repeats each
  come with their own biological diversity, annotation quirks, and
  internal subtype variation, and managing all four at once quickly
  became overwhelming for the scope and timeframe of the study

- Extracting raw exon and intron sequences was also unexpectedly
  difficult, because these features are no longer provided as standalone
  FASTA files that can be accessed from any database. I had to improvise
  a custom extraction workflow using GENCODE annotations and gffread to
  reconstruct introns from exon coordinates

- Early model behavior revealed that incorrect classifications were
  ambiguous— there was no way to tell whether the model genuinely
  believed a sequence belonged to the wrong class or whether it simply
  rejected all other classes. Consequently, I instated a rejection
  option for the model to distinguish between “true misclassification”
  and “none of the above,” which despite being essential, added another
  layer of complexity to the pipeline

**What Would Be Different if Starting Over**

- I would limit the study to two or three genomic features rather than
  four. The full four‑class design introduced far more biological
  heterogeneity, data imbalance, and computational overhead than
  initially anticipated, and narrowing the scope would have allowed for
  deeper exploration of model behavior without being overwhelmed by
  scale

- I would also build the rejection mechanism into the design from the
  beginning rather than retrofitting it mid‑pipeline, since it
  fundamentally changes how classification decisions are interpreted

- I would plan a more streamlined strategy for obtaining raw exon and
  intron sequences, either by pre‑selecting a dataset with readily
  available FASTA files or by writing a dedicated extraction script
  earlier in the process so the data preparation phase did not detract
  focus from the model workflow

**Personal Learning Obstacles**

- Realizing how quickly small design decisions can cascade into large
  analytical consequences: early in the project, I struggled with
  interpreting model outputs because I had not yet internalized how
  generative classifiers behave when faced with sparse data, short
  sequences, or ambiguous signals. Implementing the rejection option
  forced me to confront these issues directly and develop a more
  principled understanding of log‑likelihood distributions and
  thresholding.

- Navigating the practical challenges of genomic data extraction. This
  required more command‑line use and annotation literacy than I
  initially expected. I addressed this by refining my extraction
  workflow, validating intermediate outputs, and learning to rely on
  documentation and community resources when the process became too
  complicated for me.

These experiences ultimately strengthened my confidence in debugging,
data handling, and methodological reasoning, even when the path forward
was not immediately clear. They also gave me tools to strategically
mitigate each obstacle and adapt my approach rather than forfeiting the
study

#### Algorithmic Lessons

Working with a generative Markov classifier in a real biological context
gave me a much clearer sense of how well this algorithm fits the
structure of the problem. For short, motif rich genomic features like
promoters and certain repeat families, the model aligns surprisingly
well with the underlying biology because these regions express strong
local sequence dependencies that a k order Markov chain can capture.
However, the tradeoff becomes obvious as soon as the classes become more
heterogeneous. Accuracy improves with higher k, but the computational
and statistical complexity grows exponentially, and sparsity quickly
becomes the dominant force shaping model behavior.

Implementing the classifier also revealed nuances that were not fully
apparent in lecture, particularly how sensitive generative models are to
data imbalance, annotation noise, and within class variability. One of
the biggest surprises was how often the model produced confident
misclassifications simply because it rejected all other classes, which
ultimately forced me to implement a rejection option to distinguish
genuine classifications from default assignment. Overall, the algorithm
fits the problem with much greater reliability than I initially expected
to see for successful predictions based on such short recognition
patterns. Prior to this study, I would never had believed that such a
simply implemented algorithmic model could achieve such successful
classifications with so little prior context to learn a genomic
feature’s inherent patterns from.

#### Future Directions

There are several viable possibilities for expanding and improving this
project beyond its current scope:

- Exploring algorithmic variants that capture richer dependencies than
  fixed‑order Markov chains, such as variable‑order Markov models,
  hidden Markov models, or even lightweight neural architectures that
  can learn longer‑range patterns without exploding in complexity

- Improving the data would also strengthen the analysis, particularly by
  incorporating subtype‑specific annotations for promoters, exons,
  introns, and repeats, increasing the sample sizes, or by using more
  recent genome assemblies with expanded experimental validation

- Additional evaluation methods could provide deeper insight into model
  behavior, including cross‑validation, calibration curves, per‑subtype
  confusion matrices, and likelihood‑ratio diagnostics to quantify how
  confidently the model distinguishes between classes

- The classifier could be integrated into a larger genomic analysis
  pipeline, such as a preprocessing module for gene prediction, repeat
  masking, or regulatory element discovery, where a rejection option and
  probabilistic scoring could help flag ambiguous or novel sequences for
  downstream analysis

#### Generative AI Disclosure

Microsoft CoPilot was solicited for assistance using the following
prompts:

- Can you help me sanity‑check my understanding of how higher‑order
  Markov models handle sparsity and backoff? I want to make sure my
  interpretation aligns with the underlying mathematical concepts before
  I finalize my write‑up and justify my observations

- I’ve already implemented a Markov Chain approach over varying k values
  for classifying canonical genomic features, but I want to explore
  possible extensions or alternative approaches. What are some options
  that could capture longer‑range dependencies without exploding in
  complexity or requiring unreasonable amounts of data?

- I wrote a section in my README explaining the limitations of my
  classifier, but it is verbose and unfocused. Can you help me refine
  the wording to make it more concise and relevant while still
  addressing each limitation?
