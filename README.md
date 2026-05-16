# Algorithms Final Project

Project Title:

# The Memoization Goldilocks Zone: Investigating Hyperparameter Boundaries in Markov Models for Classifying Structural Features in the Human Genome

## **Research Question:**

How much Markov “memory” is required for a model to faithfully
distinguish among promoters, coding regions (exons), noncoding regions
(introns), and repetitive elements in the human genome? Understanding
the relationship between structural signal complexity and Markov memory
is both biologically meaningful and algorithmically valuable. By
systematically varying the Markov order, we can gain insight into:

- How easily different genomic features can be detected using
  sequence‑based statistical models

- Where the model begins to break down for each structural class

- How increasing memoization influences overfitting versus
  overgeneralization

- Which biological signals are inherently local and which require
  long‑range contextual information:

  - A structural class is local if the information needed for
    recognition is contained in short, contiguous patterns — the kind of
    signal a low‑order Markov chain (small 𝑘) can reliably capture.
  - In contrast, a class is long‑range if its defining features are
    dispersed or extend beyond a few nucleotides, requiring higher
    Markov memory (larger k) for accurate classification predictions

### **The Algorithm**

The successful completion of numerous genome sequencing projects, paired
with considerable advancements in Next Generation Sequencing (NGS)
technologies, has generated an unprecedented wealth of genomic data,
providing a rich substrate for scientific advancements among the omics
disciplines. Markov chains form a fundamental class of probabilistic
models that have become indispensible tools for investigating a wide
range of biological unknowns, from gene prediction to protein structual
analysis. These models are underpinned by a robust mathematical
framework that can be exploited to capture the dynamic dependencies
between adjacent nucleotides while simultaneously simulating their
structural characteristics (Ma et al., 2025), making them inherently
well-suited for extracting meaningful insights from complex sequence
datasets.

A key consideration of Markov applications is the Markov property, which
asserts that the next state depends only on a finite number of previous
states. While this assumption greatly simplifies computational
complexity, it also imposes a “memoryless” constraint that is
exceedingly limiting for investigating biological sequences, which
contain countless motifs, periodicities, and regulatory signals, many
extending well beyond a single nucleotide of context (Ryabco &
Usotskaya, 2008). To address this limitation, higher‑order Markov models
incorporate a longer history by conditioning on the previous k
nucleotides. And while increasing k allows the model to uncover richer
local dependencies and more complex biologically meaningful patterns, it
also expands the state space exponentially (4^k), requiring
substantially more data to accurately estimate transition probabilities,
in addition to rendering an alltogether insurmountable computational
burden (Burkes & Azad, 2020).

This trade‑off between contextual richness, data sparsity, and memory
bottleneck makes the choice of k a critical strategic decision. If k is
too small, the model fails to capture important structural features; if
k is too large, the model becomes unstable, overfits, collapses due to
insufficient observations, or freezes due to computational overload.
Understanding how Markov memory interacts with the complexity of
biological signals is therefore essential for determining when these
models are applicable, where they break down, and how much context is
required for reliable structural discrimination in the human genome.

For this study, I implement a deterministic k‑order Markov chain model
as a multi‑class generative classifier, training a separate model for
each canonical genomic feature. In this formulation, nucleotides A, T,
C, and G constitute the basic alphabet, and the effective states of a
k‑order model correspond to all possible kmers. Each model learns the
class‑specific transition probabilities governing how nucleotides follow
one another within these kmer constructs that vary in size according to
k. Each model estimates the likelihood of a sequence by evaluating the
conditional probability of a nucleotide given the previous k nucleotides
and accumulating these probabilities in logspace. By systematically
increasing the Markov order and evaluating classification performance
across promoters, introns, exons, and repeats, this study directly
measures how much contextual memory is required for faithful
classification and identifies the precise moment at which the models for
each class begin to fail and why. These insights can help pave the way
for more successful applications of Markov Chain models in genomics,
informing the design of more sophisticated models that balance context
and complexity to maximize reliability.

### **Predictions, Assumptions, & Limitations**

- As k increases, I presume that each class model will more reliably
  pick up on the characteristic structure of each corresponding
  biological signal:

  - Promoters, which are relatively short and enriched for motifs like
    TATA boxes, CpG‑dense regions, and transcription factor binding
    patterns, should benefit up to a moderate k: small k values (2–3)
    will mostly capture GC bias and simple dinucleotide structure, while
    mid‑range k (4–6) should begin to reflect motif‑like patterns;
    beyond that, the model may start overfitting specific promoter
    instances rather than general promoter architecture

  - Exons, with their codon structure and reading‑frame constraints,
    should show progressively stronger discrimination as k increases
    into the 3–6 range, where codon usage, periodicity, and
    splice‑proximal patterns become more pronounced; very large k may
    again overfit individual exon sequences rather than capturing
    general coding structure

  - Introns, which are compositionally looser but still contain splice
    site motifs and weak local biases, will likely show modest
    improvements at low to mid k (2–4) from capturing splice junction
    signals and subtle compositional structure, but they are unlikely to
    benefit dramatically from large k because their long‑range
    organization is less pronounced

  - Repeats, especially tandem or interspersed elements with strong
    internal periodicity, may show the most dramatic improvement with
    increasing k: small k captures only base composition, but larger k
    (5 and beyond, depending on repeat length) can lock onto the
    repeated unit structure, producing sharply separated log‑likelihood
    distributions as k grows, though at the cost of becoming highly
    specific to the particular repeat families present in training, as
    well as risking computational overload

- Because all sequences were accessed from highly regarded databases
  with experimentally validated class identifications, I expect
  classification accuracies to fall well above what would be achievable
  by chance alone. That said, each structural feature included in this
  study contains numerous biological subtypes, each with its own
  characteristic signal patterns. Even though these sequences share the
  same high‑level structural identity, the internal diversity of their
  motif composition, periodicity, and local sequence organization
  introduces substantial within‑class variability. This variability can
  weaken classification confidence by broadening the true‑class
  log‑likelihood distribution, reducing separation from other classes,
  and increasing the likelihood that atypical or subtype‑specific
  sequences fall near or below the rejection threshold

- A major limitation of this study is the increasing sparsity of kmer
  observations as k grows. Because possible kmers scales exponentially
  with increasing values of k (4^k), higher‑order Markov models require
  disproportionately more data to estimate transition probabilities
  reliably. Even with thousands of sequences per class, many
  higher‑order k‑mers will never be observed, forcing the model to rely
  heavily on smoothing and backoff behavior. This sparsity broadens the
  log‑likelihood distribution for true‑class sequences, reduces
  separation between classes, and increases the likelihood that rare or
  subtype‑specific sequences fall below the rejection threshold. As a
  result, classification performance at high k values may reflect data
  sparsity rather than true biological signal.

- In a k‑order Markov model, backoff behavior refers to what the model
  does when it encounters a kmer that was never observed in the training
  data. Instead of assigning a probability of zero (which would collapse
  the entire log‑likelihood to −∞), the model “backs off” to a
  lower‑order estimate- for example, using (k‑1)mer or unigram
  frequencies to approximate the missing transition. Backoff is
  essential at higher k values because the number of possible kmers
  grows exponentially, and even large datasets cannot cover the full
  space. However, this behavior is not demonstrative of the true
  biological signals at the actual k value being tested, and if frequent
  enough, can result in unrealistic classifications from the model which
  would not be discernable

- Sequence length introduces another important limitation. Because
  log‑likelihood is computed as the sum of all kmer transitions, longer
  sequences naturally accumulate more evidence (both positive and
  negative) than shorter ones. Very short sequences may contain too few
  kmers to express the characteristic patterns of their class, causing
  unstable or weak likelihood estimates. Conversely, very long sequences
  can dominate the scoring process, amplifying even subtle compositional
  biases. Although length thresholds were applied to minimize
  distortion, the inherent variability in biological sequence lengths-
  particularly for exons, introns, and repeats- still influences
  classification confidence and may contribute to borderline or
  misclassified cases

- This study also assumes that all sequences extracted from the
  reference databases are correctly annotated. Any misannotations in
  exon–intron boundaries, promoter regions, or repeat classifications
  would directly propagate into the training labels and distort the
  learned models. Because the classifier is generative and highly
  sensitive to the statistical properties of each class, even a small
  number of mislabeled sequences could shift the estimated transition
  probabilities and weaken class separation. To mitigate this risk, I
  extracted all raw sequences from the *H. sapiens* hg19/GRCh37 assembly
  as hosted by the UCSC Genome Browser. This particular collection
  provides experimentally validated annotations and has been publicly
  available for more than a decade, offering a long history of
  documented use and high confidence in canonical features such as
  promoters, introns, exons, and repeats.

### Requirements Overview

*System Requirements*

This project requires a standard Python scientific computing environment
with the following dependencies:

- Python Version:
  - Python 3.10 or later
- Python Packages:
  - NumPy — numerical operations and array handling
  - Matplotlib — visualization of class separation and score
    distributions
  - Biopython — FASTA parsing and sequence handling
  - tqdm — progress bars for long operations
  - pathlib — filesystem path management (standard library)
  - random, math, statistics — standard library utilities used
    throughout the pipeline

All required packages are listed in requirements.txt and can be
installed within the IDE terminal using:

    pip3 install -r requirements.txt

*Data Requirements*

- Training and testing FASTA files generated by the data preparation
  notebook for the following H. sapien hg19/GRCh37 assembly
  experimentally validated canonical genomic features: promoters, exons,
  introns, and repeat regions

- Each sequence must include a class label in the FASTA header

- The unified train/test files must be preserved to maintain
  reproducibility and prevent resampling or contamination

- For detailed instructions regarding locations from which raw datafiles
  were extracted, please see DATA.md

*Environment Requirements*

- All scripts have been written to run exclusively within the
  implementation notebooks included in the top-level notebooks/
  directory. While runnning the scripts individually from a local
  terminal is technically possible, this would require either inputting
  the entirety of a call in the terminal, or modifying the scripts
  themselves to manually reproduce the full workflow, output structure,
  and visualization pipelines documented in the notebook files. In
  practice, this involves rewriting large portions of the driver logic,
  restructuring return objects, and adding additional I/O handling to
  generate results in the same format produced automatically within the
  notebooks. For this reason, the notebooks serve as the intended and
  most efficient execution environment for the complete data preparation
  and model pipelines.

- Any Integrated Development Environment (IDE) compatible with Python,
  Jupyter notebooks, and standard text‑based formats (FASTA, CSV,
  Markdown, TXT, JSON) can be used to access and inspect the files in
  this project. The development and analysis for this study were
  conducted using PyCharm, RStudio, Cursor, and the Python IDLE Shell
  3.11.9 (v3.11.9:de54cf5be3).

- To extract exon and intron raw sequences from the hg19/GRCh37 genome
  assembly, a clean virtual environment was created using Miniconda on
  the local terminal. GENCODE v19 annotation files (GFF/GTF) and the
  corresponding GRCh37.p13 reference genome FASTA were downloaded, and
  gffread was installed within this isolated environment to identify
  exons and infer introns for extraction. This environment was kept
  minimal and used exclusively for sequence extraction to avoid
  interference from other packages or system‑level configurations.

### QuickStart Instructions

To run this program, the user must first install Python 3.10 or later.
The following packages must also be installed for scripts to run
properly: Bio and Numpy. Please see REQUIREMENTS.txt for full list of
required packages and transitive dependencies.

**QuickStart**

    The user must first prepare the datafiles to be used in the implementation. To do this, they must open the following implementation notebook: *notebooks/data_preparation.ipynb*

    From the Data Preparation notebook, the user can perform all tasks involved in extracting and preparing FASTA datafiles used to train and test the model for both the prototype and final runs.

    **Data Preparation Steps:**
    Running the cells of this notebook in order of appearance, the user will accomplish the following:
        
        - Import packages and modules
        
        - Set working directory to root and add src/ and notebook/ to Python path

    Prototype Run Data Preprocessing:

        - infer class type from filepath
        
        - load raw FASTA into a dictionary {seq_id: SeqRecord}
        
        - count raw sequences for each class
        
        - filter exons/introns for valid lengths (promoters/repeats unchanged)
        
        - sample positive sequences (k=300)
        
        - rewrite headers for positive sequences
        
        - write positive FASTA file
        
        - remove sampled positives from dictionary
        
        - sample negative candidates from remaining sequences
        
        - apply dinucleotide shuffling to destroy biological signal
        
        - rewrite headers for negative sequences
        
        - write negative FASTA file
        
        - compute preliminary statistics for all positive and negative datasets
        
        - write summary CSV of prototype dataset characteristics

    Final Run Data Preprocessing:
        
        - extract sequences from raw datafiles for each class
        
        - run preliminary statistics on each class to identify threshold boundaries
        if needed
        
        - filter exons, introns, and repeats by length
        
        - combine files for all classes and split into 80/20 training/testing sets
        
        - shuffle test dataset to randomize sequences
        
        - write training and testing sequences to respective files
        
    **Notice!** Please note that the prototype calls and the final run calls inside the data preparation notebook can be called independently and running the prototype cell is not required before running the final run cells. 

    Once datafiles have been prepared, the model can be trained, tested, and evaluated to determine the accuracy of the model's classifications for four canonical genomic structures: promoters, exons, introns, and repeats. While this can be performed from either the Prototype or Final notebooks, this instruction will explicitely refer to the steps inside the Final notebook only for brevity's sake. The user accesses this notebook at: 
    *notebooks/final_implementation.ipynb*

    **Final Implementation Steps:**
    Running the cells of this notebook in order of appearance, the user will accomplish the following:
        
        - Import packages and modules required for successful end-to-end implementation of the programs
        
        - Set working directory to root and add src/ and notebook/ to path
        
        - define final run FASTA paths to train and test file as input data
        
        - load training sequences from data/final/train/train.fa into dictionary with data_loading.py script
        
        - Define k values for training and run train_models.py script to train Markov model with class datasets containing
        80% of sequences, experimentally validated as belonging to respective classes so model can learn signal patterns
        unique to each class
        
        - Apply threshold_utils.py to take the trained model for each class and score that model’s own training sequences
        under itself. This produces a distribution of “true‑class” log‑likelihood values, from which the 5–10th (or
        90–95th) percentile boundary is used as a rejection threshold: sequences scoring below this value are rejected, and
        sequences scoring above it are eligible for classification. From the prototype run, thresholds in the 5–10% range
        were found to be optimal, whereas a threshold set at the 1st/99th percentile was too permissive and resulted in an
        excessive number of false positives
        
        - prepare test sequences and true class labels for testing model. Test sequences consist of 20% total sequences
        from master file, after filtering. Preparation includes removing labels from sequences and placing sequences in
        list format for model, labels in dictionary to append as classification with data_loading.py script
        
        - pass test sequences, k values, and threshold values to model to make predictions: promoter, exon, intron,
        repeats, or rejection. Return true label, prediction, and log likelihood score, which has been normalized per-base
        to minimize length distortion. This is all completed by running model_classifications.py, which calls upon the
        per_base_likelihoods.py script internally to calculate and return scores
        
        - display classification results in notebook in tabular format
        
        - apply statistical evaluations to results to interpret and derive meaning from predictions and per-base log likelihood scores with evaluate_results.py. This script calculates per-class metrics, confusion matrix, overall accuracy, log likelihood summarizations, 
        
        - plot results with plot_results.py script to visualize accuracy distributions across classes and k values

**Note!** Please refer to the implementation notebooks for detailed,
step‑by‑step execution calls, supporting explanations, and all generated
outputs and interpretations. During the Prototype run, incremental
modifications were introduced as needed to correct behavior, validate
assumptions, or optimize execution. For the Final run, however, all
scripts, parameters, and methodological choices were held constant; no
further code changes were made, and all analysis focused exclusively on
evaluating results and interpreting model performance

### **Configuration Parameters**

This study relies on several configuration parameters that directly
influence model behavior and must be treated as fixed experimental
values These include:

*Selected k orders tested (k = 1, 2, 3, 4)*  
- These determine the k‑mer size and directly influence sparsity,
transition probability estimation, and class separation

*Sample size of each class*  
- Sequence availability varies by feature type (e.g., promoters returned
only 1,796 raw sequences), which constrains the maximum balanced sample
size and affects the stability of transition estimates

*Minimum and maximum sequence‑length thresholds*  
- Applied during data preparation to filter exons, introns, and repeats;
these thresholds influence the distribution of k‑mer counts and the
reliability of log‑likelihood scoring

*Train/test split proportion (80/20)*  
- This fixed split ensures reproducibility but also determines how much
data is available for training versus evaluation

*Percentile‑based rejection thresholds (5–10%)*  
- These thresholds define the boundary for rejecting low‑likelihood
sequences and strongly influence classification confidence and
false‑positive rates

*Directory structure expected by the pipeline*  
- All paths are interpreted relative to the working directory, which is
treated as the project root; deviations from this structure will break
the workflow

*All sequences derived from the same GRCh37.p13 reference genome*  
- This ensures consistency across classes but assumes that the reference
assembly and its annotations are correct and representative

*Lack of structural feature subtype identifications*  
- Subtypes within promoters, exons, introns, and repeats are not labeled
separately, meaning the model must learn a single statistical profile
per class despite substantial internal heterogeneity

Although these parameters are not stored in a standalone configuration
file, they function as global settings that shape every stage of the
workflow. Any deviation from these values would change the learned
transition probabilities, the resulting log‑likelihood distributions,
and ultimately the classification performance. For reproducibility, all
parameters were held constant during the final run, but their fixed
nature underscores several inherent limitations of the study
