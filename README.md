# Amazon Gourmet‑Foods Sentiment Analysis with Apache Spark
###  Overview
This assignment demonstrates large‑scale **sentiment analysis** on Amazon Gourmet‑Foods reviews using **Apache Spark** for parallel map‑reduce processing and the **Hugging Face Transformers** pipeline for inference.

| Task | Description | Key Output |
|------|-------------|------------|
| **Task 1** | Parse raw `Gourmet_Foods.txt`, classify each review as **POSITIVE** or **NEGATIVE** with `distilbert‑base‑uncased‑finetuned‑sst‑2` | `sentiment_output/` directory containing partitioned CSVs with predicted labels |
| **Task 2** | Derive ground‑truth labels from star ratings (≥ 3 → POSITIVE), compute confusion matrix, Precision & Recall, and plot results | Console metrics + matplotlib heat‑map and bar chart |

---

### Repository Structure
```
.
├── task1.py              # Distributed sentiment inference
├── task2.ipynb           # Evaluation & visualisation
├── Gourmet_Foods.txt     # Raw review dump (key‑value format)
├── sentiment_output/     # Created by task1.py (Spark part‑*.csv files)
└── README.md             
```
---

### Running the Pipeline
#### Task 1 — Sentiment Inference
Inside `task1.py` these lines should be adjusted accordingly
````
os.environ["OMP_NUM_THREADS"] = "10" # Should be adjusted based on CPU cores
os.environ["PYSPARK_PYTHON"] = "/home/amar/Desktop/2nd_sem/MLOps/bin/python" # Path to your Python environment
````
```bash
python task1.py
```
- Parses every review.
- Broadcasts the HF pipeline to Spark workers.
- Writes predictions to sentiment_output/part-*.csv and a _SUCCESS flag.

#### Task 2 — Evaluation & Visualisation Notebook
```bash
python task2.py
```
- Reads sentiment_output/.
- Sets ground‑truth: rating ≥ 3 → POSITIVE; otherwise NEGATIVE.
- Aggregates TP, FP, TN, FN via Spark.
- Prints Precision & Recall, then opens two plots:
- Confusion‑matrix heat‑map
- Precision/Recall bar chart

---
### Key Implementation Details

| **Aspect**     | **Approach** |
|----------------|--------------|
| **Parsing**    | Custom function splits `Gourmet_Foods.txt` on blank lines, extracts `review/score` & `review/text`. |
| **Model**      | `transformers.pipeline("sentiment-analysis")` default DistilBERT fine-tuned on SST-2. |
| **Map-Reduce** | Spark UDF for per-row inference (map); `groupBy().sum()` for counts (reduce). |
| **Scalability**| Spark shuffle/parallelism limited to 4 partitions for laptop stability; remove limits on a cluster. |
| **Metrics**    | Precision = TP / (TP + FP) |
| **Visuals**    | Matplotlib heat-map (Blues) with annotated cells; separate bar chart for scalar metrics. |



