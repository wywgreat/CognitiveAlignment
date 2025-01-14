Prepare Data: Ensure BBSR dataset and query files are in the data/ directory.
Generate Predictions: Use get_GPT4om_BBSR.py to query LLMs and generate prediction files.
Analyze Results: Run BBSR_correlation_analysis.py to compute alignment metrics.
Visualize: Use BBSR_analysis_toshow.py to generate all figures and tables in the paper.

Notes: 
Qwen2 and Llama-3 series: These models are called locally as open-source models.
GPT-3.5 Turbo and GPT-4o mini: These models are accessed via API calls, following a similar procedure.