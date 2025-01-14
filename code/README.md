# README for Cognitive Alignment Between Humans and LLMs Across Multimodal Domains

## Overview

This repository contains the source code, scripts, and data used for conducting the study on cognitive alignment between large language models (LLMs) and human semantic representations using the Brain-Based Semantic Representations (BBSR) dataset. The study assesses LLMs on multisensory profiles, consistency, stability, and conceptual proximity.

## System Requirements

*   **Operating System**: Tested on Windows 10, macOS 12, and Ubuntu 20.04.

*   **Python Version**: 3.9.0 or later

*   **Dependencies**:

    *   `pandas`

    *   `numpy`

    *   `matplotlib`

    *   `seaborn`

    *   `scipy`

    *   `sklearn`

    *   `openai` (for GPT models via API)

    *   Additional utilities like `pingouin` for ICC calculations.

## Installation Guide

1.  Clone this repository:

        git clone https://github.com/wywgreat/CognitiveAlignment
        cd CognitiveAlignment

2.  Install required Python packages:

        pip install -r requirements.txt

3.  Place the BBSR dataset and query files in the `data/` directory. Ensure the following files are present:

    *   `[BBSR]WordSet1_Ratings.xlsx`

    *   `[BBSR]Queries_v4.xlsx`

## Data Collection

The LLMs tested in this study include:

*   **Open-Source Models**: Qwen2 and Llama-3 series, tested locally using the provided scripts.

*   **API-Accessed Models**: GPT-3.5 Turbo and GPT-4o mini, queried via OpenAI API.

*   **Script**: `get_GPT4om_BBSR.py`

    *   Queries LLMs using predefined prompts from the BBSR dataset.

    *   Outputs prediction files in the `results/` directory.

*

## Data Analysis

*   **Script**: `BBSR_correlation_analysis.py`

    *   Computes alignment metrics such as correlation, MSE, and KL divergence.

    *   Results are stored in the `results/` directory.

*   **Script**: `BBSR_analysis_toshow.py`

    *   Generates figures and tables used in the manuscript.

    *   Outputs visualizations in the `results/BBSR_FINAL_toshow/` directory.

## Instructions for Use

1.  **Prepare Data**:

    *   Ensure all required data files are located in the `data/` directory.

2.  **Generate Predictions**:

    *   Run `get_GPT4om_BBSR.py` to query LLMs and save prediction outputs.

3.  **Analyze Results**:

    *   Execute `BBSR_correlation_analysis.py` to compute alignment metrics.

4.  **Visualize Results**:

    *   Use `BBSR_analysis_toshow.py` to generate visualizations for the study.

## Demo

1.  **Run Demo**:

    *   Execute the following:

            python get_GPT4om_BBSR.py 
            python BBSR_correlation_analysis.py 
            python BBSR_analysis_toshow.pybash



2.  **Expected Output**:

    *   Prediction files in `results/`

    *   Correlation analysis and summary tables in `results/BBSR_FINAL_toshow/`

    *   Visualization files in `results/BBSR_FINAL_toshow/MultisensoryProfile/`

3.  **Expected Runtime**:

    *   Depends on the model and dataset size. Estimated:

        *   Local models: \~5 hours

        *   API models: \~2 hours (may vary with API response times).

## Additional Notes

*   For API-based models (GPT-3.5 Turbo and GPT-4o mini), ensure valid API keys are provided in the scripts.

*   The data processing and analysis are conducted using Python scripts, leveraging the free version of Python 3.9.





