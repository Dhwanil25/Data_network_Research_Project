# Symbolic Dynamics of LLM Temperature Sampling

A collaborative research project investigating how the temperature hyperparameter in Large Language Models (LLMs) controls randomness and structure in generated token sequences, using the classic logistic map as a theoretical baseline.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-yellow.svg)](https://huggingface.co/transformers/)

## 👥 Research Team

**Team Members**: Sanjana Kadambe, Jasreen Mehta, and Dhwanil Mori

**Advisor**: Dr. Neil Johnson, Professor at George Washington University

## 🎯 Project Overview

This research explores whether LLM temperature behaves analogously to the logistic map's r-parameter, investigating if increasing temperature produces a period-doubling route to chaos similar to deterministic dynamical systems.

### Core Research Question

**Does LLM temperature sampling exhibit symbolic dynamics comparable to deterministic chaos theory?**

We compare token sequences generated at different temperatures against the well-studied logistic map (r ∈ [3.4, 4.0]) to quantify similarities and differences in chaotic behavior.

## 🔬 Research Approach

1. **Establish Baseline**: Use the logistic map as ground truth for deterministic chaos
2. **Symbolic Encoding**: Convert both logistic trajectories and LLM tokens to a three-symbol alphabet (A/B/D)
3. **Temperature Sweep**: Generate sequences across T ∈ [0.1, 2.0] for multiple LLM families
4. **Comparative Analysis**: Compute and compare four key dynamical metrics

## 🤖 Models & Systems

### Logistic Map Baseline
- **System**: x_{t+1} = r·x_t·(1 − x_t)
- **Parameter Range**: r ∈ [3.4, 4.0] (150 points, 20 seeds each)
- **Symbolic Encoding**: 
  - A: Attractor band [0.48, 0.52]
  - B: Above band (> 0.52)
  - D: Below band (< 0.48)

### LLM Implementations

#### ✅ Implemented Models

| Model | Parameters | Status | HuggingFace ID |
|-------|-----------|--------|----------------|
| **Alibaba Qwen 1.5B** | 1.8B | ✅ Complete | `Qwen/Qwen1.5-1.8B` |
| **Google Gemma 2B** | 2.61B | ✅ Complete | `google/gemma-2-2b` |

#### 🔄 Planned Models
- OpenAI GPT-2 Series (124M → 1.5B)
- Qwen2 7B (scaling study)
- Qwen2-VL 32B (multimodal extension)

### Experimental Protocol
- **Temperature Points**: 20 evenly spaced in [0.1, 2.0]
- **Sequences per Temperature**: 10 diverse prompts
- **Sequence Length**: 200 tokens
- **Total Sequences**: 200 per model (20 temps × 10 prompts)

## 📊 Key Metrics

For each sequence, we compute:

1. **Minimal Period** (k ≤ 16; ∞ = chaotic)
2. **Entropy Rate** (bits/symbol)
3. **Spectral Gap** (mixing rate indicator)
4. **Symbol Frequencies** (A/B/D distribution)

## 🎨 Key Findings

### Gemma 2B vs Logistic Map

| Metric | Gemma 2B | Logistic Map | Δ |
|--------|----------|--------------|---|
| **Chaotic Fraction** | 90.5% | 63.4% | +27.1pp |
| **Mean Entropy Rate** | 0.788 bits | 0.488 bits | +0.300 |
| **Mean Spectral Gap** | 0.846 | 0.457 | +0.389 |
| **Symbol A Frequency** | 1.9% | 7.0% | -5.1pp |
| **Symbol B Frequency** | 32.4% | 59.7% | -27.3pp |
| **Symbol D Frequency** | 65.7% | 33.3% | +32.4pp |

### Main Conclusions

1. **Predominantly Chaotic**: LLM outputs are 90%+ aperiodic, lacking the clear period-doubling cascade of deterministic chaos
2. **Temperature Control**: Entropy increases from ~0.50 bits (T≤0.5) to ~1.01 bits (T≥1.5)
3. **Fast Mixing**: LLMs exhibit ~85% higher spectral gaps, indicating shorter memory horizons
4. **Symbol Imbalance**: Heavy bias toward D symbols (artifact of modulo-based encoding)
5. **Fundamental Stochasticity**: LLM token streams are stochastic, not deterministic, chaotic

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU recommended (8GB+ VRAM)
- 16GB+ system RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Data_network_Research_Project.git
cd Data_network_Research_Project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **Core ML/DL**: PyTorch ≥2.0.0, Transformers ≥4.35.0, Accelerate ≥0.24.0
- **Data Processing**: NumPy ≥1.24.0, Pandas ≥2.0.0, SciPy ≥1.11.0
- **Visualization**: Matplotlib ≥3.7.0, Seaborn ≥0.12.0, NetworkX ≥3.1
- **Utilities**: tqdm, Jupyter, ipywidgets

See [requirements.txt](requirements.txt) for a complete list.

## 📖 Usage

### Running Experiments

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook LLM_Temperature_Studies.ipynb
   ```

2. **Run Sections in Order**:
   - **Section 5**: Logistic map baseline (if not already computed)
   - **Section 6.3**: Qwen 1.5B implementation
   - **Section 6.2**: Gemma 2B implementation (if available)

3. **Expected Outputs**:
   - CSV files with metrics (`qwen_temperature_results.csv`, etc.)
   - Visualization plots (PNG format)
   - Console progress bars and statistics

### Customization

```python
# Adjust temperature range
TEMPERATURE_MIN, TEMPERATURE_MAX = 0.5, 1.5
N_TEMPERATURES = 30  # More granular sampling

# Change sequence length
SEQ_LENGTH = 500  # Longer sequences for better statistics

# Modify prompts
N_PROMPTS_PER_TEMP = 20  # More samples per temperature

# Try different encoding methods
symbols = token_ids_to_symbols(token_ids, method='hash')
```

## 📁 Project Structure

```
Data_network_Research_Project/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── LLM_Temperature_Studies.ipynb                # Main research notebook
├── LLM_Temperature_Study_Presentation.txt       # Presentation slides text
├── QWEN_IMPLEMENTATION_SUMMARY.md              # Qwen integration details
├── attractor_sequence_code_files/              # Baseline experiments
│   ├── llm_symbol_maps_explorer_LOGISTIC_MAP.ipynb
│   └── llm_symbol_maps_explorer_band_no_transient(1).ipynb
└── [Generated Files]
    ├── qwen_temperature_results.csv            # Qwen experiment data
    ├── logistic_baseline_results.csv           # Baseline data
    ├── qwen_temperature_results.png            # Qwen visualizations
    └── qwen_vs_logistic_comparison.png         # Comparative plots
```

## ⚙️ Computational Requirements

### Minimum
- CPU with 8GB RAM (float32 inference)
- ~35 minutes per model (CPU)

### Recommended
- GPU with 8GB+ VRAM (float16 inference)
- ~15 minutes per model (GPU)

### Optimal
- GPU with 16GB+ VRAM
- Enables larger model experiments (7B+)

## 📈 Runtime Estimates

| Task | Time |
|------|------|
| Model Loading | 1-3 min (first run) |
| Temperature Sweep | 10-30 min (200 sequences) |
| Visualization | <1 min |
| **Total per Model** | **15-35 min** |

## 🔮 Future Work

### Immediate Next Steps
1. Run the Qwen experiment and validate results
2. Analyze period-doubling behavior patterns in detail
3. Perform quantitative comparison with the logistic baseline

### Planned Extensions

#### Model Scaling
- **Qwen2 7B**: Study parameter scaling effects (1.8B → 7B)
- **Qwen2-VL 32B**: Multimodal symbolic dynamics

#### Cross-Vendor Comparison (Section 7)
- Statistical significance testing
- Identify universal vs. model-specific behaviors
- Architecture impact analysis

#### Predictive Framework (Section 8)
- Map LLM temperature to logistic parameter r
- Develop temperature selection guidelines
- Create practical recommendations for practitioners

#### Methodological Improvements
- Embedding-based symbol encodings
- Semantic clustering for A/B/D classification
- Prompt sensitivity analysis
- Longer sequence lengths for rare period detection

## 📚 Research Context

### Why This Matters

Understanding temperature's effect on token-level dynamics can:
- Inform prompt engineering best practices
- Guide sampling strategy selection
- Provide theoretical models of LLM creativity vs. coherence trade-offs
- Bridge connections between statistical models and dynamical systems theory

### Related Work

This project builds on:
- Classic chaos theory (logistic map, symbolic dynamics)
- Information theory (entropy rate, Markov chains)
- Spectral analysis (mixing times, eigengap)
- LLM sampling methods (temperature, top-p, top-k)

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional LLM model integrations
- Improved symbolic encoding methods
- Statistical analysis enhancements
- Visualization improvements
- Documentation and tutorials

## 📝 Citation

If you use this research in your work, please cite:

```bibtex
@misc{llm_temperature_dynamics,
  title={Symbolic Dynamics of LLM Temperature Sampling},
  author={Kadambe, Sanjana and Mehta, Jasreen and Mori, Dhwanil},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Dhwanil25/Data_network_Research_Project},
  note={Research conducted under the supervision of Dr. Neil Johnson, George Washington University}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaboration opportunities:
- GitHub Issues: [Create an issue](https://github.com/Dhwanil25/Data_network_Research_Project/issues)
- Email: dhwanilmori03@gmail.com

## 🙏 Acknowledgments

- **Dr. Neil Johnson**, Professor at George Washington University, for his invaluable guidance and mentorship throughout this research
- **Model Providers**: Alibaba Cloud (Qwen), Google (Gemma), OpenAI (GPT)
- **HuggingFace**: For model hosting and the transformers library
- **Open Source Community**: PyTorch, NumPy, SciPy, Matplotlib contributors

---

**Status**: 🟢 Active Research Project  
**Version**: 1.0  
