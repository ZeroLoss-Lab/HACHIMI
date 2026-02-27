# HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HACHIMI is a multi-agent collaborative system for generating comprehensive student profiles using Large Language Models (LLMs). The system creates detailed student personas including academic performance, personality traits, values, creativity, and mental health characteristics through a sophisticated five-agent collaboration framework.

## 🌟 Key Features

- **Multi-Agent Collaboration**: Five specialized LLM-based agents work together on a shared whiteboard
- **Comprehensive Profiles**: Generates 15+ dimensional student profiles with high consistency
- **Validator Architecture**: Two-stage validation (Validator-Fast and Validator-Deep) with up to configurable negotiation rounds
- **Smart Deduplication**: SimHash-based similarity filtering to ensure diversity
- **Adaptive Rate Limiting**: AIMD (Additive Increase, Multiplicative Decrease) algorithm for API quota management
- **Multi-Provider Support**: Automatic load balancing across multiple LLM API providers
- **Production Ready**: CLI and Streamlit UI modes, automatic checkpointing, incremental generation

## 📋 Table of Contents

1. [Architecture Overview](#-architecture-overview)
2. [Installation](#-installation)
3. [Quick Start](#-quick-start)
4. [Configuration](#-configuration)
5. [Usage](#-usage)
6. [Output Format](#-output-format)
7. [Evaluation](#-evaluation)
8. [Baseline System](#-baseline-system)
9. [Documentation](#-documentation)
10. [Citation](#-citation)
11. [License](#-license)

## 🏗️ Architecture Overview

### Multi-Agent System

HACHIMI uses five specialized agents that collaborate on a shared whiteboard:

1. **Enrollment & Development Agent**: Handles basic information, developmental stages
2. **Academic Profile Agent**: Generates academic performance, strengths/weaknesses
3. **Personality & Values Agent**: Creates personality traits and value systems
4. **Social & Creativity Agent**: Builds social relationships and creativity profiles
5. **Mental Health Agent**: Produces mental health assessments

### System Components

- **`app.py`**: Qwen2.5-72B version with Streamlit frontend
- **`app_for_GPT4.1.py`**: GPT-4.1 version with Streamlit frontend
- **`app_cli.py`**: Command-line interface for headless generation
- **`providers.py`**: Provider pool management with AIMD rate limiting
- **`rules.py`**: 15 comprehensive validation rules (age-grade alignment, consistency checks, etc.)
- **`baseline_single_shot.py`**: Single-shot baseline system for comparison
- **`analyse_all.py`** / **`analyse_by_chunk.py`**: Quality evaluation tools

## 📥 Installation

### Requirements

- Python 3.8 or higher
- 10GB+ free disk space (for generated profiles)
- Access to LLM API endpoints (OpenAI-compatible)

### Installation Steps

```bash
# Download the repository for anonymous submission
# Click the "Download repository" button on Anonymous GitHub

# Install dependencies
pip install -r requirements.txt

# Configure API providers
nano secrets/providers.json  # Add your API endpoints
```

### Dependencies

```
streamlit>=1.28.0
requests>=2.28.0
```

Standard library modules: `json` `re` `random` `math` `os` `glob` `time` `hashlib` `threading` `concurrent.futures`

## 🚀 Quick Start

### Method 1: CLI (Recommended for Large-Scale Generation)

```bash
# Default generation (1M records, can be interrupted and resumed)
python app_cli.py

# Configuration is done by editing variables at the top of app_cli.py:
# - RUN_ID: "my_experiment"
# - TOTAL: 1000000
# - CHUNK_SIZE: 50
# - MAX_ROUNDS: 3
# - SIMHASH: 3 (Hamming distance threshold)
# - KEYS_PATH: "secrets/providers.json"
```

### Method 2: Streamlit UI with Qwen2.5-72B (Interactive)

```bash
streamlit run app.py
```

Then configure in the sidebar:
- Total records to generate
- Academic level distribution
- SimHash threshold
- Max negotiation rounds

### Method 3: Streamlit UI with GPT-4.1 (Interactive)

```bash
streamlit run app_for_GPT4.1.py
```

Configure via sidebar (same as Method 2). This version is optimized for GPT-4.1 API.

### First-Time Setup

1. **Configure API Providers**

Edit `secrets/providers.json`:
```json
[
  {
    "name": "Provider-Name",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-your-key-here",
    "model": "gpt-4",
    "qpm": 100,
    "capacity_max": 150
  }
]
```

2. **Test Provider Connectivity**

```bash
python -c "from providers import load_providers; clients = load_providers('secrets/providers.json'); print(f'Loaded {len(clients)} providers')"
```

## ⚙️ Configuration

### API Provider Configuration

HACHIMI supports multiple provider formats:

**JSON Format:**
```json
{"name":"Provider","base_url":"https://api.example.com/v1","api_key":"sk-key","model":"gpt-4","qpm":100}
```

**Pipe-Separated Format:**
```
Provider|https://api.example.com/v1|sk-key|gpt-4|qpm=100|capacity_max=150
```

**Key Parameters:**
- `name`: Provider identifier
- `base_url`: API endpoint (OpenAI-compatible)
- `api_key`: Authentication key
- `model`: Model name
- `qpm`: Queries per minute (affects rate limiting)
- `capacity_max`: Maximum capacity for adaptive scaling

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL` | 1,000,000 | Total records to generate |
| `CHUNK_SIZE` | 50 | Records per output file |
| `MAX_ROUNDS` | 3 | Max validator-agent negotiation rounds |
| `SIMHASH` | 3 | Hamming distance threshold for deduplication |
| `PER_STEP_M` | 20 | Batch size for scheduling (does not affect speed) |

### About the Sample Data

This release includes **one dataset**:

- **10K sample** (`sample_data/merged_students_10k.jsonl`, 49MB): Real profiles generated by our system for **reviewer inspection** and **quick testing**.

This sample is provided to facilitate reproducibility checks during peer review. The **full dataset (1M+ profiles)** will be released **after the review process** via **our platform**.


### Performance Tuning

Generation speed is primarily controlled by:
1. **Provider Count**: Number of providers in `secrets/providers.json`
2. **QPM per Provider**: Queries per minute configured per provider
3. **API Latency**: Actual response time of API endpoints
4. **MAX_WORKERS** (optional): Manual concurrency limit

**NOT affected by:**
- `PER_STEP_M`: Only affects batch size, not throughput
- `TOTAL`: Only determines total runtime, not rate
- `CHUNK_SIZE`: Only affects output file organization

For detailed speed optimization guide, see the Provider Configuration and Performance Tuning sections in this README.

## 📊 Usage

### Basic Commands

Choose your preferred interface:

**Command-Line (Headless)**
```bash
# Generate profiles without UI
python app_cli.py

# Monitor progress in real-time
tail -f output/*/students_chunk_*.jsonl
```

**Interactive UI (Qwen2.5-72B version)**
```bash
streamlit run app.py
```

**Interactive UI (GPT-4.1 version)**
```bash
streamlit run app_for_GPT4.1.py
```

**Evaluation**
```bash
# Evaluate all runs
python analyse_by_chunk.py

# Evaluate specific run
python analyse_by_chunk.py --input_dir ./output/run_experiment_001

# Evaluate merged output
python analyse_all.py --simhash_threshold 3 --topk 100

# Run baseline comparison
python baseline_single_shot.py --mode age_15 --count 5000 --output ./output/baseline
```

### Analysis and Evaluation

```bash
# Run quality analysis on all outputs
python analyse_by_chunk.py

# Compare two runs
python compare_baseline.py --group1 ./output/multi_agent --group2 ./output/baseline

# Evaluate sample data (10K profiles)
python analyse_all.py --input_file ./sample_data/merged_students_10k.jsonl --simhash_threshold 3
```

## 📤 Output Format

### Directory Structure

```
./output/
  {run_id}/
    students_chunk_1.jsonl      # 50 records per chunk
    students_chunk_2.jsonl
    ...
    schedule.json               # Generation plan with sampling constraints
    meta.json                   # Total_N and chunk_size
    failures.jsonl              # Failed generation attempts

./eval/
  overview.md                   # Summary of all runs
  _index.csv                    # Index table
  _index.json                   # Index data
  {run_id}/
    summary.json                # Metrics and distributions
    per_item.jsonl              # Per-record validation results
    near_duplicates.json        # Similar profile pairs
    suspicious_samples.json     # Samples with most issues
    dashboard.md                # Human-readable report
    buckets.csv                 # Distribution breakdowns
```

### Student Profile Schema

```json
{
  "id": 1,
  "name": "Student Name",
  "age": 15,
  "gender": "Female",
  "grade": "Grade 9",
  "agent_name": "zhang1_mei3",
  "developmental_stages": {
    "piaget": "Formal Operational Stage",
    "erikson": "Identity vs Role Confusion",
    "kohlberg": "Post-conventional Level"
  },
  "strong_subjects": ["Mathematics", "Physics"],
  "weak_subjects": ["History", "Politics"],
  "academic_level": "High: Top 10% in school",
  "personality": "Conscientious and introverted...",
  "values": "Demonstrates strong moral character...",
  "social_relationships": "Maintains close friendships...",
  "creativity": "Shows moderate fluency in problem-solving...",
  "mental_health": "Generally well-adjusted with normal stress levels..."
}
```

### Academic Level (Strict Four-Choice)

- "High: Top 10% in school"
- "Medium: Top 10-30% in school"
- "Low: Top 30-50% in school"
- "Poor: Bottom 50% in school"

## 📈 Evaluation

### Quality Metrics

- **Distinct-1/2**: Character-level text diversity
- **SimHash Hamming Distance**: Near-duplicate detection
- **Template Detection**: Jaccard similarity across dimensions

### Validation Coverage

15 comprehensive rules covering:
- ✅ Required fields presence
- ✅ Age-grade alignment (Rule R1)
- ✅ Developmental stage consistency (Rule R2)
- ✅ Academic level format
- ✅ Subject set non-overlap (Rule R3)
- ✅ Creativity feasibility consistency (Rule R4)
- ✅ Values-mental health alignment (Rule R5)
- ✅ Agent name format (Rule R6)
- ✅ 7-dimension values coverage (Rule R8)
- ✅ 8-dimension creativity coverage (Rule R9)
- ✅ Mental health structure (Rule R10)
- ✅ Cross-field consistency (Rule R11)
- ✅ Non-diagnostic language (Rule R12)
- ✅ Natural language (Rule R13)
- ✅ Single-paragraph format (Rule R14)

### Sample Evaluation Results

Expected quality metrics for 10K profiles:
- Pass rate: ~95-98%
- Distinct-1: 0.15-0.25
- Distinct-2: 0.40-0.60
- Generation speed: 2-5 profiles/second (depends on API)

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)**: Development guide and architecture overview
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)**: Installation guide
- **[Validation Rules](rules.py)**: Complete validation rule specifications

## 🔬 Citation

If you use this system in your research, please cite:

```bibtex
@software{hachimi2025,
  title={HACHIMI: Human-centric Agent-based Character and Holistic Individual Modeling Infrastructure},
  author={Research Team},
  year={2025},
  note={Multi-agent LLM system for comprehensive student profile generation}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with OpenAI-compatible LLM APIs
- Uses multi-agent AI architecture and SimHash for efficient duplicate detection
- Rate limiting algorithm based on AIMD (Additive Increase, Multiplicative Decrease)
