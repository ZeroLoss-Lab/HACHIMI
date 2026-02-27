# HACHIMI Code Repository Index

<p align="center">
  <a href="INDEX.md">English</a> | <a href="INDEX_CN.md">简体中文</a>
</p>



## 📁 File Structure

```
github-release/
├── README.md                   # Main documentation (English)
├── README_CN.md                # Main documentation (中文)
├── INDEX.md                    # This file (English)
├── INDEX_CN.md                 # This file (中文)
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore patterns
│
├── requirements.txt            # Python dependencies
├── rules.py                    # 15 validation rules (R1-R15)
├── providers.py                # API provider management & rate limiting
│
├── app.py                      # [MAIN] Streamlit Web UI (Qwen2.5-72B)
├── app_cli.py                  # Command-line interface
├── app_for_GPT4.1.py           # GPT-4.1 optimized version
│
├── baseline_single_shot.py     # Single-shot baseline for comparison
├── analyse_all.py              # Analysis tool (full dataset)
├── analyse_by_chunk.py         # Analysis tool (chunked data)
├── provider_health_check.py    # API provider health check
├── test_field_mapping.py       # Field mapping unit tests
│
├── sample_data/
│   └── merged_students_10k.jsonl   # 10,000 anonymized student profiles
│
└── secrets/
    └── README.md               # API configuration guide
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `secrets/providers.json` (see `secrets/README.md` for format):

```json
{
  "providers": [
    {
      "name": "your_provider",
      "base_url": "https://api.example.com/v1",
      "api_key": "your-api-key",
      "model": "model-name",
      "qpm": 60
    }
  ]
}
```

### 3. Run the System

**Web UI (Recommended):**
```bash
streamlit run app.py
```

**Command Line:**
```bash
python app_cli.py --count 100 --grade 初一 --gender 女
```

---

## 📋 File Descriptions

| File | Purpose | How to Run |
|------|---------|------------|
| `app.py` | Main program - Multi-agent collaborative generation (Web UI) | `streamlit run app.py` |
| `app_cli.py` | CLI version - Batch generation | `python app_cli.py` |
| `app_for_GPT4.1.py` | GPT-4.1 optimized version | `streamlit run app_for_GPT4.1.py` |
| `baseline_single_shot.py` | Baseline - Single-call generation | `python baseline_single_shot.py` |
| `providers.py` | Provider pool management (imported by others) | Not runnable directly |
| `rules.py` | 15 validation rules (imported by others) | Not runnable directly |
| `analyse_all.py` | Analyze generation results (full) | `python analyse_all.py` |
| `analyse_by_chunk.py` | Analyze generation results (chunked) | `python analyse_by_chunk.py` |
| `provider_health_check.py` | Check API provider availability | `python provider_health_check.py` |
| `test_field_mapping.py` | Test field mapping correctness | `python test_field_mapping.py` |

---

## 🔧 Core Features

### Multi-Agent Architecture
- **5 Content Agents**: Enrollment & Development, Academic Profile, Personality & Values, Social & Creativity, Mental Health
- **2 Validators**: Fast Validator + Deep Validator
- **Multi-Round Negotiation**: Configurable number of refinement rounds

### Quality Control
- **15 Validation Rules (R1-R15)**: Age-grade consistency, developmental alignment, cross-field consistency, etc.
- **SimHash Deduplication**: Hamming distance threshold to ensure diversity
- **Academic Level Anchoring**: Strict four-choice format (High/Medium/Low/Poor)

### Sampling Control
- **QuotaScheduler**: Grade × Gender × Subject cluster pre-sampling
- **Light Filtering**: Real-time quality screening
- **Optimism Bias Suppression**: Cross-dimensional consistency checks

---

## 📊 Sample Data

`sample_data/merged_students_10k.jsonl` contains 10,000 anonymized student profiles with fields:
- Basic info: id, age, gender, grade, agent_name
- Developmental stages: Piaget cognitive / Erikson psychosocial / Kohlberg moral
- Academic info: strong/weak subjects, academic level
- Psychological profiles: personality, values, social relationships, creativity, mental health

---

## ⚠️ Important Notes

1. **API Configuration**: Must configure `secrets/providers.json` before running
2. **Rate Limiting**: Built-in token bucket rate limiting - adjust `qpm` based on your API quota
3. **Chunked Storage**: Results automatically saved to `output/<run_id>/` in 50-record chunks
4. **Anonymization**: All generated data automatically replaces names with "X号学生" format

---

## 📄 Citation

```bibtex
@inproceedings{jiang2026hachimi,
  title={Generating Authentic Student Profiles: A Multi-Agent Collaboration Approach},
  author={Jiang, Yilin and Tan, Fei and Yin, Xuanyu and Leng, Jing and Zhou, Aimin},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```

---

**Authors**: Yilin Jiang, Fei Tan (Corresponding), Xuanyu Yin, Jing Leng, Aimin Zhou  
**Affiliations**: East China Normal University, HKUST(GZ), Shanghai Innovation Institute  
**Contact**: ftan@mail.ecnu.edu.cn

---

> 🔔 **Open Source Notice**: Full dataset and source code will be released soon. Stay tuned!
