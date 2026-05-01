# CVDP Benchmark

The CVDP Benchmark is a framework for evaluating LLM and agent solutions on hardware verification challenges.

## Front Matter

**Dataset**: The CVDP Benchmark dataset is available on Hugging Face at [🤗 nvidia/cvdp-benchmark-dataset](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset).

**Paper**: For detailed methodology and evaluation results, see our arXiv preprint: [Comprehensive Verilog Design Problems: A Next-Generation Benchmark Dataset for Evaluating Large Language Models and Agents on RTL Design and Verification](https://arxiv.org/abs/2506.14074).

**Dataset Notes**: The dataset published on Hugging Face contains the vast majority of evaluation problems discussed in the preprint. Twenty datapoints were omitted from the initial public release release due to test harness issues and licensing restrictions. Additionally, we excluded the reference solutions—'output' for non-agentic and 'patch' for agentic—from the initial dataset release to help mitigate data contamination. We hope to make these available in the future; if you have an immediate need, please feel free to reach out to discuss.

**New LLM Benchmarking Coalition**: The Silicon Integration Initiative (Si2) is adopting CVDP into their newly launched LLM Benchmarking Coalition (LBC), where industry and academic members work together to improve benchmarking and expedite the development of high-quality large language models for semiconductor design problems. The coalition builds upon CVDP by extending problems to cover new categories and design domains, refining benchmarking metrics, and maintaining a leaderboard. Learn more at [Si2 LLM Benchmarking Coalition](https://si2.org/si2-llm-benchmarking-coalition-kicks-off/).

## Quick Start

### Prerequisites

**Python 3.12 is recommended** for optimal compatibility.

**Docker CE (Community Edition)** with a recent version is required:
- Install Docker CE from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
- **Add your user to the docker group** to run Docker without sudo permissions:
  ```bash
  # Add current user to docker group
  sudo usermod -aG docker $USER
  
  # Log out and back in, or restart your session
  # Verify Docker works without sudo:
  docker --version
  ```

### Build the Open-Source Simulation Image

CVDP v1.1.0 uses a dedicated open-source simulation image for non-commercial simulation tasks. Build it once after cloning the repository or pulling the v1.1.0 tooling updates:

```bash
docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .
```

This image provides the default `OSS_SIM_IMAGE` environment used by dataset harnesses via `__OSS_SIM_IMAGE__`. CVDP v1.1.0 no longer uses the legacy third-party simulation images for this default open-source simulation flow. The build includes cocotb 2.0.1, pytest 8.3.2, Icarus Verilog v13_0, Yosys yosys-0.40, and Verilator v5.038. If you tag the image differently, set the matching value in `.env`:

```bash
OSS_SIM_IMAGE=nvidia/cvdp-sim:v1.0.0
```

Open-source place-and-route tasks still use the separate `OSS_PNR_IMAGE` setting, but in CVDP v1.1.0 its default points at the same `nvidia/cvdp-sim:v1.0.0` image:

```bash
OSS_PNR_IMAGE=nvidia/cvdp-sim:v1.0.0
```

### Setup Instructions

**1. Create a virtual environment** (recommended):
```bash
# Create virtual environment
python -m venv cvdp_env

# Activate virtual environment
# On Linux/macOS:
source cvdp_env/bin/activate
# On Windows:
cvdp_env\Scripts\activate
```

**2. Install Python dependencies**:
```bash
pip install -r requirements.txt
```

**3. Configure environment variables**:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key:
# OPENAI_USER_KEY=your_api_key_here
```

## 🤖 Try Non-Agentic Workflow (LLM-based)

Evaluate language models via direct API calls on example problems:

**1. Test golden reference solutions first:**
```bash
# Run golden solutions to verify test harness works  
./run_benchmark.py -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl

# Check results - should show 100% pass rate
cat work/report.txt
```

**2. Multi-sample evaluation (recommended - no-commercial dataset):**
```bash
# Run 5 samples for statistical reliability with Pass@1 (requires OPENAI_USER_KEY)
./run_samples.py -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl -l -m gpt-4o-mini -n 5 -k 1 -p work_composite

# Check results in composite_report.txt
cat work_composite/composite_report.txt
```

**3. Single run with LLM (no-commercial dataset):**
```bash
# Same dataset with LLM evaluation (requires OPENAI_USER_KEY)
./run_benchmark.py -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl -l -m gpt-4o-mini -p work_llm

# Check LLM results
cat work_llm/report.txt
```

**4. Try other example datasets:**
```bash
# Commercial dataset (5 samples - requires OPENAI_USER_KEY)
# NOTE: Requires commercial EDA tools (Cadence Xcelium), license network, and VERIF_EDA_IMAGE setup
# See README sections: "EDA License Network Setup" and "Custom Verification Images" 
./run_samples.py -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_generation_commercial_with_solutions.jsonl -l -m gpt-4o-mini -n 5 -k 1 -p work_commercial_composite

# Code comprehension dataset (5 samples)  
# NOTE: Uses BLEU/ROUGE scoring + LLM-based subjective scoring (requires OPENAI_USER_KEY)
# Subjective scoring is automatically enabled - no additional configuration needed
./run_samples.py -f example_dataset/cvdp_v1.0.1_example_nonagentic_code_comprehension_with_solutions.jsonl -l -m gpt-4o-mini -n 5 -k 1 -p work_comprehension_composite

# Check results
cat work_commercial_composite/composite_report.txt
cat work_comprehension_composite/composite_report.txt
```

## 🔧 Try Agentic Workflow (Docker-based)

Evaluate custom agents running in Docker containers:

**1. Test golden reference solutions first:**
```bash
# Run golden solutions to verify test harness works
./run_benchmark.py -f example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl

# Check results - should show 100% pass rate
cat work/report.txt
```

**2. Multi-sample evaluation with agent (recommended):**
```bash
# Copy and build the example agent first
# NOTE: This is a dummy agent that will fail - it just replaces "input" with "loompa" in RTL
# For real evaluation, you'll need to implement proper agent logic in agent.py
cp -r examples/agent/ ./my-agent/
cd my-agent/
./build_agent.sh
cd ..

# Run 5 samples for statistical reliability with Pass@1
./run_samples.py -f example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl -l -g cvdp-example-agent -n 5 -k 1 -p work_composite

# Check results in composite_report.txt
cat work_composite/composite_report.txt
```

**3. Single run with agent (no-commercial dataset):**
```bash
# Run agent on same dataset  
# NOTE: Example agent is for demonstration only - expect low success rates
./run_benchmark.py -f example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl -l -g cvdp-example-agent -p work_agent

# Check agent results
cat work_agent/report.txt
```

**4. Try commercial dataset with multi-sampling:**
```bash
# Agent run on commercial problems (5 samples)
# NOTE: Requires commercial EDA tools (Cadence Xcelium), license network, and VERIF_EDA_IMAGE setup
# NOTE: This is a dummy agent that will fail - it just replaces "input" with "loompa" in RTL
./run_samples.py -f example_dataset/cvdp_v1.0.1_example_agentic_code_generation_commercial_with_solutions.jsonl -l -g cvdp-example-agent -n 5 -k 1 -p work_agentic_commercial_composite

# Check results
cat work_agentic_commercial_composite/composite_report.txt
```

**5. Debug single problem:**
```bash
# Run one problem to examine the agent workflow
# USEFUL: Good for understanding how agents interact with the environment and debug failures  
./run_benchmark.py -f example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl -i cvdp_agentic_fixed_arbiter_0001 -l -g cvdp-example-agent -p work_debug

# Navigate to the work directory to examine agent execution
cd work_debug/cvdp_agentic_fixed_arbiter_0001/harness/1/

# Manually run the agent for debugging
./run_docker_agent.sh -d

# Run evaluation manually
./run_docker_harness_direct.sh
```

## 🚀 Next Steps

**Resources:**
- **Full Dataset**: Download from [🤗 nvidia/cvdp-benchmark-dataset](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset)
- **Local Inference**: See [Local Inference Guide](LOCAL_INFERENCE_GUIDE.md) for using local models instead of API calls
- **Custom Models**: See [Non-Agentic Guide](README_NON_AGENTIC.md) for custom model development
- **Custom Agents**: See [Agentic Guide](README_AGENTIC.md) for building your own agents  
- **Advanced Features**: See [Examples](examples/README.md) for model factories and custom implementations

**📖 Complete Documentation:**
For comprehensive documentation including detailed installation instructions, configuration options, advanced usage patterns, command-line reference, and troubleshooting guides, see [README_FULL.md](README_FULL.md).

## Documentation Index

- **[README_FULL.md](README_FULL.md)** - Complete documentation with detailed guides and configuration
- **[README_NON_AGENTIC.md](README_NON_AGENTIC.md)** - Complete guide for non-agentic evaluation workflow
- **[README_AGENTIC.md](README_AGENTIC.md)** - Complete guide for Docker agent evaluation workflow
- **[LOCAL_INFERENCE_GUIDE.md](LOCAL_INFERENCE_GUIDE.md)** - Guide for using local models instead of API-based models
- **[examples/README.md](examples/README.md)** - Custom model and agent development for end users
- **[README_DEVELOPER.md](README_DEVELOPER.md)** - Internal development and architecture documentation
