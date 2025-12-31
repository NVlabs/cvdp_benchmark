# CVDP Benchmark - Full Documentation

## Detailed Documentation

## Overview

This repository provides tools to evaluate model and agent performance on hardware verification tasks. The benchmark supports two distinct workflows that use Docker Compose for test execution:

- **Non-Agentic Workflow**: Direct LLM API calls + test harness (`docker-compose.yml`)
- **Agentic Workflow**: Custom Docker agents (`docker-compose-agent.yml`) + test harness (`docker-compose.yml`)

Both workflows can be evaluated using single runs or multi-sample pass@k analysis.

## Installation

### Prerequisites

**Python 3.12 is recommended** for optimal compatibility.

**Docker CE (Community Edition)** with a recent version is required for running test harnesses and agents:
- Install Docker CE from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
- **Add your user to the docker group** to run Docker without sudo permissions:
  ```bash
  # Add current user to docker group
  sudo usermod -aG docker $USER
  
  # Log out and back in, or restart your session
  # Verify Docker works without sudo:
  docker ps
  ```

**Advanced Verification Tools** (Optional): For datasets requiring Cadence Xcelium, code coverage, and other commercial verification tools, see the [Custom Verification Images](#custom-verification-images-verif_image) section below.

**EDA License Server Connectivity** (Optional): For datasets requiring Cadence EDA tools with license server access, configure the license network in your `.env` file. The benchmark will automatically validate and create the necessary Docker network connectivity.

### Setup Instructions

1. **Create a virtual environment** (recommended):
```bash
# Create virtual environment
python -m venv cvdp_env

# Activate virtual environment
# On Linux/macOS:
source cvdp_env/bin/activate
# On Windows:
cvdp_env\Scripts\activate
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Common Usage Patterns

### Single Evaluation
```bash
# Non-Agentic: Evaluate with LLM
python run_benchmark.py -f dataset.jsonl -l -m gpt-4

# Agentic: Evaluate with Docker agent  
python run_benchmark.py -f dataset.jsonl -l -g my-agent

# Golden: Test reference solutions
python run_benchmark.py -f dataset.jsonl
```

**Note**: When using `--llm` mode, you cannot specify both `--model` and `--agent` together. Use either model-based LLM processing or agent-based processing.

### Local Inference
For using local models instead of API-based models, use the two-step export/import process:

```bash
# Step 1: Export prompts
python run_benchmark.py -f dataset.jsonl --model local_export --prompts-responses-file prompts.jsonl --llm

# Step 2: Run your local model on prompts.jsonl to generate responses.jsonl

# Step 3: Import and evaluate responses
python run_benchmark.py -f dataset.jsonl --model local_import --prompts-responses-file responses.jsonl --llm
```

👉 **[Complete Local Inference Guide](LOCAL_INFERENCE_GUIDE.md)**

### Multi-Sample Evaluation (Pass@k)
```bash
# Run 5 samples with non-agentic workflow
python run_samples.py -f dataset.jsonl -l -m gpt-4 -n 5

# Run 3 samples with agentic workflow
python run_samples.py -f dataset.jsonl -l -g my-agent -n 3
```

### Single Issue Testing
```bash
# Test one specific issue
python run_benchmark.py -f dataset.jsonl -i cvdp_copilot_my_issue_0001 -l -m gpt-4
```

### Report Analysis
```bash
# Generate human-readable report
python run_reporter.py work/report.json

# Analyze pass@k metrics from multi-sample run
python run_reporter.py work_composite/composite_report.json
```

## Project Structure

```
cvdp_benchmark/
├── run_benchmark.py         # Main evaluation script
├── run_samples.py           # Multi-sample pass@k evaluation
├── run_reporter.py          # Report analysis and visualization
├── src/                     # Core benchmark library
├── tools/                   # Dataset analysis utilities
├── examples/                # Custom model and agent examples
└── README_*.md              # Workflow-specific documentation
```

For detailed architecture and module documentation, see [README_DEVELOPER.md](README_DEVELOPER.md).

## Key Features

- **Flexible Evaluation**: Support for both LLM and Docker agent workflows
- **Statistical Analysis**: Multi-sample pass@k evaluation for reliability metrics
- **Parallel Execution**: Configurable threading for faster evaluation
- **Custom Extensions**: Plugin system for custom models and agents
- **Comprehensive Reporting**: Detailed analysis with category and complexity breakdowns
- **Docker Integration**: Automated container management with resource monitoring
- **Cross-Workflow Compatibility**: Transform and evaluate datasets across different workflow types

## Dataset Transformation and Cross-Workflow Evaluation

The benchmark supports automatic transformation between agentic and non-agentic dataset formats, enabling flexible evaluation scenarios:

### Force Mode Options

**`--force-agentic`** - Transform non-agentic datasets for agentic evaluation:
```bash
# Evaluate a non-agentic dataset using an agentic workflow
python run_benchmark.py -f non_agentic_dataset.jsonl -l -g my-agent --force-agentic

# Include golden reference solution for agent access
python run_benchmark.py -f non_agentic_dataset.jsonl -l -g my-agent --force-agentic --force-agentic-include-golden

# Include test harness files for agent inspection
python run_benchmark.py -f non_agentic_dataset.jsonl -l -g my-agent --force-agentic --force-agentic-include-harness
```

**`--force-copilot`** - Transform agentic datasets for non-agentic evaluation:
```bash
# Evaluate an agentic dataset using non-agentic workflow
python run_benchmark.py -f agentic_dataset.jsonl -l -m gpt-4 --force-copilot

# Multi-sample evaluation with transformation
python run_samples.py -f agentic_dataset.jsonl -l -m gpt-4 --force-copilot -n 5
```

### Use Cases

**Comparative Analysis**: Evaluate the same problems using both workflows:
```bash
# Original agentic evaluation
python run_benchmark.py -f dataset.jsonl -l -g my-agent -p work_agentic

# Same dataset using non-agentic workflow
python run_benchmark.py -f dataset.jsonl -l -m gpt-4 --force-copilot -p work_nonagentic

# Compare results
python run_reporter.py work_agentic/report.json
python run_reporter.py work_nonagentic/report.json
```

**Dataset Compatibility Testing**: Verify dataset quality across workflows:
```bash
# Test if non-agentic dataset works with agents
python run_benchmark.py -f copilot_dataset.jsonl -l -g test-agent --force-agentic

# Test if agentic dataset works with LLMs  
python run_benchmark.py -f agentic_dataset.jsonl -l -m gpt-4 --force-copilot
```

**Agent Development**: Use existing non-agentic datasets for agent testing:
```bash
# Develop agent using well-tested non-agentic problems
python run_benchmark.py -f proven_dataset.jsonl -l -g dev-agent --force-agentic --force-agentic-include-golden
```

### Transformation Details

- **Automatic Conversion**: Dataset formats are automatically adapted during execution
- **Preserved Semantics**: Core problem definitions remain unchanged
- **Format Adaptation**: File structures and prompt formats adjusted for target workflow
- **Backwards Compatibility**: Original datasets remain unmodified
- **Validation**: Transformed datasets undergo the same validation as native formats

## Configuration

### Environment Variables

Set up your environment by copying `.env.example` to `.env`:

```bash
cp .env.example .env
# Edit .env with your settings
```

**Core Variables:**
- `OPENAI_USER_KEY` - API key for OpenAI models
- `DEFAULT_MODEL` - Default LLM model when none specified (default: "gpt-4o-mini")
- `BENCHMARK_THREADS` - Default number of parallel threads
- `BENCHMARK_PREFIX` - Default output directory prefix

**Docker Configuration:**
- `DOCKER_TIMEOUT` - Timeout for Docker harness operations (default: 600s)
- `DOCKER_TIMEOUT_AGENT` - Timeout for Docker agent operations (default: 600s)

**EDA Tool Infrastructure Configuration (optional)**
- `VERIF_EDA_IMAGE` - Docker image for verification tasks with commercial EDA tools (default: "cvdp-cadence-verif:latest")
- `LICENSE_NETWORK` - Docker network name for EDA license server connectivity (default: "licnetwork")
- `LICENSE_NETWORK_AUTO_CREATE` - Automatically create license network if it doesn't exist (default: true)
- `OSS_SIM_IMAGE` - Docker image for simulation tasks with open-source EDA tools (default: "ghcr.io/hdl/sim/osvb")
- `OSS_PNR_IMAGE` - Docker image for place-and-route tasks with open-source EDA tools (default: "ghcr.io/hdl/impl/pnr")

**Template Variables**
- `__VERIF_EDA_IMAGE__` - Replaced with your configured `VERIF_EDA_IMAGE` value
- `__LICENSE_NETWORK__` - Replaced with your configured `LICENSE_NETWORK` value
- `__OSS_SIM_IMAGE__` - Replaced with your configured `OSS_SIM_IMAGE` value
- `__OSS_PNR_IMAGE__` - Replaced with your configured `OSS_PNR_IMAGE` value

**Advanced Options:**
- `CUSTOM_MODEL_FACTORY` - Path to custom model factory implementation
- `ENABLE_SUBJECTIVE_SCORING` - Enable LLM-based subjective scoring
- `DOCKER_QUOTA_THRESHOLD_MB` - Directory size limit for Docker containers (default: 50MB)

### Custom Extensions

**Custom Models**: Extend the benchmark with your own model implementations
```bash
python run_benchmark.py -f dataset.jsonl -l -m custom-model -c /path/to/factory.py
```
👉 **[Model Extension Guide](examples/README.md)**

**Custom Agents**: Create Docker-based agents for the agentic workflow  
👉 **[Agent Development Guide](examples/README.md)**

## Available Tools and Utilities

### Core Scripts
- **`run_benchmark.py`** - Main evaluation script for single runs
- **`run_samples.py`** - Multi-sample evaluation for pass@k metrics  
- **`run_reporter.py`** - Report analysis and human-readable summaries

### Dataset Analysis Tools (`tools/` directory)
- **`dataset_analyzer.py`** - Statistics about problems, categories, and complexity
- **`dataset_subset_creator.py`** - Create dataset subsets by criteria
- **`merge_dataset_files.py`** - Combine multiple dataset files
- **`refinement_analysis.py`** - Analyze iterative problem-solving patterns
- **`jsonl_to_yaml.py`** - Convert JSONL datasets to human-readable YAML format and back

#### Using jsonl_to_yaml.py

Convert JSONL datasets to YAML for easier reading and editing:

```bash
# Convert JSONL to single YAML file (multi-document)
tools/jsonl_to_yaml.py dataset.jsonl

# Convert to separate YAML files (one per problem)
tools/jsonl_to_yaml.py dataset.jsonl --separate-files

# Convert YAML back to JSONL
tools/jsonl_to_yaml.py converted_file.yaml -o restored_dataset.jsonl

# Test roundtrip conversion (ensure lossless conversion)
tools/jsonl_to_yaml.py dataset.jsonl --test-roundtrip
```

The tool preserves all data and supports automatic file type detection. Use this when you need to manually inspect or edit dataset files in a more readable format.

### Developer Resources
For internal development, architecture details, and contribution guidelines - see **[README_DEVELOPER.md](README_DEVELOPER.md)**

## Understanding Results

### Output Files
- **`report.json`** - Primary results with metrics for a single run
- **`raw_result.json`** - Detailed information about each test execution  
- **`composite_report.json`** - Aggregated results across multiple samples (from `run_samples.py`)
- **`agent_changes.patch`** - Changes made by agents (agentic workflow only)

### Analysis Tools
```bash
# Single run analysis
python run_reporter.py work/report.json

# Multi-sample pass@k analysis  
python run_reporter.py work_composite/composite_report.json

# Save metrics and generate text report
python run_reporter.py work_composite/composite_report.json --save -o summary.txt
```

### Pass@k Metrics
For multi-sample runs, pass@k measures the probability that a problem passes in at least k out of n samples:

```
Pass@k = 1 - (1 - c/n)^k
```
Where `c` = successful samples, `n` = total samples, `k` = threshold

## EDA License Network Setup

### Template Variable Substitution

The benchmark supports flexible infrastructure configuration through template variable substitution. In dataset files and configuration templates, use these placeholders which will be automatically replaced:

**Available Template Variables:**
- `__VERIF_EDA_IMAGE__` - Replaced with your configured `VERIF_EDA_IMAGE` value
- `__LICENSE_NETWORK__` - Replaced with your configured `LICENSE_NETWORK` value  
- `__OSS_SIM_IMAGE__` - Replaced with your configured `OSS_SIM_IMAGE` value
- `__OSS_PNR_IMAGE__` - Replaced with your configured `OSS_PNR_IMAGE` value

The benchmark automatically:
- **Detects** when datasets require EDA license connectivity
- **Validates** license network configuration
- **Creates** the license network if it doesn't exist (when auto-creation is enabled)
- **Reports** any configuration issues before execution

### Manual License Network Setup

If you prefer manual setup or need custom configuration:

```bash
# Create the license network manually
docker network create licnetwork

# Verify the network exists
docker network ls | grep licnetwork
```

### Configuration Options

Configure license network settings in your `.env` file:

```bash
# License network configuration
LICENSE_NETWORK=licnetwork
LICENSE_NETWORK_AUTO_CREATE=true
```

### Advanced License Server Setup

For advanced license server configurations (remote servers, custom ports, SSH tunneling), disable auto-creation and set up the license network manually:

```bash
# Disable auto-creation in .env
LICENSE_NETWORK_AUTO_CREATE=false

# Create custom license network manually
docker network create licnetwork

# For SSH tunneling, create a tunnel container
# Example SSH tunnel Dockerfile:
```

**SSH Tunnel Example** (for remote license servers):
```dockerfile
FROM kroniak/ssh-client

COPY ssh.config /root/.ssh/config

RUN  chmod 644  /root/.ssh/config
RUN  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa

CMD [
"ssh", "-i", "/root/.ssh/id_rsa", "tunnel",
"-L", "0.0.0.0:5280:192.168.0.2:5280", "-v", "-N" ]
```

Then connect both your tunnel container and verification containers to the `licnetwork`.

### Troubleshooting License Connectivity

**Network Not Found Error:**
```bash
# Check if network exists
docker network ls | grep licnetwork

# Create manually if needed
docker network create licnetwork
```

**License Checkout Failed:**
```bash
# Test connectivity from within a container on the license network
docker run --rm --network licnetwork alpine telnet <license_server> 5280
```

**Verification Image Issues:**
Make sure `VERIF_EDA_IMAGE` is set to an image that includes Cadence tools:
```bash
# Example verification image configuration (default)
VERIF_EDA_IMAGE=cvdp-cadence-verif:latest
```

### Custom Verification Images (VERIF_EDA_IMAGE)

The `VERIF_EDA_IMAGE` environment variable allows you to specify a custom Docker image with advanced verification tools beyond the standard open-source toolchain. This is essential for datasets requiring:

- **Cadence Xcelium** simulation and verification
- **Advanced code coverage** analysis  
- **Functional assertion** checking
- **SystemVerilog/UVM** testbenches
- **Other enterprise verification tools**

**Basic Configuration:**
```bash
# Set in your .env file (default: cvdp-cadence-verif:latest)
VERIF_EDA_IMAGE=cvdp-cadence-verif:latest
```

The benchmark automatically uses this image when datasets contain commercial EDA tool categories (12, 13, 14). 

**Building the Default Image:**
For a complete example of creating the default verification image with Cadence tools:
```bash
# Navigate to the Cadence Docker example
cd examples/cadence_docker

# Follow setup instructions in README.md, then build
docker build -t cvdp-cadence-verif:latest .
```

See the [Cadence Docker example](examples/cadence_docker/README.md) for detailed setup instructions.

## Docker Directory Size Monitoring

The benchmark includes automatic monitoring of Docker workspace directories to prevent runaway disk usage. Containers are automatically terminated if directory sizes exceed configurable thresholds (default: 50MB).

**Configuration:**
- `DOCKER_QUOTA_THRESHOLD_MB` - Size limit before termination
- `DOCKER_QUOTA_CHECK_INTERVAL` - Check frequency in seconds  
- `DOCKER_QUOTA_MIN_COMPRESS_SIZE_MB` - Minimum file size for compression
- `DOCKER_HARNESS_MEMORY_LIMIT` - Memory limit for harness Docker containers (e.g., '12g'). Sets both --memory and --memory-swap
- `DOCKER_AGENT_MEMORY_LIMIT` - Memory limit for agent Docker containers (e.g., '12g'). Sets both --memory and --memory-swap

For detailed configuration and implementation details, see [README_DEVELOPER.md](README_DEVELOPER.md).

---

## Documentation Index

- **[README_NON_AGENTIC.md](README_NON_AGENTIC.md)** - Complete guide for non-agentic evaluation workflow
- **[README_AGENTIC.md](README_AGENTIC.md)** - Complete guide for Docker agent evaluation workflow
- **[LOCAL_INFERENCE_GUIDE.md](LOCAL_INFERENCE_GUIDE.md)** - Guide for using local models instead of API-based models
- **[Custom Verification Images](#custom-verification-images-verif_eda_image)** - Docker images with Cadence Xcelium and enterprise verification tools
- **[examples/README.md](examples/README.md)** - Custom model and agent development for end users
- **[README_DEVELOPER.md](README_DEVELOPER.md)** - Internal development and architecture documentation

---

## Complete Command-Line Reference

### `run_benchmark.py` Options

**Core Arguments:**
- `-f, --filename <file>` - **Required.** Input dataset file
- `-i, --id <issue_id>` - Evaluate single specific issue only
- `-l, --llm` - Enable LLM/agent mode (vs golden solution mode)

**Model and Agent Selection:**
- `-m, --model <name>` - LLM model name (uses DEFAULT_MODEL if not specified)
- `-g, --agent <name>` - Docker agent image name (agentic workflow)
- `-c, --custom-factory <path>` - Path to custom model factory (recommended to set in .env as CUSTOM_MODEL_FACTORY)

**Configuration:**
- `-a, --answers <file>` - File containing pre-computed answers
- `-p, --prefix <name>` - Output directory prefix (default: "work")
- `-t, --threads <n>` - Number of parallel threads

**Execution Control:**
- `-q, --queue-timeout <seconds>` - Timeout for entire queue

- `-r, --regenerate-report` - Only regenerate report from existing results

**Advanced Options:**
- `-d, --no-patch` - Disable golden patch (golden mode only)
- `-e, --external-network` - External Docker network management
- `--network-name <name>` - Specific Docker network name
- `--enable-sbj-scoring` - Enable LLM-based subjective scoring

**Dataset Transformation:**
- `--force-agentic` - Force agentic mode for non-agentic datasets
- `--force-agentic-include-golden` - Expose golden patch to agent
- `--force-agentic-include-harness` - Expose harness files to agent
- `--force-copilot` - Force non-agentic mode for agentic datasets
- `--copilot-refine <model>` - Refine datasets with additional model

### `run_samples.py` Options

Supports all `run_benchmark.py` options plus:
- `-n, --n-samples <number>` - Number of samples to run (default: 5)
- `-k, --k-threshold <number>` - Pass@k threshold (default: 1)

### `run_reporter.py` Options

**Required:**
- `json_file` - Path to JSON results file

**Optional:**
- `-k <number>` - Override k threshold for pass@k metrics
- `--save` - Save calculated pass@k results back to file
- `-o, --output <file>` - Output file for text report (default: stdout) 