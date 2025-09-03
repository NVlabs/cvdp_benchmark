# Agentic Workflow Guide

A - The agentic workflow evaluates custom Docker-based agents on hardware verification challenges.

## Overview

**How it works:**
1. 🐳 Runs your custom agent in Docker containers
2. 📁 Provides isolated workspace with challenge files
3. 🔍 Tracks all changes made by your agent
4. ⚡ Evaluates results using test harnesses

**Key advantages:**
- Complete control over agent behavior
- Support for complex, multi-step processing
- Language and framework agnostic
- Isolated execution environment
- Detailed change tracking

**Compared to [Non-Agentic Workflow](README_NON_AGENTIC.md):** Uses Docker containers instead of API calls, enabling custom agent logic and tools.

## Creating Your Agent

### Agent Requirements
Your Docker agent must:
- ✅ Read task from `/code/prompt.json`
- ✅ Access files in mounted directories (`/code/docs`, `/code/rtl`, `/code/verif`, `/code/rundir`)
- ✅ Make appropriate file modifications
- ✅ Exit with code 0 when complete

### Quick Start Example

**1. Copy the agent example and build:**
```bash
# Copy the complete agent example
cp -r examples/agent/ ./my-agent/
cd my-agent/

# Build using the provided script
./build_agent.sh
```

**2. Create agent.py:**
```python
import json

def main():
    # Read the task
    with open("/code/prompt.json", "r") as f:
        task = json.load(f)["prompt"]
    
    # Your agent logic here
    print(f"Processing: {task}")
    
    # Make file modifications in /code/* directories
    
if __name__ == "__main__":
    main()
```

**3. Build and run:**
```bash
docker build -t my-agent .
python run_benchmark.py -f dataset.jsonl -l -g my-agent
```

👉 **[Complete Agent Examples](examples/README.md)**

## Container Base Images and Tool Availability

### Recommended Base Images for Agents

**`ghcr.io/hdl/sim/osvb`** - Open Source Verification Base Image:
```dockerfile
# Use osvb as base for agents that need EDA tools
FROM ghcr.io/hdl/sim/osvb
WORKDIR /app

# Install your agent dependencies
COPY requirements.txt agent.py ./
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "agent.py"]
```

**Available Tools in osvb:**
- **Icarus Verilog** - Open-source Verilog simulator
- **Verilator** - Fast Verilog simulator and lint tool
- **Yosys** - Open-source synthesis suite
- **GTKWave** - Waveform viewer
- **Standard build tools** (make, gcc, python, etc.)

### Advanced Verification Environment

For datasets requiring advanced design verification (DV) capabilities, a custom verification image can be configured:

**VERIF_EDA_IMAGE Configuration:**
```bash
# Set environment variable for custom verification image
export VERIF_EDA_IMAGE=your-registry/cadence-xcelium:latest

# Run benchmark with custom verification image
python run_benchmark.py -f dv_dataset.jsonl -l -g my-agent
```

👉 **[Custom Verification Images](README.md#custom-verification-images-verif_eda_image)** - Information about creating custom verification images with Cadence Xcelium, coverage tools, and enterprise verification capabilities.

### Tool Availability Matrix

| Image Type | Icarus | Verilator | Yosys | Xcelium | Code Coverage | Assertions |
|------------|--------|-----------|--------|---------|---------------|------------|
| **osvb** | ✅ | ✅ | ✅ | ❌ | Basic | Basic |
| **VERIF_EDA_IMAGE** | ✅ | ✅ | ✅ | ✅ | Advanced | Advanced |

### Usage Guidelines

**For Basic RTL Development:**
```dockerfile
# Use osvb for most verification tasks
FROM ghcr.io/hdl/sim/osvb
# Your agent implementation
```

**For Advanced DV Requirements:**
```bash
# Configure custom verification image
export VERIF_EDA_IMAGE=your-registry/cadence-tools:latest

# Required for dataset categories like:
# - Advanced code coverage analysis
# - Complex functional verification
# - SystemVerilog/UVM testbenches
# - Formal verification workflows
```

**Test Harness Considerations:**
- **Default**: `ghcr.io/hdl/sim/osv` (note: different from osvb) is commonly used for test harness execution
- **Advanced DV**: `VERIF_EDA_IMAGE` is automatically used when required by specific dataset categories
- **Tool Consistency**: Both agent and test harness should have compatible tool environments

### Agent Development Best Practices

**Tool Selection Strategy:**
```python
# Example: Agent detecting available tools
import subprocess
import os

def detect_simulator():
    """Detect available simulation tools"""
    if shutil.which("xrun"):  # Cadence Xcelium
        return "xcelium"
    elif shutil.which("verilator"):
        return "verilator" 
    elif shutil.which("iverilog"):  # Icarus Verilog
        return "icarus"
    else:
        raise RuntimeError("No supported simulator found")

def main():
    simulator = detect_simulator()
    print(f"Using simulator: {simulator}")
    
    # Adapt agent behavior based on available tools
    if simulator == "xcelium":
        # Use advanced features like coverage, assertions
        run_advanced_verification()
    else:
        # Use basic simulation capabilities
        run_basic_simulation()
```

**Docker Image Size Optimization:**
- Use multi-stage builds to minimize final image size
- Only include tools needed for your specific agent
- Consider tool licensing requirements for distribution

## Directory Structure and Docker Compose Files

The agentic workflow uses two Docker Compose files:

1. **docker-compose-agent.yml**: Generated automatically for running your agent
2. **docker-compose.yml**: Used for running the test harness

### Docker Compose for Agent

The system automatically generates a `docker-compose-agent.yml` file with:

```yaml
version: '3'
services:
  agent:
    image: your-agent-name
    volumes:
      - ./docs:/code/docs
      - ./rtl:/code/rtl
      - ./verif:/code/verif
      - ./rundir:/code/rundir
      - ./prompt.json:/code/prompt.json
    working_dir: /code
```

### Docker Compose for Test Harness

The test harness uses a `docker-compose.yml` file from the dataset, or generates a default:

```yaml
version: '3'
services:
  test:
    image: us-central1-docker.pkg.dev/turing-gpt/verilogeval/cadence-tools
    volumes:
      - ./docs:/code/docs
      - ./rundir:/code/rundir
      - ./rtl:/code/rtl
      - ./verif:/code/verif
      - ./src:/code/src
      - ../../src/llm_lib:/pysubj
    working_dir: /code/rundir
    environment:
      - OPENAI_USER_KEY=${OPENAI_USER_KEY}
```

Your agent will have access to the following directories mounted as volumes:

- `/code/docs/` - Documentation files
- `/code/rtl/` - RTL (hardware description) files
- `/code/verif/` - Verification files
- `/code/rundir/` - Working directory for the agent
- `/code/prompt.json` - Contains the prompt/instructions for the task

## Running Tests with Your Agent

Use the benchmark script with the following parameters:

```bash
python run_benchmark.py -f input.json -l -g your-agent-name
```

### Basic Usage Examples

```bash
# Basic agentic evaluation
python run_benchmark.py -f dataset.jsonl -l -g my-agent

# Single datapoint evaluation
python run_benchmark.py -f dataset.jsonl -i cvdp_agentic_my_issue_0001 -l -g my-agent

# With custom output directory and host mode
python run_benchmark.py -f dataset.jsonl -l -g my-agent -p work_experiment1 -o

# Multi-threaded execution
python run_benchmark.py -f dataset.jsonl -l -g my-agent -t 4
```

### Complete Command-Line Options

**Core Arguments:**
- `-f, --filename <file>` - **Required.** Input JSON file with test cases
- `-l, --llm` - Enable agent mode (required for agentic workflow)
- `-g, --agent <name>` - **Required.** Docker image name for your agent
- `-i, --id <issue_id>` - Run evaluation for single datapoint only

**Configuration Options:**
- `-a, --answers <file>` - File containing pre-computed answers to prompts
- `-p, --prefix <name>` - Output directory prefix (default: "work")
- `-t, --threads <n>` - Number of parallel threads (default: environment var or 1)

**Docker Configuration:**

- `-e, --external-network` - Use external Docker network management
- `--network-name <name>` - Specify custom Docker network name

**Execution Control:**
- `-q, --queue-timeout <seconds>` - Timeout for entire queue of tasks
- `-r, --regenerate-report` - Only regenerate report from existing raw_result.json

**Advanced Options:**
- `--enable-sbj-scoring` - Enable LLM-based subjective scoring
- `--force-agentic` - Force agentic mode for non-agentic datasets
- `--force-agentic-include-golden` - Expose golden patch file to agent
- `--force-agentic-include-harness` - Expose harness files to agent

## Running Multiple Samples for Pass@k Evaluation

To evaluate your agent's reliability across multiple runs, use the run_samples.py script:

```bash
python run_samples.py -f input.json -l -g your-agent-name -n 5 -k 1
```

### Sample-Specific Options

In addition to all the options available in `run_benchmark.py`, `run_samples.py` provides:

- `-n, --n-samples <number>` - Number of samples to run (default: 5)
- `-k, --k-threshold <number>` - Pass@k threshold (default: 1)

### Usage Examples

```bash
# Basic multi-sample agent evaluation
python run_samples.py -f dataset.jsonl -l -g my-agent -n 10 -k 1

# With custom directory prefix and host mode
python run_samples.py -f dataset.jsonl -l -g my-agent -n 5 -p work_experiment1 -o

# Force agentic mode with golden patch access
python run_samples.py -f non_agentic_dataset.jsonl -l -g my-agent -n 3 --force-agentic --force-agentic-include-golden

# Regenerate reports only (skip running if samples exist)
python run_samples.py -f dataset.jsonl -l -g my-agent -n 5 -r
```

**Multi-sample benefits:**
- 🎯 Statistical reliability across multiple runs
- 📈 Pass@k metrics for consistency evaluation
- 🔄 Tests agent robustness with different random seeds
- 📊 Comprehensive reliability statistics

The pass@k metric: `Pass@k = 1 - (1 - c/n)^k` where `c` = successful samples, `n` = total samples, `k` = threshold.

## Evaluating Non-Agentic Datasets with Agentic Workflow

The `--force-agentic` flags enable you to evaluate non-agentic datasets using agents, providing access to additional testing data and enabling comparative analysis:

### Basic Usage
```bash
# Evaluate a non-agentic dataset using agents
python run_benchmark.py -f non_agentic_dataset.jsonl -l -g my-agent --force-agentic

# Include golden reference solution for debugging
python run_benchmark.py -f non_agentic_dataset.jsonl -l -g my-agent --force-agentic --force-agentic-include-golden

# Include test harness files for agent inspection
python run_benchmark.py -f non_agentic_dataset.jsonl -l -g my-agent --force-agentic --force-agentic-include-harness
```

### Advanced Force Options

**`--force-agentic-include-golden`** - Expose reference solution to agent:
```bash
# Agent can access the expected solution file(s)
python run_benchmark.py -f dataset.jsonl -l -g debug-agent --force-agentic --force-agentic-include-golden

# Useful for agent development and debugging
python run_samples.py -f dataset.jsonl -l -g dev-agent --force-agentic --force-agentic-include-golden -n 3
```

**`--force-agentic-include-harness`** - Expose test harness to agent:
```bash
# Agent can inspect test files and docker-compose.yml
python run_benchmark.py -f dataset.jsonl -l -g analysis-agent --force-agentic --force-agentic-include-harness

# Helpful for agents that need to understand test structure
python run_samples.py -f dataset.jsonl -l -g smart-agent --force-agentic --force-agentic-include-harness -n 5
```

### Use Cases

**Agent Development**: Use established non-agentic datasets for agent training:
```bash
# Use well-tested problems for agent development
python run_benchmark.py -f proven_dataset.jsonl -l -g dev-agent --force-agentic --force-agentic-include-golden

# Test agent robustness on diverse problem types
python run_samples.py -f large_copilot_dataset.jsonl -l -g my-agent --force-agentic -n 10
```

**Performance Comparison**: Compare agent vs LLM performance on the same problems:
```bash
# Evaluate with LLMs (original format)
python run_benchmark.py -f non_agentic_dataset.jsonl -l -m gpt-4 -p work_llm

# Evaluate with agents (transformed format)
python run_benchmark.py -f non_agentic_dataset.jsonl -l -g my-agent --force-agentic -p work_agent

# Compare results
python run_reporter.py work_llm/report.json  
python run_reporter.py work_agent/report.json
```

**Dataset Expansion**: Leverage existing non-agentic datasets for agent testing:
```bash
# Use existing copilot benchmarks for agent evaluation
python run_samples.py -f comprehensive_copilot_benchmark.jsonl -l -g production-agent --force-agentic -n 20
```

### Transformation Details

When using `--force-agentic`:
- **Prompt Conversion**: Non-agentic prompts become agent task instructions
- **Context Expansion**: Single-file context becomes multi-file agent workspace
- **Output Adaptation**: Expected changes are converted to file structure expectations
- **Environment Setup**: Agent workspace mirrors problem structure
- **Golden Access**: Optional access to reference solutions for debugging

### File Structure Mapping

Non-agentic context → Agent workspace:
```
Original:                    Transformed:
"context": {                /code/
  "module.v": "...",    →      rtl/module.v
  "test.sv": "...",     →      verif/test.sv  
  "README.md": "..."    →      docs/README.md
}                           /code/prompt.json (task)
```

### Debugging Capabilities

**With Golden Access**: Agent can compare its work against reference:
- `/code/golden/` directory contains expected solutions
- Useful for development and validation
- Should not be used in production evaluation

**With Harness Access**: Agent can understand test structure:
- `/code/harness/` directory contains test files
- `docker-compose.yml` shows test execution setup
- Enables test-aware agent strategies

## How It Works

### Process Flow
1. **Setup** - Creates challenge directories and backs up original state
2. **Agent Execution** - Runs your Docker agent with mounted directories
3. **Change Tracking** - Compares before/after states and generates diffs
4. **Test Evaluation** - Runs verification tests on modified files
5. **Report Generation** - Combines results with detailed change analysis

### Change Tracking
The system automatically tracks all agent modifications:
- **`agent_changes.patch`** - Complete diff of all changes
- **File additions, modifications, deletions** - Detailed in unified diff format
- **Clear debugging trail** - Understand exactly what your agent changed

### Volume Mounts
Your agent has access to these directories:
- **`/code/docs`** - Documentation files
- **`/code/rtl`** - Hardware description files
- **`/code/verif`** - Verification files  
- **`/code/rundir`** - Working directory
- **`/code/prompt.json`** - Task instructions

## Agent Best Practices

### Implementation Guidelines
- 📖 **Read task carefully** from `/code/prompt.json`
- 🔍 **Analyze context files** before making changes
- ✏️ **Make targeted modifications** to solve the specific problem
- 🚪 **Exit cleanly** with code 0 when finished
- 📝 **Log progress** to help with debugging

### Local Testing
```bash
# Create test structure
mkdir -p test_agent/{docs,rtl,verif,rundir}
echo '{"prompt": "test task"}' > test_agent/prompt.json

# Create docker-compose.yml
cat > test_agent/docker-compose.yml << EOF
version: '3'
services:
  agent:
    image: my-agent
    volumes:
      - ./docs:/code/docs
      - ./rtl:/code/rtl
      - ./verif:/code/verif
      - ./rundir:/code/rundir
      - ./prompt.json:/code/prompt.json
    working_dir: /code
EOF

# Test your agent
cd test_agent && docker compose up
```

## Manual Debugging and Inspection Tools

For debugging agent behavior and test harness issues, the benchmark provides utility scripts for manual Docker container inspection and execution.

### Agent Debugging with `run_docker_agent.sh`

The `run_docker_agent.sh` script allows you to manually run and inspect your agent in the same environment used by the benchmark:

```bash
# Navigate to the harness directory for your datapoint
cd work/cvdp_agentic_my_issue/harness/1/

# Run your agent normally
./run_docker_agent.sh

# Debug mode - start with bash shell instead of running agent
./run_docker_agent.sh -d
```

The script supports only the `-d` flag for debug mode. In debug mode, the container starts with a bash shell instead of running the agent automatically, allowing you to inspect the environment and run commands manually.

**Interactive Debugging:**
```bash
# Enter agent container for manual inspection
cd work/cvdp_agentic_my_issue/harness/1/
./run_docker_agent.sh -d

# Inside the container, you can:
# - Examine the task: cat /code/prompt.json
# - Check available files: ls -la /code/*/
# - Run your agent manually: python /app/agent.py
# - Test tools: which xrun verilator yosys
# - Check environment: env | grep -E "(OPENAI|MODEL)"
```

### Test Harness Debugging with `run_docker_harness.sh`

The test harness scripts are generated with service-specific names (e.g., `run_docker_harness_harness.sh`). These scripts let you manually run the test harness to debug verification issues:

```bash
# Navigate to the harness directory for your datapoint
cd work/cvdp_agentic_my_issue/harness/1/

# Run test harness normally
./run_docker_harness_harness.sh

# Debug mode - start with bash shell
./run_docker_harness_harness.sh -d
```

**Interactive Debugging:**
```bash
# Enter harness container for manual inspection
cd work/cvdp_agentic_my_issue/harness/1/
./run_docker_harness_harness.sh -d

# Inside the container, you can:
# - Check test files: ls -la /code/
# - Debug compilation: cd verilator --lint-only *.v
# - Check tool availability: which xrun verilator yosys
```

### Advanced Debugging Workflows

**Agent Development Cycle:**
```bash
# 1. Run benchmark to generate work directory
python run_benchmark.py -f dataset.jsonl -i specific_issue -l -g my-agent

# 2. Navigate to results
cd work/cvdp_agentic_specific_issue/harness/1/

# 3. Inspect what happened
cat agent_changes.patch  # See what agent changed

# 4. Debug agent and test harness interactively  
./run_docker_agent.sh -d
# Inside container: examine environment, run agent manually, test changes
```

### Environment and Volume Access

Both scripts preserve the same environment and volume mounts as the benchmark:

**Available Directories:**
- `/code/docs/` - Documentation files
- `/code/rtl/` - RTL source files  
- `/code/verif/` - Verification files
- `/code/rundir/` - Working directory
- `/code/prompt.json` - Task description (agent script only)

**Environment Variables:**
- Same environment as benchmark execution
- `VERIF_EDA_IMAGE` settings respected
- License configurations preserved

**Network Access:**
- External network access (for downloads, license servers)
- Isolated from other containers unless explicitly configured

### Common Debugging Patterns

**Agent Not Working:**
```bash
# 1. Navigate to harness directory and start agent container in debug mode
cd work/cvdp_agentic_my_issue/harness/1/
./run_docker_agent.sh -d

# Inside container:
# - Check if agent runs: python --version
# - Check tool availability: which xrun verilator yosys
# - Check file permissions: ls -la /code/
# - Run agent manually: python /app/agent.py
```

**Test Failures:**
```bash
# 1. Navigate to harness directory and start harness container in debug mode
cd work/cvdp_agentic_my_issue/harness/1/
./run_docker_harness_harness.sh -d

# Inside container:
# - Run tests manually: cd /code && make test
# - Check for missing files: find /code -name "*.v" -o -name "*.sv"
# - Validate tool versions: verilator --version
# - Check compilation: verilator --lint-only *.v
```

### Script Locations and Requirements

**Generated Scripts:**
- Scripts are automatically created in each work directory after benchmark execution
- `run_docker_agent.sh` - Only present for agentic evaluations (`-g` flag)
- `run_docker_harness_*.sh` - Present for all evaluations with test harness (named by service)

**Script Interface:**
- Both scripts support only the `-d` flag for debug mode
- No arbitrary command arguments are supported
- Scripts are self-contained and execute specific docker-compose configurations

**Prerequisites:**
- Docker must be installed and accessible
- Same Docker images must be available as used by benchmark
- Work directory must be from a completed (or interrupted) benchmark run

These scripts provide the same environment as the benchmark but allow for interactive debugging and manual step-through of the evaluation process.

## Dataset Format

Agentic datasets include additional fields for Docker execution:

```json
{
  "id": "cvdp_agentic_name_0001", 
  "categories": ["category", "difficulty"],
  "system_message": "System instructions for the agent",
  "prompt": "The task instruction here",
  "context": {
    "file_path.v": "file content here"
  },
  "patch": {
    "file_to_patch.v": "expected changes in diff format"
  },
  "harness": {
    "files": {
      "docker-compose.yml": "test harness configuration"
    }
  }
}
```

## Results Analysis

### Generated Files
- **`report.json`** - Aggregated results with metrics
- **`raw_result.json`** - Detailed test execution data
- **`agent_changes.patch`** - Complete record of agent modifications
- **`composite_report.json`** - Multi-sample aggregation (from run_samples.py)

### Analysis Tools
```bash
# Basic report analysis
python run_reporter.py work/report.json

# Multi-sample pass@k analysis
python run_reporter.py work_composite/composite_report.json

# Analyze agent changes
cat work/cvdp_agentic_my_issue_0001/agent_changes.patch
```

### Key Metrics
- **Pass/fail status** per test case
- **Problem-level success** (all tests passing)
- **Agent modifications** tracked in patch files
- **Category and difficulty** breakdowns
- **Pass@k probabilities** (multi-sample runs)

---

## Next Steps

- 🤖 **Try Non-Agentic Workflow**: [LLM-based evaluation](README_NON_AGENTIC.md)
- 🛠️ **Agent Examples**: [Complete development guide](examples/README.md)
- ⚙️ **Advanced Config**: [See main README](README.md) for complete options
- 📊 **Dataset Tools**: Analyze and manipulate datasets with `tools/` utilities