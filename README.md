# 🧬 Causal-MCP

> A comprehensive causal inference platform for biomedical data analysis

Perform Mendelian Randomization analysis to investigate causal relationships between genes and diseases using real GWAS and eQTL data.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- OpenGWAS JWT token (for real GWAS data access)

### Option 1: Local Installation (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
cd client-app
streamlit run app.py

# Access the web interface at http://localhost:8501
```

### Option 2: Docker Deployment

```bash
# Navigate to project directory
cd causal-mcp

# Start all services
docker-compose up -d

# Access the web interface at http://localhost:8501
```

## ⚙️ Configuration Guide

### Step 1: OpenGWAS JWT Token Setup

**Required for accessing real GWAS data:**

1. Visit [OpenGWAS website](https://api.opengwas.io/profile/)
2. Register and obtain your JWT token
3. In the web interface:
   - Click "Configuration" in the sidebar
   - Enter your JWT token
   - Click "Save Configuration"
   - You should see "✅ JWT token validated successfully"

### Step 2: LLM Configuration (Optional)

**For AI-powered result interpretation:**

**Option A: Cloud LLM (Recommended)**

- **DeepSeek**: Get API key from [DeepSeek Platform](https://platform.deepseek.com/)
- **Gemini**: Get API key from [Google AI Studio](https://makersuite.google.com/)
- Configure in web interface: Provider → DeepSeek/Gemini, enter API key

**Option B: Local LLM (Free)**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull deepseek-r1:1.5b

# In web interface: Provider → Ollama, Model → deepseek-r1:1.5b
```

## � How to Use

### Basic Analysis Workflow

1. **Launch the Application**

   - Open http://localhost:8501 in your browser
   - Ensure JWT token is configured (see Configuration Guide)

2. **Choose Input Method**

   The system provides two input approaches:

   **🤖 Smart Chat Mode (Recommended)**

   - Use natural language to describe your analysis needs
   - AI automatically extracts gene, disease, and tissue information
   - More intuitive and user-friendly

   **📝 Traditional Form Mode**

   - Manually enter gene symbol, disease/trait, and tissue type
   - More precise control over parameters
   - Suitable for experienced users

3. **Run Analysis**
   - Click "Start Intelligent Analysis" (Smart Mode) or "Start Causal Analysis" (Form Mode)
   - Wait for data retrieval and processing
   - Review results and interpretations

### Example Analyses

#### 🤖 Smart Chat Mode Examples

**Natural Language Inputs:**

- "Analyze the causal relationship between PCSK9 gene and coronary heart disease"
- "Study the effect of IL6R gene on coronary heart disease"
- "Explore the association between SORT1 gene and type 2 diabetes"
- "Does LPA gene cause coronary heart disease?"
- "分析 PCSK9 基因与冠心病的因果关系" (Chinese)

**What the AI automatically identifies:**

- 🧬 **Gene names**: PCSK9, IL6R, SORT1, LPA, etc.
- 🏥 **Disease names**: coronary heart disease, type 2 diabetes, cardiovascular disease, etc.
- 🧪 **Tissue types**: whole blood, liver, brain, etc. (auto-recommended based on disease)

#### 📝 Traditional Form Mode Examples

**Example 1: Cholesterol Regulation**

```
Gene: PCSK9
Disease: Coronary artery disease
Tissue: Whole_Blood
```

**Example 2: Inflammation Pathway**

```
Gene: IL6R
Disease: Rheumatoid arthritis
Tissue: Whole_Blood
```

**Example 3: Lipid Metabolism**

```
Gene: SORT1
Disease: Type 2 diabetes
Tissue: Liver
```

## � Deployment

## 📊 Understanding Results

### Analysis Output Components

1. **MR Analysis Results**

   - **IVW (Inverse Variance Weighted)**: Primary causal estimate
   - **MR-Egger**: Tests for pleiotropy and provides adjusted estimate
   - **Weighted Median**: Robust estimate when up to 50% of instruments are invalid
   - **P-values**: Statistical significance of causal effects

2. **Visualization Plots**

   - **Forest Plot**: Shows effect sizes and confidence intervals
   - **Scatter Plot**: Displays SNP effects on exposure vs. outcome
   - **Funnel Plot**: Detects potential pleiotropy and bias

3. **Quality Assessment**

   - **Data Quality Score**: Overall reliability of the analysis
   - **Heterogeneity Tests**: Consistency across instrumental variables
   - **Sensitivity Analysis**: Robustness of findings

4. **AI Interpretation** (if configured)
   - **Biological Context**: Gene function and pathway information
   - **Clinical Relevance**: Potential therapeutic implications
   - **Study Limitations**: Important caveats and considerations

### Interpreting P-values

- **P < 0.05**: Statistically significant causal effect
- **P ≥ 0.05**: No significant causal effect detected
- **Consistent results across methods**: Stronger evidence for causality

## �📊 Core Modules

## 🔧 Troubleshooting

### Common Issues

**1. JWT Token Errors**

```
Error: "OpenGWAS JWT token not available"
Solution: Configure JWT token in the web interface
```

**2. No eQTL Data Found**

```
Error: "No significant eQTLs found"
Solution: Try different tissue types or check gene symbol spelling
```

**3. GWAS Data Not Found**

```
Error: "No GWAS data found for trait"
Solution: Use more specific disease names or try alternative terms
```

**4. Docker Issues**

```
Error: Port already in use
Solution: docker-compose down && docker-compose up -d
```

### Getting Help

- Check the application logs for detailed error messages
- Ensure all prerequisites are installed
- Verify internet connection for API access
- Try with well-known gene-disease pairs first

## 🏗️ System Overview

**Four Microservices:**

- **eQTL Server**: GTEx expression data
- **GWAS Server**: OpenGWAS association data
- **MR Server**: Statistical analysis engine
- **Knowledge Server**: Biological annotations

**Data Sources:**

- GTEx (Gene expression data)
- OpenGWAS (GWAS summary statistics)
- STRING (Protein interactions)
- KEGG (Pathway information)

## 💡 Input Mode Comparison

### 🤖 Smart Chat Mode vs 📝 Traditional Form Mode

| Feature              | Smart Chat Mode                       | Traditional Form Mode               |
| -------------------- | ------------------------------------- | ----------------------------------- |
| **Ease of Use**      | ⭐⭐⭐⭐⭐ Natural language           | ⭐⭐⭐ Requires parameter knowledge |
| **Flexibility**      | ⭐⭐⭐⭐⭐ Various phrasings accepted | ⭐⭐ Exact format required          |
| **Speed**            | ⭐⭐⭐ AI processing time             | ⭐⭐⭐⭐⭐ Immediate                |
| **Accuracy**         | ⭐⭐⭐⭐ AI interpretation            | ⭐⭐⭐⭐⭐ User-controlled          |
| **Language Support** | ⭐⭐⭐⭐⭐ English & Chinese          | ⭐⭐⭐ English primarily            |
| **LLM Dependency**   | ❌ Requires LLM setup                 | ✅ No LLM needed                    |

### When to Use Each Mode

**Use Smart Chat Mode when:**

- You're new to Mendelian Randomization analysis
- You prefer describing research questions naturally
- You want AI assistance in parameter selection
- You're working with complex or ambiguous trait names

**Use Traditional Form Mode when:**

- You know exact gene symbols and trait IDs
- You want precise control over tissue selection
- You're conducting systematic analyses
- LLM is not configured or available
