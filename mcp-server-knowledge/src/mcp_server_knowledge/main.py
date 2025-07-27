# src/mcp_server_knowledge/main.py
from mcp.server.fastmcp import FastMCP
import logging

from .models import (
    PathwayAnnotationInput, PathwayAnnotationOutput,
    GeneAnnotationInput, GeneAnnotationOutput,
    DrugTargetInput, DrugTargetOutput
)
from .knowledge_apis import BiologicalKnowledgeAPI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 初始化 FastMCP 服务器
mcp = FastMCP(
    name="mcp-server-knowledge",
    description="A server providing biological knowledge and pathway analysis tools."
)

# 初始化生物学知识API (延迟初始化)
knowledge_api = None

def get_knowledge_api():
    """Get or create knowledge API instance."""
    global knowledge_api
    if knowledge_api is None:
        knowledge_api = BiologicalKnowledgeAPI()
    return knowledge_api

@mcp.tool()
async def get_pathway_annotation(params: PathwayAnnotationInput) -> PathwayAnnotationOutput:
    """
    Gets pathway annotation information for a single gene.

    This tool takes a gene symbol and returns the biological pathways and processes
    that the gene participates in. This is annotation (not statistical enrichment analysis).
    Useful for understanding the biological context of a gene for LLM interpretation.
    """
    logging.info(f"🧬 获取基因 {params.gene_symbol} 的通路注释信息")
    logging.info("🔍 从真实的生物学数据库获取通路信息，用于LLM解释")

    # Validate input
    if not params.gene_symbol or not params.gene_symbol.strip():
        raise ValueError("Gene symbol cannot be empty")
    
    try:
        # Get pathway annotation for single gene
        api = get_knowledge_api()
        pathways = await api.get_gene_pathways(
            params.gene_symbol,
            organism="hsa" if params.organism == "hsapiens" else params.organism
        )

        # Generate summary
        summary = {
            "gene_symbol": params.gene_symbol,
            "total_pathways_found": len(pathways),
            "organism": params.organism,
            "data_sources": list(set(p.source_database for p in pathways)) if pathways else []
        }

        # Generate interpretation
        interpretation = _generate_gene_pathway_interpretation(pathways, params.gene_symbol)

        logging.info(f"✅ 通路注释完成: 基因 {params.gene_symbol} 参与 {len(pathways)} 个通路")

        return PathwayAnnotationOutput(
            gene_symbol=params.gene_symbol,
            pathways=pathways,
            interpretation=interpretation
        )
        
    except Exception as e:
        logging.error(f"Error in pathway annotation: {e}")
        raise ValueError(f"Pathway annotation failed: {str(e)}")

@mcp.tool()
async def annotate_gene(params: GeneAnnotationInput) -> GeneAnnotationOutput:
    """
    Provides comprehensive functional annotation for a gene.
    
    This tool retrieves detailed information about a gene including its function,
    protein interactions, disease associations, and druggability assessment.
    """
    logging.info(f"🧬 开始基因注释分析: {params.gene_symbol}")
    logging.info("🔍 严格使用真实的生物学数据库，严禁使用模拟数据")
    
    try:
        # Get basic gene annotation
        api = get_knowledge_api()
        annotation_data = await api.get_gene_annotation(params.gene_symbol)

        # Get protein interactions if requested
        protein_interactions = []
        if params.include_interactions:
            protein_interactions = await api.get_protein_interactions(params.gene_symbol)

        # Get disease associations if requested
        disease_associations = []
        if params.include_diseases:
            disease_associations = await api.get_disease_associations(params.gene_symbol)
        
        # Generate interpretation
        interpretation = _generate_gene_interpretation(
            params.gene_symbol, 
            annotation_data, 
            protein_interactions, 
            disease_associations
        )
        
        logging.info(f"✅ 基因注释分析完成: {params.gene_symbol}（基于真实数据库）")
        
        # 适配输出格式
        gene_info = {
            "gene_symbol": annotation_data.get("gene_symbol", params.gene_symbol),
            "species": annotation_data.get("species", "Homo sapiens"),
            "description": annotation_data.get("summary", f"Gene annotation for {params.gene_symbol}"),
            "uniprot_data": annotation_data.get("uniprot_data", {}),
            "data_sources": annotation_data.get("data_sources", [])
        }

        return GeneAnnotationOutput(
            gene_info=gene_info,
            functional_annotation=annotation_data.get("functional_annotation", {}),
            protein_interactions=protein_interactions,
            disease_associations=disease_associations,
            interpretation=interpretation
        )
        
    except Exception as e:
        logging.error(f"Error in gene annotation: {e}")
        raise ValueError(f"Gene annotation failed: {str(e)}")

@mcp.tool()
async def analyze_drug_targets(params: DrugTargetInput) -> DrugTargetOutput:
    """
    Analyzes drug targeting opportunities for a gene/protein.
    
    This tool identifies existing drugs that target the specified gene/protein,
    assesses druggability, and provides insights into therapeutic opportunities.
    """
    logging.info(f"Starting drug target analysis for: {params.gene_symbol}")
    
    try:
        # Get basic gene information
        api = get_knowledge_api()
        gene_annotation = await api.get_gene_annotation(params.gene_symbol)

        # Get targeting drugs
        targeting_drugs = await api.get_drug_targets(params.gene_symbol)
        
        # Assess druggability
        druggability_assessment = _assess_druggability(params.gene_symbol, targeting_drugs)
        
        # Generate therapeutic opportunities summary
        therapeutic_opportunities = _generate_therapeutic_summary(
            params.gene_symbol, 
            targeting_drugs, 
            druggability_assessment
        )
        
        logging.info(f"Drug target analysis completed for {params.gene_symbol}")
        
        return DrugTargetOutput(
            target_info=gene_annotation["gene_info"],
            targeting_drugs=targeting_drugs,
            druggability_assessment=druggability_assessment,
            therapeutic_opportunities=therapeutic_opportunities
        )
        
    except Exception as e:
        logging.error(f"Error in drug target analysis: {e}")
        raise ValueError(f"Drug target analysis failed: {str(e)}")

def _generate_gene_pathway_interpretation(pathways, gene_symbol) -> str:
    """Generate interpretation of gene pathway annotation results."""
    if not pathways:
        return (
            f"No pathway information was found for gene {gene_symbol}. "
            "This could indicate that the gene is not well-characterized in pathway databases "
            "or that it has unique functions not captured in standard pathway annotations."
        )

    interpretation_parts = []

    # Header
    interpretation_parts.append(
        f"## Pathway Annotation for Gene {gene_symbol}\n"
        f"**Gene analyzed:** {gene_symbol}\n"
        f"**Pathways found:** {len(pathways)}\n"
        f"**Data sources:** {', '.join(set(p.source_database for p in pathways))}\n"
    )

    # Top pathways
    interpretation_parts.append("### Key Pathways")

    for i, pathway in enumerate(pathways[:5]):  # Top 5 pathways
        interpretation_parts.append(
            f"{i+1}. **{pathway.pathway_name}** ({pathway.source_database})\n"
            f"   - Role: {pathway.gene_role or 'Participates in this pathway'}\n"
            f"   - Description: {pathway.description}\n"
        )
    
    # Biological interpretation
    interpretation_parts.append("\n### Biological Interpretation")

    # Group pathways by theme
    metabolism_pathways = [p for p in pathways if "metabolism" in p.pathway_name.lower()]
    signaling_pathways = [p for p in pathways if any(term in p.pathway_name.lower() for term in ["signaling", "pathway", "cascade"])]
    disease_pathways = [p for p in pathways if any(term in p.pathway_name.lower() for term in ["disease", "cancer", "disorder"])]

    if metabolism_pathways:
        interpretation_parts.append(
            f"Gene {gene_symbol} participates in **metabolic pathways** "
            f"({len(metabolism_pathways)} pathways), suggesting important roles "
            f"in cellular metabolism and energy homeostasis."
        )

    if signaling_pathways:
        interpretation_parts.append(
            f"The gene is also involved in **signaling pathways** ({len(signaling_pathways)} pathways), "
            f"indicating roles in cellular communication and response mechanisms."
        )

    if disease_pathways:
        interpretation_parts.append(
            f"Involvement in **disease-related pathways** ({len(disease_pathways)} pathways) "
            f"suggests potential therapeutic relevance of this gene."
        )

    # Summary for LLM
    interpretation_parts.append(
        f"\n### Summary for Analysis\n"
        f"Gene {gene_symbol} is involved in {len(pathways)} biological pathways, "
        f"providing context for understanding its role in biological processes and potential disease mechanisms."
    )

    return "\n".join(interpretation_parts)

def _generate_gene_interpretation(gene_symbol, annotation_data, interactions, diseases) -> str:
    """Generate interpretation of gene annotation results."""
    interpretation_parts = []
    
    # Header with data quality indicator
    data_sources = annotation_data.get("data_sources", [])
    quality_indicator = "🟢 高质量" if len(data_sources) >= 2 else "🟡 中等质量" if len(data_sources) == 1 else "🔴 有限数据"

    interpretation_parts.append(
        f"## 基因 {gene_symbol} 功能注释报告 {quality_indicator}\n"
    )

    # 数据来源说明
    if data_sources:
        interpretation_parts.append(f"**数据来源:** {', '.join(data_sources)}")

    # 基因摘要（如果可用）
    if annotation_data.get("summary"):
        interpretation_parts.append(f"\n### 📋 基因摘要")
        interpretation_parts.append(annotation_data["summary"])

    # UniProt详细信息
    uniprot_data = annotation_data.get("uniprot_data", {})
    if uniprot_data:
        interpretation_parts.append(f"\n### 🧬 蛋白质信息")

        if uniprot_data.get("protein_name"):
            interpretation_parts.append(f"**蛋白质名称:** {uniprot_data['protein_name']}")

        if uniprot_data.get("uniprot_id"):
            interpretation_parts.append(f"**UniProt ID:** {uniprot_data['uniprot_id']}")

        if uniprot_data.get("length"):
            interpretation_parts.append(f"**蛋白质长度:** {uniprot_data['length']} 氨基酸")

    # 功能注释详情
    functional_annotation = annotation_data.get("functional_annotation", {})
    if functional_annotation:
        interpretation_parts.append(f"\n### ⚙️ 功能注释")

        confidence = functional_annotation.get("confidence", "unknown")
        interpretation_parts.append(f"**数据置信度:** {confidence}")

        if functional_annotation.get("molecular_function"):
            function_text = functional_annotation["molecular_function"]
            if len(function_text) > 300:
                function_text = function_text[:300] + "..."
            interpretation_parts.append(f"**分子功能:** {function_text}")

        if functional_annotation.get("subcellular_location"):
            locations = functional_annotation["subcellular_location"]
            if isinstance(locations, list):
                locations = ", ".join(locations)
            interpretation_parts.append(f"**亚细胞定位:** {locations}")

    # 蛋白质相互作用分析
    if interactions:
        interpretation_parts.append(f"\n### 🔗 蛋白质相互作用网络")
        interpretation_parts.append(f"发现 {len(interactions)} 个蛋白质相互作用:")

        # 按置信度排序并显示前5个
        sorted_interactions = sorted(interactions,
                                   key=lambda x: x.confidence_score if x.confidence_score else 0,
                                   reverse=True)

        for i, interaction in enumerate(sorted_interactions[:5], 1):
            confidence = f"{interaction.confidence_score:.3f}" if interaction.confidence_score else "N/A"
            interpretation_parts.append(
                f"{i}. **{interaction.partner_gene}** - {interaction.interaction_type} (置信度: {confidence})"
            )

        if len(interactions) > 5:
            interpretation_parts.append(f"... 还有 {len(interactions) - 5} 个相互作用")
    else:
        interpretation_parts.append(f"\n### 🔗 蛋白质相互作用网络")
        interpretation_parts.append("⚠️ 未找到蛋白质相互作用数据")

    # 疾病关联分析
    if diseases:
        interpretation_parts.append(f"\n### 🏥 疾病关联")
        interpretation_parts.append(f"发现 {len(diseases)} 个疾病关联:")

        for i, disease in enumerate(diseases[:5], 1):
            interpretation_parts.append(
                f"{i}. **{disease.disease_name}** - {disease.association_type} ({disease.evidence_level}证据)"
            )

        if len(diseases) > 5:
            interpretation_parts.append(f"... 还有 {len(diseases) - 5} 个疾病关联")
    else:
        interpretation_parts.append(f"\n### 🏥 疾病关联")
        interpretation_parts.append("⚠️ 未找到疾病关联数据")

    # 数据质量和建议
    interpretation_parts.append(f"\n### 📊 数据质量评估")

    total_data_points = len(data_sources) + len(interactions) + len(diseases)
    if total_data_points >= 5:
        interpretation_parts.append("✅ **数据完整性:** 优秀 - 获得了丰富的功能注释信息")
    elif total_data_points >= 2:
        interpretation_parts.append("⚠️ **数据完整性:** 良好 - 获得了基本的功能注释信息")
    else:
        interpretation_parts.append("❌ **数据完整性:** 有限 - 建议查阅其他数据库或文献")

    # 使用建议
    interpretation_parts.append(f"\n### 💡 使用建议")
    if interactions:
        interpretation_parts.append("- 可以基于蛋白质相互作用网络进行通路分析")
    if diseases:
        interpretation_parts.append("- 可以进一步研究疾病关联的分子机制")
    if not interactions and not diseases:
        interpretation_parts.append("- 建议查阅最新文献获取更多功能信息")
        interpretation_parts.append("- 可以尝试同源基因或蛋白质家族分析")
    
    return "\n".join(interpretation_parts)

def _assess_druggability(gene_symbol, targeting_drugs) -> dict:
    """Assess druggability of a gene/protein."""
    assessment = {
        "druggability_score": 0.0,
        "assessment_level": "Unknown",
        "existing_drugs": len(targeting_drugs),
        "development_opportunities": "Limited information available"
    }
    
    if targeting_drugs:
        approved_drugs = [d for d in targeting_drugs if d.development_stage == "Approved"]
        clinical_drugs = [d for d in targeting_drugs if "Phase" in d.development_stage]
        
        if approved_drugs:
            assessment["druggability_score"] = 0.9
            assessment["assessment_level"] = "High - Validated target"
            assessment["development_opportunities"] = f"Validated target with {len(approved_drugs)} approved drugs"
        elif clinical_drugs:
            assessment["druggability_score"] = 0.7
            assessment["assessment_level"] = "Moderate - Under investigation"
            assessment["development_opportunities"] = f"Target under investigation with {len(clinical_drugs)} drugs in clinical trials"
        else:
            assessment["druggability_score"] = 0.5
            assessment["assessment_level"] = "Moderate - Some evidence"
            assessment["development_opportunities"] = "Some targeting compounds identified"
    
    return assessment

def _generate_therapeutic_summary(gene_symbol, targeting_drugs, druggability) -> str:
    """Generate therapeutic opportunities summary."""
    if not targeting_drugs:
        return (
            f"No known drugs currently target {gene_symbol}. "
            f"This represents a potential opportunity for drug development, "
            f"pending further validation of therapeutic relevance."
        )
    
    approved_drugs = [d for d in targeting_drugs if d.development_stage == "Approved"]
    
    if approved_drugs:
        drug_names = [d.drug_name for d in approved_drugs]
        return (
            f"{gene_symbol} is a validated therapeutic target with {len(approved_drugs)} "
            f"approved drugs: {', '.join(drug_names)}. This demonstrates the clinical "
            f"relevance and druggability of this target."
        )
    else:
        return (
            f"{gene_symbol} is under investigation as a therapeutic target with "
            f"{len(targeting_drugs)} compounds in development. This suggests emerging "
            f"therapeutic potential that requires further validation."
        )

# 3. 配置服务器入口点
def run():
    mcp.run()

if __name__ == "__main__":
    run()
