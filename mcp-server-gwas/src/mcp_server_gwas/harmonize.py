# src/mcp_server_gwas/harmonize.py
"""
Data harmonization logic for GWAS and eQTL data.
This module handles the critical task of ensuring that effect estimates
from different sources are aligned to the same reference allele.
"""

from typing import List, Tuple, Optional
import logging
from .models import SNPInstrument, HarmonizedDataPoint

logger = logging.getLogger(__name__)

def is_palindromic(allele1: str, allele2: str) -> bool:
    """
    Check if a SNP is palindromic (A/T or C/G).
    Palindromic SNPs are ambiguous and difficult to harmonize.
    """
    palindromic_pairs = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
    return (allele1.upper(), allele2.upper()) in palindromic_pairs

def flip_alleles(allele1: str, allele2: str) -> Tuple[str, str]:
    """
    Flip alleles to match the reference.
    """
    return allele2, allele1

def harmonize_snp_data(
    exposure_snp: SNPInstrument,
    outcome_data: dict,
    outcome_study_id: str = "unknown"
) -> Optional[HarmonizedDataPoint]:
    """
    Harmonize a single SNP's exposure and outcome data.
    
    Args:
        exposure_snp: SNP data from eQTL analysis
        outcome_data: SNP data from GWAS analysis
        
    Returns:
        HarmonizedDataPoint if harmonization successful, None otherwise
    """
    
    # Extract outcome alleles and effects
    outcome_effect_allele = outcome_data.get("effect_allele", "").upper()
    outcome_other_allele = outcome_data.get("other_allele", "").upper()
    outcome_beta = outcome_data.get("beta", 0.0)
    outcome_se = outcome_data.get("se", 0.0)
    outcome_pval = outcome_data.get("pval", 1.0)
    
    # Extract exposure alleles and effects
    exposure_effect_allele = exposure_snp.effect_allele.upper()
    exposure_other_allele = exposure_snp.other_allele.upper()
    
    # Check for palindromic SNPs
    if is_palindromic(exposure_effect_allele, exposure_other_allele):
        logger.warning(f"Palindromic SNP {exposure_snp.snp_id} excluded from analysis")
        return None
    
    # Check if alleles match directly
    if (exposure_effect_allele == outcome_effect_allele and 
        exposure_other_allele == outcome_other_allele):
        # Perfect match - no flipping needed
        harmonization_status = "Harmonized: alleles matched."
        
    elif (exposure_effect_allele == outcome_other_allele and 
          exposure_other_allele == outcome_effect_allele):
        # Alleles are flipped - need to flip outcome effects
        outcome_beta = -outcome_beta
        outcome_effect_allele, outcome_other_allele = flip_alleles(
            outcome_effect_allele, outcome_other_allele
        )
        harmonization_status = "Harmonized: alleles flipped."
        
    else:
        # Alleles don't match - exclude this SNP
        logger.warning(
            f"SNP {exposure_snp.snp_id} excluded: allele mismatch. "
            f"Exposure: {exposure_effect_allele}/{exposure_other_allele}, "
            f"Outcome: {outcome_effect_allele}/{outcome_other_allele}"
        )
        return None
    
    # Create harmonized data point
    harmonized_point = HarmonizedDataPoint(
        SNP=exposure_snp.snp_id,
        beta_exposure=exposure_snp.beta,
        se_exposure=exposure_snp.se,
        pval_exposure=exposure_snp.p_value,
        beta_outcome=outcome_beta,
        se_outcome=outcome_se,
        pval_outcome=outcome_pval,
        effect_allele=exposure_effect_allele,
        other_allele=exposure_other_allele,
        harmonization_status=harmonization_status,
        outcome_study_id=outcome_study_id
    )
    
    return harmonized_point

def harmonize_datasets(
    exposure_instruments: List[SNPInstrument],
    outcome_data: dict,
    outcome_study_id: str = "unknown"
) -> Tuple[List[HarmonizedDataPoint], List[str]]:
    """
    Harmonize exposure and outcome datasets.
    
    Args:
        exposure_instruments: List of SNP instruments from eQTL analysis
        outcome_data: Dictionary mapping SNP IDs to outcome data
        
    Returns:
        Tuple of (harmonized_data_points, excluded_snp_ids)
    """
    harmonized_data = []
    excluded_snps = []
    
    for exposure_snp in exposure_instruments:
        snp_id = exposure_snp.snp_id
        
        if snp_id not in outcome_data:
            logger.warning(f"SNP {snp_id} not found in outcome data")
            excluded_snps.append(snp_id)
            continue
        
        harmonized_point = harmonize_snp_data(exposure_snp, outcome_data[snp_id], outcome_study_id)
        
        if harmonized_point is not None:
            harmonized_data.append(harmonized_point)
        else:
            excluded_snps.append(snp_id)
    
    logger.info(
        f"Harmonization complete: {len(harmonized_data)} SNPs harmonized, "
        f"{len(excluded_snps)} SNPs excluded"
    )
    
    return harmonized_data, excluded_snps
