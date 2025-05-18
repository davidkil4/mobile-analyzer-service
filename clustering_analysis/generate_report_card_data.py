#!/usr/bin/env python3
"""
Script to extract and combine key data from clustering analysis files
to create a concise input for report card generation.
"""

import json
import os
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_report_card_data(primary_json_path, secondary_json_path, secondary_prioritized_json_path, output_path, analysis_output_path=None, recommendation_output_path=None):
    """
    Extract and combine key data from analysis files to create concise inputs for report card generation.
    
    Args:
        primary_json_path: Path to the primary analysis JSON file
        secondary_json_path: Path to the secondary analysis JSON file
        secondary_prioritized_json_path: Path to the secondary prioritized JSON file
        output_path: Path to save the combined output file (legacy, contains everything)
        analysis_output_path: Path to save the analysis data for LLM 1 (Parts 1 & 2)
        recommendation_output_path: Path to save the recommendation data for LLM 2 (Part 3)
    """
    logger.info(f"Loading primary data from {primary_json_path}")
    with open(primary_json_path, 'r') as f:
        primary_data = json.load(f)
    
    logger.info(f"Loading secondary data from {secondary_json_path}")
    with open(secondary_json_path, 'r') as f:
        secondary_data = json.load(f)
    
    logger.info(f"Loading prioritized data from {secondary_prioritized_json_path}")
    with open(secondary_prioritized_json_path, 'r') as f:
        prioritized_data = json.load(f)
    
    # Create the combined data structure
    logger.info("Creating combined report data structure")
    
    # Extract components
    tendency_zone = extract_tendency_zone(primary_data)
    global_stats = extract_global_stats(primary_data)
    zone_stats = extract_zone_stats_from_secondary(secondary_data)
    
    # Remove the Filtered zone as it's not needed
    if "Filtered" in zone_stats:
        del zone_stats["Filtered"]
    
    # Mark which zone contains the tendency zone's complexity score
    tendency_complexity = tendency_zone.get("complexity", 0)
    logger.info(f"Identifying which zone contains tendency complexity: {tendency_complexity}")
    for zone_name, zone_data in zone_stats.items():
        complexity_bounds = zone_data.get("complexity_bounds", [0, 0])
        # Handle None values in complexity bounds
        lower_bound = complexity_bounds[0] if complexity_bounds[0] is not None else 0
        upper_bound = complexity_bounds[1] if complexity_bounds[1] is not None else float('inf')
        
        if lower_bound <= tendency_complexity <= upper_bound:
            logger.info(f"Zone {zone_name} contains the tendency zone complexity")
            zone_data["contains_tendency_zone"] = True
        else:
            zone_data["contains_tendency_zone"] = False
    
    # Extract focus areas (skip strengths as they're not needed)
    focus_areas = extract_focus_areas(prioritized_data)
    
    # Extract recommendations
    recommendations = extract_recommendations(prioritized_data)
    
    # Create a more prompt-aligned structure
    
    # 1. Sort zones by complexity for easier progression analysis
    sorted_zones = {}
    zone_names = list(zone_stats.keys())
    zone_names.sort(key=lambda z: zone_stats[z].get("avg_complexity", 0))
    
    for zone_name in zone_names:
        sorted_zones[zone_name] = zone_stats[zone_name]
    
    # 2. Create complexity and accuracy progression data
    complexity_progression = []
    accuracy_progression = []
    
    for zone_name in zone_names:
        zone_data = sorted_zones[zone_name]
        complexity_progression.append({
            "zone": zone_name,
            "complexity": zone_data.get("avg_complexity"),
            "utterance_count": zone_data.get("utterance_count"),
            "contains_tendency_zone": zone_data.get("contains_tendency_zone", False),
            "sample_utterances": zone_data.get("sample_utterances", [])
        })
        
        accuracy_progression.append({
            "zone": zone_name,
            "complexity": zone_data.get("avg_complexity"),
            "accuracy": zone_data.get("avg_accuracy"),
            "error_rates": zone_data.get("error_rates", {})
        })
    
    # 3. Organize errors by zone for easier analysis
    errors_by_zone = {}
    for error_type, error_data in focus_areas.items():
        for zone_name, examples in error_data.get("examples_by_zone", {}).items():
            if zone_name not in errors_by_zone:
                errors_by_zone[zone_name] = []
            
            for example in examples:
                errors_by_zone[zone_name].append({
                    "type": error_type,
                    "original": example.get("original"),
                    "corrected": example.get("corrected"),
                    "description": example.get("description"),
                    "severity": example.get("severity")
                })
    
    # 4. Create the reorganized structure
    report_data = {
        "learner_summary": {
            "tendency_zone_metrics": {
                "complexity": tendency_zone.get("complexity"),
                "accuracy": tendency_zone.get("accuracy"),
                "size": tendency_zone.get("size")
            },
            "global_stats": global_stats,
            "complexity_progression": complexity_progression,
            "accuracy_progression": accuracy_progression
        },
        "error_patterns": {
            "by_type": focus_areas,
            "by_zone": errors_by_zone
        },
        "recommendations": recommendations
    }
    
    # Round all numeric values in the report data
    logger.info("Rounding numeric values in report data")
    report_data = round_nested_values(report_data)
    
    # Write the combined data to the output file (legacy)
    logger.info(f"Writing combined report card data to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Create separate files for the two LLMs if paths are provided
    if analysis_output_path:
        # Create data for LLM 1 (Parts 1 & 2) - focus on description
        analysis_data = {
            "learner_summary": report_data["learner_summary"],
            "error_patterns": {
                "by_type": report_data["error_patterns"]["by_type"]
            }
        }
        
        logger.info(f"Writing analysis data for LLM 1 to {analysis_output_path}")
        with open(analysis_output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
    
    if recommendation_output_path:
        # Create data for LLM 2 (Part 3) - focus on recommendations
        recommendation_data = {
            "complexity_accuracy_relationship": {
                "tendency_zone_metrics": report_data["learner_summary"]["tendency_zone_metrics"],
                "complexity_progression": report_data["learner_summary"]["complexity_progression"],
                "accuracy_progression": report_data["learner_summary"]["accuracy_progression"]
            },
            "error_patterns": {
                "by_zone": report_data["error_patterns"]["by_zone"]
            },
            "recommendations": report_data["recommendations"]
        }
        
        logger.info(f"Writing recommendation data for LLM 2 to {recommendation_output_path}")
        with open(recommendation_output_path, 'w') as f:
            json.dump(recommendation_data, f, indent=2)
    
    logger.info(f"Report card data processing completed successfully")
    return report_data


def round_nested_values(obj, decimal_places=3):
    """Recursively round all numeric values in nested dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: round_nested_values(v, decimal_places) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_nested_values(item, decimal_places) for item in obj]
    elif isinstance(obj, float):
        return round(obj, decimal_places)
    else:
        return obj

def extract_tendency_zone(primary_data):
    """Extract tendency zone information from primary data"""
    logger.info("Extracting tendency zone information")
    zone = primary_data.get("tendency_zone", {})
    return {
        "zone_id": zone.get("zone_id"),
        "size": zone.get("size"),
        "complexity": zone.get("feature_averages", {}).get("complexity", {}).get("mean"),
        "accuracy": zone.get("feature_averages", {}).get("accuracy", {}).get("mean"),
        "error_distribution": zone.get("feature_averages", {}).get("error_distribution", {})
    }

def extract_global_stats(primary_data):
    """Extract global statistics from primary data"""
    logger.info("Extracting global statistics")
    global_stats = primary_data.get("global_stats", {})
    feature_averages = global_stats.get("feature_averages", {})
    
    return {
        "complexity": feature_averages.get("complexity", {}),
        "accuracy": feature_averages.get("accuracy", {}),
        "error_distribution": feature_averages.get("error_distribution", {}),
        "total_utterances": global_stats.get("total_utterances", 0)
    }

def extract_zone_stats_from_secondary(secondary_data):
    """Extract statistics for each ZPD zone from the secondary analysis JSON file"""
    logger.info("Extracting ZPD zone statistics from secondary data")
    
    zone_stats = {}
    
    # Extract regions from the secondary data
    regions = secondary_data.get("regions", {})
    
    # Process each region
    for region_name, region_data in regions.items():
        # Get basic statistics
        statistics = region_data.get("statistics", {})
        averages = statistics.get("averages", {})
        
        # Initialize zone stats
        zone_stats[region_name] = {
            "utterance_count": statistics.get("utterance_count", 0),
            "complexity_bounds": statistics.get("complexity_bounds", [0, 0]),
            "avg_complexity": averages.get("complexity", 0),
            "avg_accuracy": averages.get("accuracy", 0),
            "proficiency_tier": region_data.get("proficiency_tier", ""),
            "error_rates": {}
        }
        
        # Calculate error rates from individual utterances
        if "utterances" in region_data:
            total_utterances = len(region_data["utterances"])
            if total_utterances > 0:
                # Initialize error counters
                error_counts = {"critical": 0, "moderate": 0, "minor": 0}
                error_sums = {"critical": 0.0, "moderate": 0.0, "minor": 0.0}
                
                # Count errors in each utterance
                for utterance in region_data["utterances"]:
                    metrics = utterance.get("metrics", {})
                    errors = metrics.get("E", {})
                    
                    for severity, value in errors.items():
                        if value > 0:
                            error_counts[severity] += 1
                            error_sums[severity] += value
                
                # Calculate average error rates
                zone_stats[region_name]["error_rates"] = {
                    severity: {
                        "frequency": count / total_utterances,  # Percentage of utterances with this error type
                        "avg_severity": error_sums[severity] / count if count > 0 else 0  # Average severity when present
                    }
                    for severity, count in error_counts.items()
                }
                
                # Add sample utterances (2 max)
                zone_stats[region_name]["sample_utterances"] = [
                    {
                        "original": u.get("original", ""),
                        "corrected": u.get("corrected", ""),
                        "metrics": u.get("metrics", {})
                    }
                    for u in region_data["utterances"][:2]
                ]
    
    return zone_stats


def extract_zone_stats(primary_data):
    """Extract statistics for each ZPD zone from primary data (fallback method)"""
    logger.info("Extracting ZPD zone statistics from primary data")
    
    # First try to extract directly from the primary data structure if available
    # This is more reliable than reconstructing from clusters
    zone_stats = {}
    
    # Check if we have a secondary analysis section with ZPD regions
    if "secondary_analysis" in primary_data and "zpd_regions" in primary_data["secondary_analysis"]:
        logger.info("Found ZPD regions in secondary analysis")
        zpd_regions = primary_data["secondary_analysis"]["zpd_regions"]
        
        for region_name, region_data in zpd_regions.items():
            zone_stats[region_name] = {
                "utterance_count": region_data.get("utterance_count", 0),
                "avg_complexity": region_data.get("avg_complexity", 0),
                "avg_accuracy": region_data.get("avg_accuracy", 0),
                "error_rates": region_data.get("error_rates", {})
            }
        
        return zone_stats
    
    # If no secondary analysis, try to reconstruct from clusters
    logger.info("Reconstructing ZPD zones from clusters")
    zones = {}
    
    # Define standard ZPD zones if we need to reconstruct them
    standard_zones = ["ZPD/2", "ZPD-1", "ZPD", "ZPD+1"]
    for zone in standard_zones:
        zones[zone] = {
            "utterance_count": 0,
            "complexity_sum": 0,
            "accuracy_sum": 0,
            "utterances": []
        }
    
    # Process each cluster to identify its ZPD zone
    for cluster in primary_data.get("clusters", []):
        cluster_id = cluster.get("cluster_id")
        # Extract ZPD zone from cluster metadata if available
        zone_info = cluster.get("zone_info", {})
        zone_name = zone_info.get("zone_name")
        
        if zone_name and zone_name in zones:
            # Count utterances and sum metrics for this zone
            utterances = cluster.get("utterances", [])
            zones[zone_name]["utterance_count"] += len(utterances)
            
            for utterance in utterances:
                metrics = utterance.get("metrics", {})
                complexity = metrics.get("C", 0)
                accuracy = metrics.get("A", 0)
                
                zones[zone_name]["complexity_sum"] += complexity
                zones[zone_name]["accuracy_sum"] += accuracy
                zones[zone_name]["utterances"].append({
                    "original": utterance.get("original"),
                    "corrected": utterance.get("corrected"),
                    "metrics": metrics
                })
    
    # Calculate averages for each zone
    for zone_name, data in zones.items():
        count = data["utterance_count"]
        if count > 0:
            zone_stats[zone_name] = {
                "utterance_count": count,
                "avg_complexity": data["complexity_sum"] / count,
                "avg_accuracy": data["accuracy_sum"] / count,
                "sample_utterances": data["utterances"][:3]  # Include a few sample utterances
            }
    
    return zone_stats

def extract_strengths(primary_data):
    """Extract patterns that the learner uses correctly (limited to top 3)"""
    logger.info("Extracting learner strengths")
    strengths = []
    
    # Find utterances with high accuracy
    for cluster in primary_data.get("clusters", []):
        for utterance in cluster.get("utterances", []):
            # If accuracy is high (e.g., > 0.8) and there are patterns
            if utterance.get("metrics", {}).get("A", 0) > 0.8:
                # Check if original and corrected are the same (or very similar)
                original = utterance.get("original", "")
                corrected = utterance.get("corrected", "")
                
                # Simple check - if they're exactly the same or differ only by punctuation
                is_correct = original == corrected or original.strip('.,!?') == corrected.strip('.,!?')
                
                if is_correct:
                    # Extract patterns from clauses
                    for clause in utterance.get("clauses", []):
                        patterns = clause.get("clause_pattern_analysis", [])
                        for pattern in patterns:
                            if pattern.get("frequency_level", 0) >= 3.0:
                                # Streamline pattern data
                                streamlined_pattern = {
                                    "intention": pattern.get("intention"),
                                    "category": pattern.get("category"),
                                    "component": pattern.get("component")
                                }
                                
                                strengths.append({
                                    "pattern": streamlined_pattern,
                                    "example": utterance.get("original"),
                                    "metrics": utterance.get("metrics")
                                })
    
    # Deduplicate patterns by component
    unique_strengths = {}
    for strength in strengths:
        component = strength["pattern"].get("component")
        if component and component not in unique_strengths:
            unique_strengths[component] = strength
    
    # Return only top 3 strengths
    return list(unique_strengths.values())[:3]

def extract_focus_areas(prioritized_data):
    """Extract common error types from prioritized data (limited to top 5 types, with examples organized by zone)"""
    logger.info("Extracting focus areas")
    error_types = {}
    
    # First pass: collect all examples by error type and zone
    examples_by_type_and_zone = {}
    
    # Go through all zones and recommendations
    for zone in prioritized_data.get("analysis_zones", []):
        zone_name = zone.get("zone_name")
        for rec in zone.get("recommendations", []):
            for error in rec.get("errors", []):
                error_type = error.get("type")
                if error_type:
                    # Initialize data structures if needed
                    if error_type not in error_types:
                        error_types[error_type] = {
                            "count": 0,
                            "examples_by_zone": {}
                        }
                    
                    if error_type not in examples_by_type_and_zone:
                        examples_by_type_and_zone[error_type] = {}
                    
                    if zone_name not in examples_by_type_and_zone[error_type]:
                        examples_by_type_and_zone[error_type][zone_name] = []
                    
                    # Increment count
                    error_types[error_type]["count"] += 1
                    
                    # Add example to the appropriate zone
                    examples_by_type_and_zone[error_type][zone_name].append({
                        "original": rec.get("original"),
                        "corrected": rec.get("corrected"),
                        "description": error.get("description"),
                        "severity": error.get("severity"),
                        "zone": zone_name
                    })
    
    # Second pass: select top 2 examples from each zone for each error type
    for error_type, zones in examples_by_type_and_zone.items():
        error_types[error_type]["examples_by_zone"] = {}
        
        for zone_name, examples in zones.items():
            # Take up to 2 examples from each zone
            error_types[error_type]["examples_by_zone"][zone_name] = examples[:2]
    
    # Sort by count and take top errors
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1]["count"], reverse=True)
    return {error_type: data for error_type, data in sorted_errors[:5]}

def extract_recommendations(prioritized_data):
    """Extract top recommendations from each ZPD zone (limited to 2 per zone)"""
    logger.info("Extracting recommendations by zone")
    recommendations = {}
    
    for zone in prioritized_data.get("analysis_zones", []):
        zone_name = zone.get("zone_name")
        zone_recs = []
        
        # Take top 2 recommendations from each zone
        for rec in zone.get("recommendations", [])[:2]:
            # Streamline the data structure
            streamlined_rec = {
                "original": rec.get("original"),
                "corrected": rec.get("corrected"),
                "priority_score": rec.get("priority_score")
            }
            
            # Include only essential error information
            if rec.get("errors"):
                streamlined_rec["error_types"] = [error.get("type") for error in rec.get("errors", [])[:2]]
            
            zone_recs.append(streamlined_rec)
        
        recommendations[zone_name] = zone_recs
    
    return recommendations

def main():
    """Main function to run the script"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate report card data from clustering analysis outputs")
    parser.add_argument("primary_json", help="Path to the primary analysis JSON file")
    parser.add_argument("secondary_json", help="Path to the secondary analysis JSON file")
    parser.add_argument("secondary_prioritized_json", help="Path to the secondary prioritized JSON file")
    parser.add_argument("output", help="Path to save the combined output file")
    parser.add_argument("--analysis_output", help="Path to save the analysis data for LLM 1 (Parts 1 & 2)")
    parser.add_argument("--recommendation_output", help="Path to save the recommendation data for LLM 2 (Part 3)")
    
    args = parser.parse_args()
    
    # Get the base directory (mobile-analyzer-service)
    base_dir = Path(__file__).parent.parent
    
    # Use command-line arguments for file paths
    primary_json_path = args.primary_json
    secondary_json_path = args.secondary_json
    secondary_prioritized_json_path = args.secondary_prioritized_json
    output_path = args.output
    
    # Use command-line arguments for analysis and recommendation output paths if provided
    analysis_output_path = args.analysis_output if args.analysis_output else None
    recommendation_output_path = args.recommendation_output if args.recommendation_output else None
    
    # If analysis_output or recommendation_output are not provided, generate default paths
    if not analysis_output_path:
        output_path_obj = Path(output_path)
        analysis_output_path = str(output_path_obj.parent / f"{output_path_obj.stem}_analysis.json")
    
    if not recommendation_output_path:
        output_path_obj = Path(output_path)
        recommendation_output_path = str(output_path_obj.parent / f"{output_path_obj.stem}_recommendation.json")
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Primary JSON path: {primary_json_path}")
    logger.info(f"Secondary JSON path: {secondary_json_path}")
    logger.info(f"Secondary prioritized JSON path: {secondary_prioritized_json_path}")
    logger.info(f"Combined output path: {output_path}")
    logger.info(f"Analysis output path (LLM 1): {analysis_output_path}")
    logger.info(f"Recommendation output path (LLM 2): {recommendation_output_path}")
    
    # Check if input files exist
    if not Path(primary_json_path).exists():
        logger.error(f"Primary JSON file not found: {primary_json_path}")
        return
    
    if not Path(secondary_json_path).exists():
        logger.error(f"Secondary JSON file not found: {secondary_json_path}")
        return
    
    if not Path(secondary_prioritized_json_path).exists():
        logger.error(f"Secondary prioritized JSON file not found: {secondary_prioritized_json_path}")
        return
    
    # Create the report card data
    try:
        report_data = create_report_card_data(
            primary_json_path,
            secondary_json_path,
            secondary_prioritized_json_path,
            output_path,
            analysis_output_path,
            recommendation_output_path
        )
        logger.info("Report card data generation completed successfully")
    except Exception as e:
        logger.error(f"Error generating report card data: {e}", exc_info=True)

if __name__ == "__main__":
    main()
