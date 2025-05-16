import json
import sys
import os
import statistics
from copy import deepcopy
from langdetect import detect as detect_language, LangDetectException

PRIORITY_BONUS = {"high": 0.15, "medium": 0.07, "low": 0.0, None: 0.0}


def zone_statistical_filtering(zones, filter_stats):
    filtered_zones = []
    for zone in zones:
        zone_name = zone.get('zone_name')
        recs = zone.get('recommendations', [])
        # Count utterances filtered by FILTER (metadata)
        num_filtered_metadata = sum(1 for utt in recs if utt.get('filtering_metadata', {}).get('decision', '').upper() == 'FILTER')
        filter_stats['by_metadata'] += num_filtered_metadata
        # Only keep utterances with decision == KEEP
        kept_recs = [utt for utt in recs if utt.get('filtering_metadata', {}).get('decision', '').upper() == 'KEEP']
        N = len(kept_recs)
        orig_N = len(recs)
        # Apply statistical filtering based on priority_score only
        if N > 0:
            scores = [utt.get('priority_score', 0) for utt in kept_recs]
            mu = statistics.mean(scores)
            sigma = statistics.pstdev(scores) if N > 1 else 0.0
            # Set threshold based on zone
            if zone_name == 'ZPD-1':
                threshold = mu - 0.5 * sigma
            elif zone_name == 'ZPD':
                threshold = mu
            elif zone_name == 'ZPD+1':
                threshold = mu + 0.5 * sigma
            elif zone_name == 'ZPD/2':
                threshold = mu - 0.5 * sigma
            else:
                threshold = mu - sigma
            filtered = [utt for utt in kept_recs if utt.get('priority_score', 0) >= threshold]
            num_filtered_zone = N - len(filtered)
            filter_stats['by_zone'][zone_name] = filter_stats['by_zone'].get(zone_name, 0) + num_filtered_zone
            M = min(2, N)
            if len(filtered) < M:
                # refill top scoring utterances
                filtered = sorted(kept_recs, key=lambda utt: utt.get('priority_score', 0), reverse=True)[:M]
            filtered_zones.append({'zone_name': zone_name, 'recommendations': filtered})
        else:
            filtered_zones.append({'zone_name': zone_name, 'recommendations': []})
    return filtered_zones


def filter_and_split(input_path, output_path_1, output_path_2):
    # Always clear the Korean teaching input file at the start to prevent stale data
    output_directory = os.path.dirname(output_path_1)
    korean_output_path = os.path.join(output_directory, 'korean_teaching_input.json')
    os.makedirs(output_directory, exist_ok=True)
    # Overwrite with empty structure
    with open(korean_output_path, 'w', encoding='utf-8') as f_ko:
        json.dump({"metadata": {}, "analysis_zones": []}, f_ko, indent=2, ensure_ascii=False)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    all_input_zones = data.get('analysis_zones', [])

    # Stats for filtering
    filter_stats = {'by_metadata': 0, 'by_zone': {}}

    korean_utterances_for_processing = []  # List of (zone_name, rec)
    english_other_zones_for_filtering = []

    print("\n-- Processing utterances for language segregation and metadata filtering...")
    for zone_data in all_input_zones:
        current_zone_kept_english_recs = []
        num_filtered_metadata_in_zone = 0
        zone_name = zone_data.get('zone_name')
        for rec in zone_data.get('recommendations', []):
            # Apply metadata filter first
            if rec.get('filtering_metadata', {}).get('decision', '').upper() == 'KEEP':
                try:
                    original_text = rec.get('original', '')
                    # Handle empty or whitespace-only strings for langdetect
                    if not original_text.strip():
                        lang = 'en' # Default to English path if text is effectively empty
                    else:
                        lang = detect_language(original_text)
                    
                    if lang == 'ko':
                        # Make a deep copy 
                        korean_rec = deepcopy(rec)
                        korean_utterances_for_processing.append((zone_name, korean_rec))
                    else:
                        current_zone_kept_english_recs.append(rec)
                except LangDetectException:
                    # If langdetect fails (e.g., too short, mixed), default to English path
                    current_zone_kept_english_recs.append(rec)
            else: # Utterance was filtered by metadata ('FILTER' decision)
                num_filtered_metadata_in_zone += 1
        
        filter_stats['by_metadata'] += num_filtered_metadata_in_zone
        english_other_zones_for_filtering.append({
            'zone_name': zone_name, 
            'recommendations': current_zone_kept_english_recs
        })

    # Write Korean utterances to their own file with the same nested structure and filtering as English utterances
    if korean_utterances_for_processing:
        output_directory = os.path.dirname(output_path_1)
        os.makedirs(output_directory, exist_ok=True) # Ensure directory exists
        korean_output_path = os.path.join(output_directory, 'korean_teaching_input.json')

        # Group Korean utterances by zone
        korean_zones = {}
        for zone_name, utt in korean_utterances_for_processing:
            if zone_name not in korean_zones:
                korean_zones[zone_name] = []
            korean_zones[zone_name].append(utt)

        # Apply statistical filtering to each zone (parallel to English logic)
        filtered_korean_zones = []
        for zone_name, recs in korean_zones.items():
            # Only keep utterances that pass the statistical threshold
            N = len(recs)
            if N > 0:
                scores = [utt.get('priority_score', 0) for utt in recs]
                mu = statistics.mean(scores)
                sigma = statistics.pstdev(scores) if N > 1 else 0.0
                # Set threshold based on zone
                if zone_name == 'ZPD-1':
                    threshold = mu - 0.5 * sigma
                elif zone_name == 'ZPD':
                    threshold = mu
                elif zone_name == 'ZPD+1':
                    threshold = mu + 0.5 * sigma
                elif zone_name == 'ZPD/2':
                    threshold = mu - 0.5 * sigma
                else:
                    threshold = mu - sigma
                filtered = [utt for utt in recs if utt.get('priority_score', 0) >= threshold]
                M = min(2, N)
                if len(filtered) < M:
                    # refill top scoring utterances
                    filtered = sorted(recs, key=lambda utt: utt.get('priority_score', 0), reverse=True)[:M]
                filtered_korean_zones.append({'zone_name': zone_name, 'recommendations': filtered})
            else:
                filtered_korean_zones.append({'zone_name': zone_name, 'recommendations': []})

        korean_structured_output = {
            "metadata": {},
            "analysis_zones": filtered_korean_zones
        }

        with open(korean_output_path, 'w', encoding='utf-8') as f_ko:
            json.dump(korean_structured_output, f_ko, indent=2, ensure_ascii=False)
        print(f"-- Korean utterances segregated, filtered, and saved to: {korean_output_path} ({len(korean_utterances_for_processing)} utterances)")
    else:
        print("-- No Korean utterances found or kept after metadata filtering.")
        # Explicitly overwrite the Korean file with empty structure if no utterances found
        with open(korean_output_path, 'w', encoding='utf-8') as f_ko:
            json.dump({"metadata": {}, "analysis_zones": []}, f_ko, indent=2, ensure_ascii=False)

    # Step 1 (for English/Other): Apply zone-specific statistical filtering
    filtered_zones = zone_statistical_filtering(english_other_zones_for_filtering, filter_stats)

    # Step 2: Gather all kept English/Other utterances (across all zones) for splitting
    all_kept_utterances = []
    for zone in filtered_zones:
        zone_name = zone['zone_name']
        for utt in zone['recommendations']:
            all_kept_utterances.append((zone_name, utt))

    # Step 3: Split all kept utterances into two groups (A and B)
    group_A = set()
    group_B = set()
    for i, (zone_name, utt) in enumerate(all_kept_utterances):
        if i % 2 == 0:
            group_A.add(id(utt))
        else:
            group_B.add(id(utt))

    # Step 4: For each group, reconstruct the full structure with only the utterances in that group
    def build_output(group_ids):
        output = {'metadata': deepcopy(metadata), 'analysis_zones': []}
        for zone in filtered_zones:
            zone_name = zone['zone_name']
            recs = zone['recommendations']
            recs_in_group = [utt for utt in recs if id(utt) in group_ids]
            output['analysis_zones'].append({'zone_name': zone_name, 'recommendations': recs_in_group})
        return output

    output_A = build_output(group_A)
    output_B = build_output(group_B)

    with open(output_path_1, 'w', encoding='utf-8') as f:
        json.dump(output_A, f, indent=2, ensure_ascii=False)
    with open(output_path_2, 'w', encoding='utf-8') as f:
        json.dump(output_B, f, indent=2, ensure_ascii=False)

    # Print filtering summary
    print("\nFiltering Summary:")
    print(f"Filtered by FILTER (metadata): {filter_stats['by_metadata']}")
    print("Filtered by zone threshold:")
    for zone_name, count in filter_stats['by_zone'].items():
        print(f"  {zone_name}: {count}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python filter_and_split.py <input_file> [output_file_1] [output_file_2]")
        sys.exit(1)
    input_path = sys.argv[1]
    # Default output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output_files')
    os.makedirs(output_dir, exist_ok=True)
    output_path_1 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(output_dir, 'teaching_input_part1.json')
    output_path_2 = sys.argv[3] if len(sys.argv) > 3 else os.path.join(output_dir, 'teaching_input_part2.json')
    filter_and_split(input_path, output_path_1, output_path_2)
    print(f"Done. Outputs: {output_path_1}, {output_path_2}")


if __name__ == '__main__':
    main()
