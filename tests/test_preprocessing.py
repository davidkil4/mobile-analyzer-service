# tests/test_preprocessing.py

# TODO: Add comprehensive unit and integration tests for the preprocessing functions
#       (e.g., run_segmentation, run_alignment, run_clause_identification, preprocess_batch)
#       using mocks for the LangChain chains to cover more scenarios and edge cases.

import pytest
from analyzer_service.preprocessing import parse_segmentation_output

# Test case 1: Standard JSON list output
def test_parse_segmentation_json_list():
    raw_output = '```json\n["This is segment one.", "And segment two."]\n```'
    expected = ["This is segment one.", "And segment two."]
    assert parse_segmentation_output(raw_output) == expected

# Test case 2: JSON list of objects with 'as_unit' key
def test_parse_segmentation_json_object_as_unit():
    raw_output = '[{"as_unit": "First unit."}, {"as_unit": "Second unit."}]'
    expected = ["First unit.", "Second unit."]
    assert parse_segmentation_output(raw_output) == expected

# Test case 3: JSON list of objects with 'segment_text' key
def test_parse_segmentation_json_object_segment_text():
    raw_output = '[{"segment_text": "Unit A text."}, {"segment_text": "Unit B text."}]'
    expected = ["Unit A text.", "Unit B text."]
    assert parse_segmentation_output(raw_output) == expected

# Test case 4: Simple markdown list
def test_parse_segmentation_markdown_list():
    raw_output = """
- Segment using markdown.
- Another markdown segment.
"""
    expected = ["Segment using markdown.", "Another markdown segment."]
    # Assuming the parser strips leading/trailing whitespace and hyphens
    assert parse_segmentation_output(raw_output) == expected

# Test case 5: Plain text lines (fallback)
def test_parse_segmentation_plain_lines():
    raw_output = "Just line one.\nJust line two."
    expected = ["Just line one.", "Just line two."]
    assert parse_segmentation_output(raw_output) == expected

# Test case 6: Empty input
def test_parse_segmentation_empty_input():
    raw_output = ""
    expected = []
    assert parse_segmentation_output(raw_output) == expected

# Test case 7: Input that is just whitespace or formatting
def test_parse_segmentation_whitespace_input():
    raw_output = "```json\n[]\n```"
    expected = []
    assert parse_segmentation_output(raw_output) == expected
    raw_output_2 = "\n - \n - \n"
    assert parse_segmentation_output(raw_output_2) == expected
