#!/usr/bin/env python3
"""
Example usage of the Content Generator with Integrity Control

This script demonstrates how to use the system for:
1. Generating text zone descriptions with LLM
2. Generating 2D textures with Stable Diffusion + ControlNet
3. Validating integrity at each step
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.text_generator import TextZoneGenerator, ZoneDescription
from src.texture_generator import TextureGenerator
from src.integrity_checker import IntegrityChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_text_generation():
    """Example: Generate text zone descriptions with integrity control."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Text Zone Generation")
    print("="*60)
    
    # Initialize generator (mock mode - no model loading)
    generator = TextZoneGenerator(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        device="cpu"  # Use CPU for demo
    )
    
    # Generate first zone
    print("\nGenerating first zone...")
    zone1 = generator.generate_zone(
        zone_id="zone_001",
        prompt="A dimly lit corridor with wooden walls and a single window",
        timestamp="10:00"
    )
    print(f"Zone {zone1.zone_id}: {zone1.description}")
    
    # Generate second zone
    print("\nGenerating second zone...")
    zone2 = generator.generate_zone(
        zone_id="zone_002",
        prompt="An open door leading to a bright garden",
        timestamp="10:05"
    )
    print(f"Zone {zone2.zone_id}: {zone2.description}")
    
    # Check integrity
    print("\nChecking integrity of zone 2...")
    result = generator.check_integrity(zone2)
    print(f"Valid: {result.is_valid}")
    print(f"Consistency score: {result.consistency_score:.2f}")
    print(f"Temporal logic score: {result.temporal_logic_score:.2f}")
    
    if result.issues:
        print("Issues found:")
        for issue in result.issues:
            print(f"  - {issue}")
    
    if result.suggestions:
        print("Suggestions:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")
    
    return zone1, zone2, result


def example_texture_generation():
    """Example: Generate textures with detail checking."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Texture Generation")
    print("="*60)
    
    # Initialize generator (mock mode - no model loading)
    generator = TextureGenerator(
        sd_model_name="runwayml/stable-diffusion-v1-5",
        device="cpu"  # Use CPU for demo
    )
    
    # Generate texture
    print("\nGenerating texture...")
    texture = generator.generate_texture(
        texture_id="door_texture_001",
        prompt="A wooden door in an old house, realistic, detailed",
        required_details=["door handle", "lock"]
    )
    print(f"Texture ID: {texture.texture_id}")
    print(f"Prompt: {texture.prompt}")
    print(f"Required details: {texture.required_details}")
    
    # Check for required details
    print("\nChecking for required details...")
    detail_result = generator.check_details(texture)
    print(f"Valid: {detail_result.is_valid}")
    print(f"Detected details: {detail_result.detected_details}")
    print(f"Missing details: {detail_result.missing_details}")
    
    if detail_result.confidence_scores:
        print("Confidence scores:")
        for detail, score in detail_result.confidence_scores.items():
            print(f"  - {detail}: {score:.2f}")
    
    if detail_result.suggestions:
        print("Suggestions:")
        for suggestion in detail_result.suggestions:
            print(f"  - {suggestion}")
    
    return texture, detail_result


def example_integrated_workflow():
    """Example: Integrated workflow with integrity checker."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Integrated Workflow")
    print("="*60)
    
    # Initialize integrity checker
    checker = IntegrityChecker()
    
    # Generate text with validation
    print("\nGenerating text with automatic validation...")
    text_content, text_result = checker.generate_with_validation(
        modality="text",
        zone_id="zone_003",
        prompt="A mysterious room with ancient artifacts",
        timestamp="10:10"
    )
    print(f"Text generated: {text_content.zone_id}")
    print(f"Validation passed: {text_result.is_valid}")
    print(f"Scores: {text_result.scores}")
    
    # Generate texture with validation
    print("\nGenerating texture with automatic validation...")
    texture_content, texture_result = checker.generate_with_validation(
        modality="texture",
        texture_id="room_texture_001",
        prompt="Ancient room with stone walls and torches",
        required_details=["torch", "stone texture"]
    )
    print(f"Texture generated: {texture_content.texture_id}")
    print(f"Validation passed: {texture_result.is_valid}")
    print(f"Scores: {texture_result.scores}")
    
    # Get statistics
    print("\nGeneration statistics:")
    stats = checker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return text_content, texture_content


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CONTENT GENERATOR WITH INTEGRITY CONTROL")
    print("Demonstration Script")
    print("="*60)
    
    try:
        # Run examples
        example_text_generation()
        example_texture_generation()
        example_integrated_workflow()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
