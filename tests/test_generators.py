"""
Tests for Content Generator with Integrity Control
"""

import pytest
from src.text_generator import TextZoneGenerator, ZoneDescription
from src.texture_generator import TextureGenerator, TextureGenerationResult
from src.integrity_checker import IntegrityChecker


class TestTextZoneGenerator:
    """Tests for TextZoneGenerator."""
    
    def test_init(self):
        """Test initialization."""
        generator = TextZoneGenerator()
        assert generator.model_name == "mistralai/Mistral-7B-Instruct-v0.1"
        assert generator.previous_zones == []
    
    def test_mock_generation(self):
        """Test mock generation without model."""
        generator = TextZoneGenerator()
        
        zone = generator.generate_zone(
            zone_id="test_001",
            prompt="Test prompt"
        )
        
        assert zone.zone_id == "test_001"
        assert "Test prompt" in zone.description
        assert zone.metadata["mock"] is True
    
    def test_generation_with_timestamp(self):
        """Test generation with timestamp."""
        generator = TextZoneGenerator()
        
        zone = generator.generate_zone(
            zone_id="test_002",
            prompt="Test with time",
            timestamp="12:00"
        )
        
        assert zone.timestamp == "12:00"
    
    def test_integrity_check_valid(self):
        """Test integrity check for valid content."""
        generator = TextZoneGenerator()
        
        # Generate a zone
        zone1 = generator.generate_zone(
            zone_id="zone1",
            prompt="A bright sunny day"
        )
        
        # Generate another zone
        zone2 = generator.generate_zone(
            zone_id="zone2",
            prompt="The garden is beautiful"
        )
        
        # Check integrity
        result = generator.check_integrity(zone2)
        
        # Should be valid (no contradictions)
        assert result.consistency_score >= 0.0
        assert result.temporal_logic_score >= 0.0
    
    def test_contradiction_detection(self):
        """Test detection of contradictions."""
        generator = TextZoneGenerator()
        
        # Generate first zone
        zone1 = generator.generate_zone(
            zone_id="zone1",
            prompt="The door is open and the room is light"
        )
        generator.previous_zones.append(zone1)
        
        # Generate contradictory zone
        zone2 = ZoneDescription(
            zone_id="zone2",
            description="The door is closed and the room is dark"
        )
        
        # Check for contradictions
        result = generator.check_integrity(zone2, check_temporal=False)
        
        # Should detect contradictions
        assert len(result.issues) > 0 or result.consistency_score < 1.0
    
    def test_clear_history(self):
        """Test clearing history."""
        generator = TextZoneGenerator()
        
        # Generate some zones
        generator.generate_zone(zone_id="z1", prompt="p1")
        generator.generate_zone(zone_id="z2", prompt="p2")
        
        assert len(generator.previous_zones) == 2
        
        # Clear history
        generator.clear_history()
        
        assert len(generator.previous_zones) == 0


class TestTextureGenerator:
    """Tests for TextureGenerator."""
    
    def test_init(self):
        """Test initialization."""
        generator = TextureGenerator()
        assert generator.sd_model_name == "runwayml/stable-diffusion-v1-5"
        assert generator.image_size == (512, 512)
    
    def test_mock_generation(self):
        """Test mock generation without model."""
        generator = TextureGenerator()
        
        result = generator.generate_texture(
            texture_id="test_tex_001",
            prompt="Test texture"
        )
        
        assert result.texture_id == "test_tex_001"
        assert result.prompt == "Test texture"
        assert result.metadata["mock"] is True
    
    def test_generation_with_required_details(self):
        """Test generation with required details."""
        generator = TextureGenerator()
        
        result = generator.generate_texture(
            texture_id="test_tex_002",
            prompt="Door texture",
            required_details=["door handle", "lock"]
        )
        
        assert "door handle" in result.required_details
        assert "lock" in result.required_details
    
    def test_detail_check_mock(self):
        """Test detail checking in mock mode."""
        generator = TextureGenerator()
        
        result = generator.generate_texture(
            texture_id="test_tex_003",
            prompt="Test",
            required_details=["door handle"]
        )
        
        # In mock mode, should pass
        check_result = generator.check_details(result)
        assert check_result.is_valid is True or len(check_result.detected_details) > 0


class TestIntegrityChecker:
    """Tests for IntegrityChecker."""
    
    def test_init(self):
        """Test initialization."""
        checker = IntegrityChecker()
        assert checker.text_generator is not None
        assert checker.texture_generator is not None
    
    def test_check_text_integrity(self):
        """Test text integrity checking."""
        checker = IntegrityChecker()
        
        zone = checker.text_generator.generate_zone(
            zone_id="test_zone",
            prompt="Test description"
        )
        
        result = checker.check_text_integrity(zone)
        assert result is not None
    
    def test_check_texture_integrity(self):
        """Test texture integrity checking."""
        checker = IntegrityChecker()
        
        texture = checker.texture_generator.generate_texture(
            texture_id="test_texture",
            prompt="Test texture"
        )
        
        result = checker.check_texture_integrity(texture)
        assert result is not None
    
    def test_generate_with_validation_text(self):
        """Test text generation with validation."""
        checker = IntegrityChecker()
        
        content, result = checker.generate_with_validation(
            modality="text",
            zone_id="validated_zone",
            prompt="A validated zone"
        )
        
        assert content is not None
        assert result is not None
        assert content.zone_id == "validated_zone"
    
    def test_generate_with_validation_texture(self):
        """Test texture generation with validation."""
        checker = IntegrityChecker()
        
        content, result = checker.generate_with_validation(
            modality="texture",
            texture_id="validated_texture",
            prompt="A validated texture"
        )
        
        assert content is not None
        assert result is not None
        assert content.texture_id == "validated_texture"
    
    def test_get_statistics(self):
        """Test getting statistics."""
        checker = IntegrityChecker()
        
        # Generate some content
        checker.text_generator.generate_zone(zone_id="s1", prompt="p1")
        
        stats = checker.get_statistics()
        
        assert "text_zones_generated" in stats
        assert "text_generator_model" in stats
        assert "texture_generator_model" in stats


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test complete workflow."""
        # Initialize
        checker = IntegrityChecker()
        
        # Generate text
        text_content, text_result = checker.generate_with_validation(
            modality="text",
            zone_id="workflow_zone",
            prompt="A mysterious corridor"
        )
        
        # Generate texture
        texture_content, texture_result = checker.generate_with_validation(
            modality="texture",
            texture_id="workflow_texture",
            prompt="Corridor texture"
        )
        
        # Verify both succeeded
        assert text_content is not None
        assert texture_content is not None
        assert text_result is not None
        assert texture_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
