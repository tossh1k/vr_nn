"""
Integrity Checker - Unified interface for content integrity validation

Provides a unified interface to check integrity across different modalities:
- Text: consistency and temporal logic
- Images: presence of required details
"""

import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .text_generator import TextZoneGenerator, ZoneDescription, IntegrityCheckResult
from .texture_generator import TextureGenerator, TextureGenerationResult, DetailCheckResult

logger = logging.getLogger(__name__)


@dataclass
class OverallIntegrityResult:
    """Overall integrity check result for all modalities."""
    is_valid: bool
    text_result: Optional[IntegrityCheckResult] = None
    texture_result: Optional[DetailCheckResult] = None
    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)


class IntegrityChecker:
    """
    Unified integrity checker for all content modalities.
    
    Coordinates integrity checks across:
    - Text zone descriptions (LLM)
    - 2D textures/maps (Stable Diffusion + ControlNet)
    """
    
    def __init__(
        self,
        text_generator: Optional[TextZoneGenerator] = None,
        texture_generator: Optional[TextureGenerator] = None,
    ):
        """
        Initialize the integrity checker.
        
        Args:
            text_generator: Optional TextZoneGenerator instance
            texture_generator: Optional TextureGenerator instance
        """
        self.text_generator = text_generator or TextZoneGenerator()
        self.texture_generator = texture_generator or TextureGenerator()
        
        logger.info("Initialized IntegrityChecker")
    
    def check_text_integrity(
        self,
        zone: ZoneDescription,
        check_consistency: bool = True,
        check_temporal: bool = True
    ) -> IntegrityCheckResult:
        """
        Check integrity of text zone description.
        
        Args:
            zone: ZoneDescription to check
            check_consistency: Whether to check consistency with previous zones
            check_temporal: Whether to check temporal logic
            
        Returns:
            IntegrityCheckResult with validation details
        """
        logger.info(f"Checking text integrity for zone: {zone.zone_id}")
        
        result = self.text_generator.check_integrity(
            zone=zone,
            check_consistency=check_consistency,
            check_temporal=check_temporal
        )
        
        if result.is_valid:
            logger.info(f"Zone {zone.zone_id} passed integrity check")
        else:
            logger.warning(f"Zone {zone.zone_id} failed integrity check: {result.issues}")
        
        return result
    
    def check_texture_integrity(
        self,
        texture: TextureGenerationResult,
        required_details: Optional[list] = None
    ) -> DetailCheckResult:
        """
        Check integrity of generated texture.
        
        Args:
            texture: TextureGenerationResult to check
            required_details: List of required details to verify
            
        Returns:
            DetailCheckResult with validation details
        """
        logger.info(f"Checking texture integrity for: {texture.texture_id}")
        
        result = self.texture_generator.check_details(
            result=texture,
            required_details=required_details
        )
        
        if result.is_valid:
            logger.info(f"Texture {texture.texture_id} passed detail check")
        else:
            logger.warning(
                f"Texture {texture.texture_id} missing details: {result.missing_details}"
            )
        
        return result
    
    def check_overall_integrity(
        self,
        content: Union[ZoneDescription, TextureGenerationResult, dict],
        content_type: str = "auto"
    ) -> OverallIntegrityResult:
        """
        Check overall integrity of content.
        
        Args:
            content: Content to check (ZoneDescription, TextureGenerationResult, or dict)
            content_type: Type of content ('text', 'texture', or 'auto')
            
        Returns:
            OverallIntegrityResult with comprehensive validation
        """
        issues = []
        suggestions = []
        scores = {}
        
        # Auto-detect content type if needed
        if content_type == "auto":
            if isinstance(content, ZoneDescription):
                content_type = "text"
            elif isinstance(content, TextureGenerationResult):
                content_type = "texture"
            elif isinstance(content, dict):
                if "description" in content:
                    content_type = "text"
                elif "image_path" in content or "image_data" in content:
                    content_type = "texture"
                else:
                    content_type = "mixed"
            else:
                content_type = "unknown"
        
        # Check based on content type
        if content_type == "text":
            if not isinstance(content, ZoneDescription):
                # Convert dict to ZoneDescription
                content = ZoneDescription(**content)
            
            text_result = self.check_text_integrity(content)
            scores["consistency"] = text_result.consistency_score
            scores["temporal_logic"] = text_result.temporal_logic_score
            
            issues.extend(text_result.issues)
            suggestions.extend(text_result.suggestions)
            
            return OverallIntegrityResult(
                is_valid=text_result.is_valid,
                text_result=text_result,
                issues=issues,
                suggestions=suggestions,
                scores=scores
            )
        
        elif content_type == "texture":
            if not isinstance(content, TextureGenerationResult):
                # Assume it's already a TextureGenerationResult
                pass
            
            texture_result = self.check_texture_integrity(content)
            scores["detail_presence"] = 1.0 if texture_result.is_valid else 0.5
            
            issues.extend([
                f"Missing detail: {detail}"
                for detail in texture_result.missing_details
            ])
            suggestions.extend(texture_result.suggestions)
            
            return OverallIntegrityResult(
                is_valid=texture_result.is_valid,
                texture_result=texture_result,
                issues=issues,
                suggestions=suggestions,
                scores=scores
            )
        
        elif content_type == "mixed":
            # Handle mixed content (both text and texture)
            text_result = None
            texture_result = None
            
            if "zone" in content:
                zone = content["zone"]
                if not isinstance(zone, ZoneDescription):
                    zone = ZoneDescription(**zone)
                text_result = self.check_text_integrity(zone)
                scores["consistency"] = text_result.consistency_score
                scores["temporal_logic"] = text_result.temporal_logic_score
                issues.extend(text_result.issues)
                suggestions.extend(text_result.suggestions)
            
            if "texture" in content:
                texture = content["texture"]
                texture_result = self.check_texture_integrity(texture)
                scores["detail_presence"] = 1.0 if texture_result.is_valid else 0.5
                issues.extend([
                    f"Missing detail: {detail}"
                    for detail in texture_result.missing_details
                ])
                suggestions.extend(texture_result.suggestions)
            
            # Overall validity requires all checks to pass
            is_valid = (
                (text_result is None or text_result.is_valid) and
                (texture_result is None or texture_result.is_valid)
            )
            
            return OverallIntegrityResult(
                is_valid=is_valid,
                text_result=text_result,
                texture_result=texture_result,
                issues=issues,
                suggestions=suggestions,
                scores=scores
            )
        
        else:
            logger.warning(f"Unknown content type: {content_type}")
            return OverallIntegrityResult(
                is_valid=False,
                issues=[f"Unknown content type: {content_type}"],
                suggestions=["Specify correct content type"]
            )
    
    def generate_with_validation(
        self,
        modality: str,
        **generation_kwargs
    ) -> tuple[Any, OverallIntegrityResult]:
        """
        Generate content with automatic integrity validation.
        
        Args:
            modality: Type of content to generate ('text' or 'texture')
            **generation_kwargs: Arguments for the generator
            
        Returns:
            Tuple of (generated_content, OverallIntegrityResult)
        """
        logger.info(f"Generating {modality} with validation")
        
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            logger.info(f"Generation attempt {attempts}/{max_attempts}")
            
            # Generate content
            if modality == "text":
                content = self.text_generator.generate_zone(**generation_kwargs)
                result = self.check_text_integrity(content)
                
                if result.is_valid:
                    logger.info("Text generation successful with valid integrity")
                    return content, OverallIntegrityResult(
                        is_valid=True,
                        text_result=result,
                        scores={
                            "consistency": result.consistency_score,
                            "temporal_logic": result.temporal_logic_score
                        }
                    )
                
                # If not valid and we have attempts left, try again
                if attempts < max_attempts:
                    logger.info(f"Retrying generation. Issues: {result.issues}")
            
            elif modality == "texture":
                content = self.texture_generator.generate_texture(**generation_kwargs)
                result = self.check_texture_integrity(
                    content,
                    required_details=generation_kwargs.get("required_details")
                )
                
                if result.is_valid:
                    logger.info("Texture generation successful with valid details")
                    return content, OverallIntegrityResult(
                        is_valid=True,
                        texture_result=result,
                        scores={"detail_presence": 1.0}
                    )
                
                # Try regeneration with feedback
                if attempts < max_attempts and result.missing_details:
                    logger.info(f"Regenerating with feedback. Missing: {result.missing_details}")
                    content = self.texture_generator.regenerate_with_feedback(
                        content, result
                    )
                    result = self.check_texture_integrity(
                        content,
                        required_details=generation_kwargs.get("required_details")
                    )
                    
                    if result.is_valid:
                        return content, OverallIntegrityResult(
                            is_valid=True,
                            texture_result=result,
                            scores={"detail_presence": 1.0}
                        )
            else:
                raise ValueError(f"Unknown modality: {modality}")
        
        # Return last attempt even if not valid
        logger.warning(f"Generated content after {max_attempts} attempts (may have issues)")
        
        if modality == "text":
            return content, OverallIntegrityResult(
                is_valid=False,
                text_result=result,
                issues=result.issues,
                suggestions=result.suggestions,
                scores={
                    "consistency": result.consistency_score,
                    "temporal_logic": result.temporal_logic_score
                }
            )
        else:
            return content, OverallIntegrityResult(
                is_valid=False,
                texture_result=result,
                issues=[f"Missing: {d}" for d in result.missing_details],
                suggestions=result.suggestions,
                scores={"detail_presence": 0.5}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about integrity checks.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "text_zones_generated": len(self.text_generator.previous_zones),
            "text_generator_model": self.text_generator.model_name,
            "texture_generator_model": self.texture_generator.sd_model_name,
            "controlnet_model": self.texture_generator.controlnet_model,
        }
        
        return stats
