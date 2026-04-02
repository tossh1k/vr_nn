"""
Text Zone Generator using LLM (LLaMA/Mistral)

Generates zone descriptions with integrity control for:
- Consistency with previous zones
- Temporal logic
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ZoneDescription:
    """Represents a generated zone description."""
    zone_id: str
    description: str
    timestamp: Optional[str] = None
    previous_zones: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrityCheckResult:
    """Result of integrity check for text generation."""
    is_valid: bool
    consistency_score: float
    temporal_logic_score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class TextZoneGenerator:
    """
    Generator for zone descriptions using LLM models (LLaMA/Mistral).
    
    Ensures:
    - Consistency with previously generated zones
    - Temporal logic in descriptions
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        device: str = "cuda",
        max_length: int = 512,
        temperature: float = 0.7,
    ):
        """
        Initialize the text zone generator.
        
        Args:
            model_name: Name or path of the LLM model
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum length of generated text
            temperature: Sampling temperature for generation
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        self.model = None
        self.tokenizer = None
        self.previous_zones: List[ZoneDescription] = []
        
        logger.info(f"Initialized TextZoneGenerator with model: {model_name}")
    
    def load_model(self):
        """Load the LLM model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device if self.device == "cuda" else None,
            )
            
            logger.info("Model loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_zone(
        self,
        zone_id: str,
        prompt: str,
        previous_zones: Optional[List[ZoneDescription]] = None,
        timestamp: Optional[str] = None,
        **kwargs
    ) -> ZoneDescription:
        """
        Generate a zone description.
        
        Args:
            zone_id: Unique identifier for the zone
            prompt: Prompt for generating the description
            previous_zones: List of previous zone descriptions for context
            timestamp: Optional timestamp for temporal logic
            **kwargs: Additional generation parameters
            
        Returns:
            ZoneDescription object with generated content
        """
        if self.model is None:
            logger.warning("Model not loaded, using mock generation")
            return self._mock_generate(zone_id, prompt, previous_zones, timestamp)
        
        # Build context from previous zones
        context = self._build_context(previous_zones or self.previous_zones)
        
        # Create full prompt with context
        full_prompt = self._create_prompt(context, prompt, timestamp)
        
        # Generate text
        description = self._generate_text(full_prompt, **kwargs)
        
        # Create zone description
        zone_desc = ZoneDescription(
            zone_id=zone_id,
            description=description,
            timestamp=timestamp,
            previous_zones=[z.zone_id for z in (previous_zones or self.previous_zones)],
            metadata={"model": self.model_name, "prompt": prompt}
        )
        
        # Store in history
        self.previous_zones.append(zone_desc)
        
        logger.info(f"Generated zone: {zone_id}")
        return zone_desc
    
    def _build_context(self, previous_zones: List[ZoneDescription]) -> str:
        """Build context string from previous zones."""
        if not previous_zones:
            return ""
        
        context_parts = []
        for zone in previous_zones:
            context_parts.append(f"Zone {zone.zone_id}: {zone.description}")
            if zone.timestamp:
                context_parts[-1] += f" [Time: {zone.timestamp}]"
        
        return "\n".join(context_parts)
    
    def _create_prompt(
        self,
        context: str,
        prompt: str,
        timestamp: Optional[str] = None
    ) -> str:
        """Create the full prompt for generation."""
        system_msg = (
            "You are a content generator for zone descriptions. "
            "Ensure consistency with previous zones and maintain temporal logic."
        )
        
        if context:
            user_msg = f"Previous zones:\n{context}\n\nGenerate description for new zone: {prompt}"
        else:
            user_msg = f"Generate description for zone: {prompt}"
        
        if timestamp:
            user_msg += f"\nTimestamp: {timestamp}"
        
        return f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"
    
    def _generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the loaded model."""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "do_sample": True,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def _mock_generate(
        self,
        zone_id: str,
        prompt: str,
        previous_zones: Optional[List[ZoneDescription]],
        timestamp: Optional[str]
    ) -> ZoneDescription:
        """Mock generation for testing without model."""
        description = f"[Mock] Generated description for zone {zone_id}: {prompt}"
        
        zone_desc = ZoneDescription(
            zone_id=zone_id,
            description=description,
            timestamp=timestamp,
            previous_zones=[z.zone_id for z in (previous_zones or [])],
            metadata={"mock": True, "prompt": prompt}
        )
        
        return zone_desc
    
    def check_integrity(
        self,
        zone: ZoneDescription,
        check_consistency: bool = True,
        check_temporal: bool = True
    ) -> IntegrityCheckResult:
        """
        Check integrity of generated zone description.
        
        Args:
            zone: ZoneDescription to check
            check_consistency: Whether to check consistency with previous zones
            check_temporal: Whether to check temporal logic
            
        Returns:
            IntegrityCheckResult with validation results
        """
        issues = []
        suggestions = []
        consistency_score = 1.0
        temporal_logic_score = 1.0
        
        # Check consistency with previous zones
        if check_consistency and self.previous_zones:
            consistency_score, consistency_issues, consistency_suggestions = \
                self._check_consistency(zone)
            issues.extend(consistency_issues)
            suggestions.extend(consistency_suggestions)
        
        # Check temporal logic
        if check_temporal and zone.timestamp:
            temporal_logic_score, temporal_issues, temporal_suggestions = \
                self._check_temporal_logic(zone)
            issues.extend(temporal_issues)
            suggestions.extend(temporal_suggestions)
        
        is_valid = consistency_score > 0.7 and temporal_logic_score > 0.7
        
        return IntegrityCheckResult(
            is_valid=is_valid,
            consistency_score=consistency_score,
            temporal_logic_score=temporal_logic_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _check_consistency(
        self,
        zone: ZoneDescription
    ) -> tuple[float, List[str], List[str]]:
        """Check consistency with previous zones."""
        issues = []
        suggestions = []
        score = 1.0
        
        # Simple keyword-based consistency check
        # In production, this would use semantic similarity or NLI models
        for prev_zone in self.previous_zones:
            # Check for contradictory keywords
            contradictions = self._find_contradictions(
                zone.description,
                prev_zone.description
            )
            
            if contradictions:
                score -= 0.1 * len(contradictions)
                for contradiction in contradictions:
                    issues.append(
                        f"Potential contradiction with zone {prev_zone.zone_id}: {contradiction}"
                    )
                suggestions.append(
                    f"Review description for consistency with zone {prev_zone.zone_id}"
                )
        
        score = max(0.0, score)
        return score, issues, suggestions
    
    def _find_contradictions(
        self,
        text1: str,
        text2: str
    ) -> List[str]:
        """Find potential contradictions between two texts."""
        contradictions = []
        
        # Simple contradiction patterns
        contradiction_pairs = [
            ("open", "closed"),
            ("light", "dark"),
            ("day", "night"),
            ("inside", "outside"),
            ("north", "south"),
            ("east", "west"),
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for word1, word2 in contradiction_pairs:
            if word1 in text1_lower and word2 in text2_lower:
                contradictions.append(f"{word1} vs {word2}")
            elif word2 in text1_lower and word1 in text2_lower:
                contradictions.append(f"{word2} vs {word1}")
        
        return contradictions
    
    def _check_temporal_logic(
        self,
        zone: ZoneDescription
    ) -> tuple[float, List[str], List[str]]:
        """Check temporal logic of the zone description."""
        issues = []
        suggestions = []
        score = 1.0
        
        if not zone.timestamp:
            return score, issues, suggestions
        
        # Check temporal consistency with previous zones
        for prev_zone in self.previous_zones:
            if prev_zone.timestamp:
                # Simple temporal check
                # In production, this would parse and compare timestamps properly
                if zone.timestamp < prev_zone.timestamp:
                    score -= 0.2
                    issues.append(
                        f"Temporal inconsistency: zone {zone.zone_id} ({zone.timestamp}) "
                        f"comes before zone {prev_zone.zone_id} ({prev_zone.timestamp})"
                    )
                    suggestions.append(
                        "Verify the chronological order of events"
                    )
        
        score = max(0.0, score)
        return score, issues, suggestions
    
    def clear_history(self):
        """Clear the history of previous zones."""
        self.previous_zones = []
        logger.info("Cleared zone history")
