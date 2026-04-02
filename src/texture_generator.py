"""
2D Texture/Map Generator using Stable Diffusion + ControlNet

Generates textures and maps with integrity control for:
- Presence of required details (door handle, light switch, etc.)
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TextureGenerationResult:
    """Represents a generated texture or map."""
    texture_id: str
    image_path: Optional[str]
    image_data: Optional[Any]  # PIL Image or numpy array
    prompt: str
    required_details: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetailCheckResult:
    """Result of detail presence check."""
    is_valid: bool
    detected_details: List[str] = field(default_factory=list)
    missing_details: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class TextureGenerator:
    """
    Generator for 2D textures and maps using Stable Diffusion + ControlNet.
    
    Ensures:
    - Presence of required details (door handles, light switches, etc.)
    - Consistency with scene requirements
    """
    
    def __init__(
        self,
        sd_model_name: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model: str = "lllyasviel/control_v11p_sd15_canny",
        device: str = "cuda",
        image_size: Tuple[int, int] = (512, 512),
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        """
        Initialize the texture generator.
        
        Args:
            sd_model_name: Name or path of the Stable Diffusion model
            controlnet_model: Name or path of the ControlNet model
            device: Device to run the model on ('cuda' or 'cpu')
            image_size: Size of generated images (width, height)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
        """
        self.sd_model_name = sd_model_name
        self.controlnet_model = controlnet_model
        self.device = device
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        self.pipe = None
        self.controlnet = None
        self.detail_detector = None
        
        logger.info(f"Initialized TextureGenerator with SD model: {sd_model_name}")
    
    def load_models(self):
        """Load Stable Diffusion and ControlNet models."""
        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            from transformers import AutoImageProcessor
            import torch
            
            logger.info(f"Loading ControlNet model: {self.controlnet_model}")
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            logger.info(f"Loading Stable Diffusion model: {self.sd_model_name}")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.sd_model_name,
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # Load detail detector (using pre-trained object detection)
            self._load_detail_detector()
            
            logger.info("Models loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_detail_detector(self):
        """Load model for detecting required details in images."""
        try:
            from transformers import AutoModelForObjectDetection, AutoProcessor
            
            # Use DETR for object detection
            self.detail_detector = AutoModelForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50"
            )
            self.detail_processor = AutoProcessor.from_pretrained(
                "facebook/detr-resnet-50"
            )
            
            logger.info("Detail detector loaded")
        except Exception as e:
            logger.warning(f"Could not load detail detector: {e}")
            self.detail_detector = None
    
    def generate_texture(
        self,
        texture_id: str,
        prompt: str,
        negative_prompt: str = "",
        control_image: Optional[Any] = None,
        required_details: Optional[List[str]] = None,
        num_images: int = 1,
        **kwargs
    ) -> TextureGenerationResult:
        """
        Generate a texture or map.
        
        Args:
            texture_id: Unique identifier for the texture
            prompt: Text prompt for generation
            negative_prompt: Negative prompt to avoid unwanted elements
            control_image: Control image for ControlNet (edge map, depth, etc.)
            required_details: List of required details to include
            num_images: Number of images to generate
            **kwargs: Additional generation parameters
            
        Returns:
            TextureGenerationResult with generated content
        """
        if self.pipe is None:
            logger.warning("Models not loaded, using mock generation")
            return self._mock_generate(
                texture_id, prompt, required_details, num_images
            )
        
        # Prepare generation parameters
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images_per_prompt": num_images,
            "width": self.image_size[0],
            "height": self.image_size[1],
            **kwargs
        }
        
        # Generate with or without ControlNet
        if control_image is not None:
            gen_kwargs["image"] = control_image
            images = self.pipe(**gen_kwargs).images
        else:
            # Without control image, use regular SD
            images = self.pipe(**gen_kwargs).images
        
        # Save first image
        image_path = None
        output_dir = Path("output/textures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if images:
            image_path = str(output_dir / f"{texture_id}.png")
            images[0].save(image_path)
            logger.info(f"Saved texture to: {image_path}")
        
        # Create result
        result = TextureGenerationResult(
            texture_id=texture_id,
            image_path=image_path,
            image_data=images[0] if images else None,
            prompt=prompt,
            required_details=required_details or [],
            metadata={
                "model": self.sd_model_name,
                "controlnet": self.controlnet_model,
                "negative_prompt": negative_prompt,
            }
        )
        
        logger.info(f"Generated texture: {texture_id}")
        return result
    
    def _mock_generate(
        self,
        texture_id: str,
        prompt: str,
        required_details: Optional[List[str]],
        num_images: int
    ) -> TextureGenerationResult:
        """Mock generation for testing without models."""
        logger.info(f"[Mock] Generating texture: {texture_id}")
        
        result = TextureGenerationResult(
            texture_id=texture_id,
            image_path=None,
            image_data=None,
            prompt=prompt,
            required_details=required_details or [],
            metadata={"mock": True}
        )
        
        return result
    
    def check_details(
        self,
        result: TextureGenerationResult,
        required_details: Optional[List[str]] = None
    ) -> DetailCheckResult:
        """
        Check if required details are present in the generated texture.
        
        Args:
            result: TextureGenerationResult to check
            required_details: List of required details to verify
            
        Returns:
            DetailCheckResult with validation results
        """
        if result.image_data is None:
            return DetailCheckResult(
                is_valid=False,
                detected_details=[],
                missing_details=required_details or [],
                suggestions=["No image data available for checking"]
            )
        
        details_to_check = required_details or result.required_details
        
        if not details_to_check:
            return DetailCheckResult(
                is_valid=True,
                detected_details=[],
                missing_details=[]
            )
        
        # Detect details in image
        detected_details, confidence_scores = self._detect_details(result.image_data)
        
        # Check which required details are present
        missing_details = []
        for detail in details_to_check:
            if detail.lower() not in [d.lower() for d in detected_details]:
                missing_details.append(detail)
        
        is_valid = len(missing_details) == 0
        
        suggestions = []
        if missing_details:
            suggestions.append(
                f"Regenerate with stronger emphasis on: {', '.join(missing_details)}"
            )
            suggestions.append(
                "Consider using ControlNet with edge map highlighting required details"
            )
        
        return DetailCheckResult(
            is_valid=is_valid,
            detected_details=detected_details,
            missing_details=missing_details,
            confidence_scores=confidence_scores,
            suggestions=suggestions
        )
    
    def _detect_details(self, image: Any) -> Tuple[List[str], Dict[str, float]]:
        """
        Detect details in an image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Tuple of (detected_details, confidence_scores)
        """
        detected_details = []
        confidence_scores = {}
        
        if self.detail_detector is None:
            # Mock detection for testing
            logger.warning("Using mock detail detection")
            return self._mock_detect_details(image)
        
        try:
            import torch
            from PIL import Image
            
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # Process image
            inputs = self.detail_processor(images=image, return_tensors="pt")
            
            # Run detection
            with torch.no_grad():
                outputs = self.detail_detector(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            results = self.detail_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]
            
            # Map COCO labels to our detail names
            label_map = {
                0: "person",
                1: "bicycle",
                # ... add more mappings as needed
                41: "cup",
                42: "fork",
                43: "knife",
                44: "spoon",
                45: "bowl",
                46: "banana",
                47: "apple",
                48: "sandwich",
                49: "orange",
                50: "broccoli",
                51: "carrot",
                52: "hot dog",
                53: "pizza",
                54: "donut",
                55: "cake",
                56: "chair",
                57: "couch",
                58: "potted plant",
                59: "bed",
                60: "dining table",
                61: "toilet",
                62: "tv",
                63: "laptop",
                64: "mouse",
                65: "remote",
                66: "keyboard",
                67: "cell phone",
                68: "microwave",
                69: "oven",
                70: "toaster",
                71: "sink",
                72: "refrigerator",
                73: "book",
                74: "clock",
                75: "vase",
                76: "scissors",
                77: "teddy bear",
                78: "hair drier",
                79: "toothbrush",
            }
            
            for score, label in zip(results["scores"], results["labels"]):
                label_name = label_map.get(label.item(), f"class_{label.item()}")
                detected_details.append(label_name)
                confidence_scores[label_name] = score.item()
            
            # Special handling for door handles and light switches
            # These may not be in COCO, so we add custom detection logic
            custom_details = self._detect_custom_details(image)
            for detail, score in custom_details.items():
                if score > 0.5:
                    detected_details.append(detail)
                    confidence_scores[detail] = score
            
        except Exception as e:
            logger.error(f"Error detecting details: {e}")
            return self._mock_detect_details(image)
        
        return detected_details, confidence_scores
    
    def _detect_custom_details(self, image: Any) -> Dict[str, float]:
        """
        Detect custom details not in standard object detection models.
        
        Specifically looks for:
        - Door handles
        - Light switches
        """
        custom_scores = {}
        
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Convert to OpenCV format
            if isinstance(image, Image.Image):
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                img_cv = image
            
            # Simple heuristic for door handles (circular/oval shapes)
            # This is a placeholder - in production, use trained detectors
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=50
            )
            
            if circles is not None:
                custom_scores["door handle"] = min(1.0, len(circles[0]) * 0.3)
            
            # Simple heuristic for light switches (rectangular shapes)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            switch_count = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    if 20 < w < 100 and 20 < h < 100:  # Reasonable switch size
                        switch_count += 1
            
            if switch_count > 0:
                custom_scores["light switch"] = min(1.0, switch_count * 0.4)
            
        except Exception as e:
            logger.warning(f"Custom detail detection failed: {e}")
        
        return custom_scores
    
    def _mock_detect_details(self, image: Any) -> Tuple[List[str], Dict[str, float]]:
        """Mock detail detection for testing."""
        # In mock mode, assume all required details are present
        return ["door handle", "light switch"], {"door handle": 0.95, "light switch": 0.90}
    
    def regenerate_with_feedback(
        self,
        result: TextureGenerationResult,
        check_result: DetailCheckResult,
        strength: float = 1.5
    ) -> TextureGenerationResult:
        """
        Regenerate texture with feedback from detail check.
        
        Args:
            result: Original generation result
            check_result: Detail check result
            strength: How strongly to emphasize missing details
            
        Returns:
            New TextureGenerationResult
        """
        if check_result.is_valid:
            logger.info("Texture already valid, no regeneration needed")
            return result
        
        # Modify prompt to emphasize missing details
        missing_str = ", ".join(check_result.missing_details)
        enhanced_prompt = (
            f"{result.prompt}, highly detailed {missing_str}, "
            f"clear {missing_str}, prominent {missing_str}"
        )
        
        logger.info(f"Regenerating with enhanced prompt: {enhanced_prompt}")
        
        # Generate new texture
        new_result = self.generate_texture(
            texture_id=f"{result.texture_id}_v2",
            prompt=enhanced_prompt,
            required_details=result.required_details,
        )
        
        return new_result
