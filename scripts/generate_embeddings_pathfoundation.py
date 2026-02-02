#!/usr/bin/env python3
"""
Path Foundation Batch Embedding Generation for Histopathology Slides.

Uses Google's Path Foundation model (HAI-DEF) for pathology-specific embeddings.
Path Foundation produces 384-dimensional embeddings specifically trained on
histopathology images - more appropriate than generic DINOv2 for pathology tasks.

Requirements:
    - Python 3.11 (TensorFlow not compatible with 3.14)
    - TensorFlow 2.x
    - HuggingFace Hub access to google/path-foundation (gated model)

Usage:
    # Activate TF venv first:
    source .venv-tf/bin/activate
    
    python scripts/generate_embeddings_pathfoundation.py \
        --input data/slides \
        --output data/embeddings_pathfoundation \
        --batch-size 32
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SlideResult:
    """Result of processing a single slide."""
    slide_id: str
    status: str  # success, skipped, failed
    num_patches: int = 0
    embedding_shape: Tuple[int, int] = (0, 0)
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ProcessingStats:
    """Overall processing statistics."""
    total_slides: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    total_patches: int = 0
    start_time: float = 0.0
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def eta_seconds(self) -> float:
        if self.processed == 0:
            return 0
        remaining = self.total_slides - self.processed - self.skipped - self.failed
        avg_time = self.elapsed() / self.processed
        return remaining * avg_time
    
    def eta_str(self) -> str:
        eta = self.eta_seconds()
        if eta <= 0:
            return "N/A"
        return str(timedelta(seconds=int(eta)))


class PathFoundationBatchEmbedder:
    """
    Path Foundation batch embedding generator for histopathology slides.
    
    Path Foundation (google/path-foundation) is part of Google's Health AI 
    Developer Foundations (HAI-DEF). It produces 384-dimensional embeddings 
    specifically trained on pathology images using self-supervised learning 
    on TCGA data.
    
    Key differences from DINOv2:
    - Domain-specific: trained on 60M pathology patches vs generic images
    - Multi-scale: trained at 5x, 10x, 20x magnifications
    - Optimized for H&E: specifically designed for hematoxylin & eosin stained slides
    """
    
    EMBEDDING_DIM = 384
    INPUT_SIZE = 224
    
    def __init__(
        self,
        batch_size: int = 32,
        patch_size: int = 224,
        max_patches_per_slide: int = 2000,
        tissue_threshold: float = 0.15,
    ):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_patches = max_patches_per_slide
        self.tissue_threshold = tissue_threshold
        
        self._model = None
        self._infer_fn = None
        
    def _load_model(self) -> None:
        """Lazy load Path Foundation model from HuggingFace."""
        if self._model is not None:
            return
            
        import tensorflow as tf
        from huggingface_hub import snapshot_download
        
        logger.info("Loading Path Foundation model from HuggingFace...")
        logger.info("Note: This is a gated model - requires HF login and license acceptance")
        
        # Download the model files from HuggingFace Hub
        model_dir = snapshot_download(
            repo_id="google/path-foundation",
            allow_patterns=["*.pb", "variables/*", "keras_metadata.pb"]
        )
        
        # Load as TensorFlow SavedModel
        self._model = tf.saved_model.load(model_dir)
        
        # Get the inference function
        self._infer_fn = self._model.signatures["serving_default"]
        
        logger.info("Path Foundation model loaded successfully")
        logger.info(f"Embedding dimension: {self.EMBEDDING_DIM}")
        
    def extract_patches(
        self,
        slide_path: Path,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract tissue patches from a whole slide image.
        
        Returns:
            patches: List of 224x224 RGB patches
            coords: List of (x, y) coordinates at level 0
        """
        import cv2
        
        try:
            import openslide
        except ImportError:
            logger.error("openslide-python not installed. Run: pip install openslide-python")
            raise
            
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            return [], []
            
        # Get slide dimensions
        width, height = slide.dimensions
        
        # Use level 0 or 1 depending on slide size
        level = 0
        if slide.level_count > 1 and width * height > 100000 * 100000:
            level = 1
        
        level_dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        logger.debug(f"Slide {slide_path.name}: {width}x{height}, using level {level}")
        
        patches = []
        coords = []
        
        # Calculate step (with some overlap reduction for efficiency)
        step_at_level = self.patch_size
        step_at_level0 = int(step_at_level * downsample)
        
        # Grid sampling
        y_steps = list(range(0, int(level_dims[1]) - step_at_level, step_at_level))
        x_steps = list(range(0, int(level_dims[0]) - step_at_level, step_at_level))
        
        for y in y_steps:
            for x in x_steps:
                if len(patches) >= self.max_patches:
                    break
                    
                # Convert to level 0 coordinates
                x0 = int(x * downsample)
                y0 = int(y * downsample)
                
                try:
                    # Read region at level
                    region = slide.read_region((x0, y0), level, (self.patch_size, self.patch_size))
                    region = region.convert("RGB")
                    patch = np.array(region)
                    
                    # Tissue detection
                    if self._is_tissue(patch):
                        patches.append(patch)
                        coords.append((x0, y0))
                        
                except Exception:
                    continue
                    
            if len(patches) >= self.max_patches:
                break
                
        slide.close()
        return patches, coords
        
    def _is_tissue(self, patch: np.ndarray) -> bool:
        """
        Detect if patch contains tissue (not background).
        """
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Check mean intensity (tissue is typically between 50-220)
        mean_val = np.mean(gray)
        if mean_val < 30 or mean_val > 230:
            return False
            
        # Check standard deviation (tissue has texture)
        std_val = np.std(gray)
        if std_val < 15:
            return False
            
        # Check for sufficient non-white pixels
        non_white = np.sum(gray < 220) / gray.size
        if non_white < self.tissue_threshold:
            return False
            
        return True
        
    def embed_patches(self, patches: List[np.ndarray]) -> np.ndarray:
        """
        Generate Path Foundation embeddings for a list of patches.
        
        Args:
            patches: List of 224x224 RGB numpy arrays
            
        Returns:
            embeddings: (N, 384) array of embeddings
        """
        self._load_model()
        
        import tensorflow as tf
        
        if len(patches) == 0:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)
            
        all_embeddings = []
        
        for i in range(0, len(patches), self.batch_size):
            batch = patches[i:i + self.batch_size]
            
            # Convert batch to tensor
            # Path Foundation expects [0, 1] normalized float32 tensors
            batch_array = np.stack(batch, axis=0).astype(np.float32) / 255.0
            batch_tensor = tf.constant(batch_array)
            
            # Forward pass
            outputs = self._infer_fn(batch_tensor)
            
            # Extract embeddings from output
            embeddings = outputs['output_0'].numpy()
            all_embeddings.append(embeddings)
            
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Keep float16 for storage efficiency
        embeddings = embeddings.astype(np.float16)
        
        return embeddings
        
    def embed_slide(
        self,
        slide_path: Path,
        output_dir: Path,
        force: bool = False,
    ) -> SlideResult:
        """
        Process a single slide: extract patches, generate embeddings, save.
        """
        slide_id = slide_path.stem
        output_path = output_dir / f"{slide_id}.npy"
        coords_path = output_dir / f"{slide_id}_coords.npy"
        
        start_time = time.time()
        
        # Check if already processed
        if output_path.exists() and coords_path.exists() and not force:
            return SlideResult(
                slide_id=slide_id,
                status="skipped",
            )
            
        try:
            # Extract patches
            patches, coords = self.extract_patches(slide_path)
            
            if len(patches) == 0:
                return SlideResult(
                    slide_id=slide_id,
                    status="failed",
                    error="No tissue patches extracted",
                    duration_seconds=time.time() - start_time,
                )
                
            # Generate embeddings
            embeddings = self.embed_patches(patches)
            
            # Save atomically (write to temp, then rename)
            temp_emb = output_path.with_name(output_path.stem + '.tmp.npy')
            temp_coords = coords_path.with_name(coords_path.stem + '.tmp.npy')
            
            np.save(temp_emb, embeddings)
            np.save(temp_coords, np.array(coords))
            
            # Atomic rename
            temp_emb.rename(output_path)
            temp_coords.rename(coords_path)
            
            duration = time.time() - start_time
            
            return SlideResult(
                slide_id=slide_id,
                status="success",
                num_patches=len(patches),
                embedding_shape=embeddings.shape,
                duration_seconds=duration,
            )
            
        except Exception as e:
            return SlideResult(
                slide_id=slide_id,
                status="failed",
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
            
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.svs",
        force: bool = False,
    ) -> List[SlideResult]:
        """
        Process all slides in a directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all slides
        slides = list(input_dir.glob(pattern))
        
        # Also check for other common formats
        if pattern == "*.svs":
            slides.extend(input_dir.glob("*.tif"))
            slides.extend(input_dir.glob("*.tiff"))
            slides.extend(input_dir.glob("*.ndpi"))
            
        slides = sorted(set(slides))
        
        if not slides:
            logger.warning(f"No slides found in {input_dir} with pattern {pattern}")
            return []
            
        logger.info(f"Found {len(slides)} slides to process")
        logger.info("Using Path Foundation (google/path-foundation) for pathology-specific embeddings")
        
        # Initialize stats
        stats = ProcessingStats(
            total_slides=len(slides),
            start_time=time.time(),
        )
        
        results = []
        
        # Pre-load model before starting
        self._load_model()
        
        # Process slides
        for i, slide_path in enumerate(slides, 1):
            result = self.embed_slide(slide_path, output_dir, force=force)
            results.append(result)
            
            # Update stats
            if result.status == "success":
                stats.processed += 1
                stats.total_patches += result.num_patches
            elif result.status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
                
            # Log progress
            self._log_progress(result, i, stats)
            
        # Final summary
        self._log_summary(stats, output_dir)
        
        return results
        
    def _log_progress(self, result: SlideResult, current: int, stats: ProcessingStats):
        """Log progress for a completed slide."""
        total = stats.total_slides
        
        if result.status == "success":
            patches_str = f"{result.num_patches} patches"
            time_str = f"{result.duration_seconds:.1f}s"
            logger.info(
                f"[{current}/{total}] {result.slide_id}: {patches_str} in {time_str} "
                f"(ETA: {stats.eta_str()})"
            )
        elif result.status == "skipped":
            logger.info(f"[{current}/{total}] {result.slide_id}: skipped (already exists)")
        else:
            logger.warning(f"[{current}/{total}] {result.slide_id}: FAILED - {result.error}")
            
    def _log_summary(self, stats: ProcessingStats, output_dir: Path):
        """Log final processing summary."""
        elapsed = timedelta(seconds=int(stats.elapsed()))
        
        logger.info("=" * 60)
        logger.info("Processing Complete - Path Foundation Embeddings")
        logger.info("=" * 60)
        logger.info(f"Model: google/path-foundation (HAI-DEF)")
        logger.info(f"Total slides: {stats.total_slides}")
        logger.info(f"  Processed: {stats.processed}")
        logger.info(f"  Skipped:   {stats.skipped}")
        logger.info(f"  Failed:    {stats.failed}")
        logger.info(f"Total patches: {stats.total_patches}")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Output: {output_dir}")
        
        if stats.processed > 0:
            avg_time = stats.elapsed() / stats.processed
            avg_patches = stats.total_patches / stats.processed
            logger.info(f"Avg time/slide: {avg_time:.1f}s")
            logger.info(f"Avg patches/slide: {avg_patches:.0f}")


def save_processing_log(results: List[SlideResult], output_dir: Path):
    """Save processing results to a JSON log file."""
    log_path = output_dir / "processing_log.json"
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "google/path-foundation",
        "framework": "HAI-DEF (Health AI Developer Foundations)",
        "embedding_dim": 384,
        "total": len(results),
        "success": sum(1 for r in results if r.status == "success"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "results": [asdict(r) for r in results],
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
        
    logger.info(f"Processing log saved to {log_path}")


def check_hf_access():
    """Check HuggingFace authentication and model access."""
    from huggingface_hub import HfApi, hf_hub_download
    
    api = HfApi()
    
    # Check if logged in
    try:
        user = api.whoami()
        logger.info(f"Logged in to HuggingFace as: {user.get('name', 'Unknown')}")
    except Exception:
        logger.error("HuggingFace not authenticated!")
        logger.error("Run: python -c \"from huggingface_hub import login; login()\"")
        logger.error("Then accept the license at: https://huggingface.co/google/path-foundation")
        return False
        
    # Check model access by trying to download a small file
    try:
        hf_hub_download(repo_id="google/path-foundation", filename="README.md")
        logger.info("Path Foundation model access verified")
        return True
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "restricted" in error_msg.lower():
            logger.error("Path Foundation access denied - license not accepted")
            logger.error("Accept the license at: https://huggingface.co/google/path-foundation")
        else:
            logger.error(f"Cannot access Path Foundation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate Path Foundation embeddings for histopathology slides",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing slide files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/embeddings_pathfoundation"),
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=2000,
        help="Maximum patches per slide",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.svs",
        help="Glob pattern for slide files",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process existing files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--skip-auth-check",
        action="store_true",
        help="Skip HuggingFace authentication check",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Check HuggingFace access
    if not args.skip_auth_check:
        if not check_hf_access():
            sys.exit(1)
        
    # Validate input directory
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
        
    # Create embedder
    embedder = PathFoundationBatchEmbedder(
        batch_size=args.batch_size,
        max_patches_per_slide=args.max_patches,
    )
    
    # Process directory
    results = embedder.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        force=args.force,
    )
    
    # Save processing log
    if results:
        save_processing_log(results, args.output)
        
    # Exit with error code if any failures
    failed = sum(1 for r in results if r.status == "failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
