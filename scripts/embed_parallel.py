#!/usr/bin/env python3
"""
Parallelized Path Foundation embedding with concurrent extraction.
Uses multiprocessing for extraction + main thread for GPU embedding.
"""

import os
import sys
import glob
import queue
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Lazy imports
openslide = None
tf = None
cv2 = None

def import_libs():
    global openslide, tf, cv2
    if openslide is None:
        import openslide as _os
        openslide = _os
    if tf is None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as _tf
        tf = _tf
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/workspace/embed_parallel.log')
    ]
)
logger = logging.getLogger(__name__)


def is_tissue(patch, threshold=0.1):
    """Check if patch contains tissue (not background)."""
    import cv2
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    tissue_ratio = np.mean(saturation > 20) 
    return tissue_ratio > threshold


def extract_patch_batch(args):
    """Worker function: extract patches from a batch of coordinates.
    
    Each worker opens its own slide handle (OpenSlide not thread-safe).
    Returns list of (patch, coord) tuples.
    """
    slide_path, coords, patch_size = args
    import openslide
    import cv2
    
    results = []
    try:
        slide = openslide.OpenSlide(slide_path)
        for x, y in coords:
            try:
                region = slide.read_region((x, y), 0, (patch_size, patch_size))
                patch = np.array(region.convert('RGB'))
                if is_tissue(patch):
                    results.append((patch, (x, y)))
            except Exception:
                pass
        slide.close()
    except Exception as e:
        pass
    return results


def get_grid_coordinates(slide_path, patch_size, step):
    """Get all grid coordinates for a slide."""
    import openslide
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.dimensions
    slide.close()
    
    coords = []
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            coords.append((x, y))
    return coords, w, h


def chunk_list(lst, n_chunks):
    """Split list into n roughly equal chunks."""
    chunk_size = max(1, len(lst) // n_chunks)
    chunks = []
    for i in range(0, len(lst), chunk_size):
        chunks.append(lst[i:i + chunk_size])
    return chunks


class PathFoundationEmbedder:
    """Embeds patches using TF SavedModel Path Foundation."""
    
    def __init__(self, model_path='/workspace/path-foundation'):
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        import_libs()
        logger.info('Loading Path Foundation model...')
        self.model = tf.saved_model.load(self.model_path)
        logger.info('Model loaded')
    
    def embed_batch(self, patches):
        """Embed a batch of patches."""
        import_libs()
        batch_array = np.stack(patches).astype(np.float32) / 255.0
        batch_tensor = tf.constant(batch_array)
        outputs = self.model.signatures['serving_default'](batch_tensor)
        embeddings = list(outputs.values())[0].numpy()
        return embeddings


def process_slide_parallel(slide_path, output_dir, patch_size=224, step=None,
                          max_patches=20000, batch_size=256, n_workers=6):
    """Process a single slide with parallel extraction."""
    
    slide_name = Path(slide_path).stem
    output_path = Path(output_dir) / f"{slide_name}.npy"
    coord_path = Path(output_dir) / f"{slide_name}_coords.npy"
    
    if output_path.exists():
        logger.info(f"SKIP: {slide_name} already exists")
        return
    
    logger.info(f"Processing: {slide_name}")
    
    if step is None:
        step = patch_size  # Dense sampling
    
    # Get all grid coordinates
    coords, w, h = get_grid_coordinates(slide_path, patch_size, step)
    logger.info(f"Slide: {w}x{h}, Grid: {len(coords)} positions")
    
    # Shuffle coordinates for better load balancing
    np.random.shuffle(coords)
    
    # Split coordinates into chunks for workers
    coord_chunks = chunk_list(coords, n_workers * 4)  # More chunks than workers for better balancing
    
    # Prepare worker arguments
    worker_args = [(slide_path, chunk, patch_size) for chunk in coord_chunks]
    
    # Load embedder
    embedder = PathFoundationEmbedder()
    embedder.load_model()
    
    all_patches = []
    all_coords = []
    all_embeddings = []
    
    start_time = time.time()
    extract_time = 0
    embed_time = 0
    
    # Parallel extraction with ProcessPoolExecutor
    logger.info(f"Starting parallel extraction with {n_workers} workers...")
    extract_start = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(extract_patch_batch, args): i for i, args in enumerate(worker_args)}
        
        patches_batch = []
        coords_batch = []
        
        for future in as_completed(futures):
            try:
                results = future.result()
                for patch, coord in results:
                    patches_batch.append(patch)
                    coords_batch.append(coord)
                    
                    # Embed when we have enough patches
                    if len(patches_batch) >= batch_size:
                        embed_start = time.time()
                        embeddings = embedder.embed_batch(patches_batch[:batch_size])
                        embed_time += time.time() - embed_start
                        
                        all_embeddings.append(embeddings)
                        all_coords.extend(coords_batch[:batch_size])
                        
                        patches_batch = patches_batch[batch_size:]
                        coords_batch = coords_batch[batch_size:]
                        
                        total_embedded = sum(e.shape[0] for e in all_embeddings)
                        elapsed = time.time() - start_time
                        logger.info(f"Embedded {total_embedded} patches ({total_embedded/elapsed:.1f}/sec)")
                        
                        # Check max patches
                        if total_embedded >= max_patches:
                            logger.info(f"Reached max_patches ({max_patches})")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
            except Exception as e:
                logger.warning(f"Worker error: {e}")
    
    extract_time = time.time() - extract_start
    
    # Embed remaining patches
    if patches_batch and sum(e.shape[0] for e in all_embeddings) < max_patches:
        embed_start = time.time()
        remaining = max_patches - sum(e.shape[0] for e in all_embeddings)
        to_embed = patches_batch[:remaining]
        if to_embed:
            embeddings = embedder.embed_batch(to_embed)
            all_embeddings.append(embeddings)
            all_coords.extend(coords_batch[:remaining])
        embed_time += time.time() - embed_start
    
    # Save results
    if all_embeddings:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_coords = np.array(all_coords)
        
        np.save(output_path, final_embeddings)
        np.save(coord_path, final_coords)
        
        total_time = time.time() - start_time
        logger.info(f"SUCCESS: {len(all_coords)} patches -> {output_path}")
        logger.info(f"Time: {total_time:.1f}s total, {extract_time:.1f}s extract, {embed_time:.1f}s embed")
        logger.info(f"Rate: {len(all_coords)/total_time:.1f} patches/sec")
    else:
        logger.warning(f"No tissue patches found in {slide_name}")


def main():
    parser = argparse.ArgumentParser(description='Parallel Path Foundation embedding')
    parser.add_argument('--slide-dir', default='/workspace/slides')
    parser.add_argument('--output-dir', default='/workspace/embeddings')
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--max-patches', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--test', action='store_true', help='Test on one slide only')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find slides
    slides = glob.glob(os.path.join(args.slide_dir, '*.svs'))
    slides.extend(glob.glob(os.path.join(args.slide_dir, '*.ndpi')))
    
    if not slides:
        logger.error(f"No slides found in {args.slide_dir}")
        return
    
    logger.info(f"Found {len(slides)} slides")
    
    if args.test:
        # Find first slide without embedding
        for slide in slides:
            name = Path(slide).stem
            if not (Path(args.output_dir) / f"{name}.npy").exists():
                slides = [slide]
                logger.info(f"Test mode: processing {name}")
                break
        else:
            logger.info("All slides already processed")
            return
    
    for i, slide_path in enumerate(slides, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Slide {i}/{len(slides)}: {Path(slide_path).name}")
        logger.info(f"{'='*60}")
        
        try:
            process_slide_parallel(
                slide_path,
                args.output_dir,
                patch_size=args.patch_size,
                step=args.step,
                max_patches=args.max_patches,
                batch_size=args.batch_size,
                n_workers=args.workers
            )
        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
