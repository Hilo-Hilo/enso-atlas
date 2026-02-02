#!/usr/bin/env python3
"""
Setup script for Path Foundation access.

Path Foundation is a gated model that requires:
1. HuggingFace account
2. Accepting the HAI-DEF license
3. Storing your HF token

Run this script to verify and set up access.

Usage:
    source .venv-tf/bin/activate
    python scripts/setup_pathfoundation.py
"""

import sys
import os
from pathlib import Path


def check_tensorflow():
    """Verify TensorFlow is installed."""
    print("Checking TensorFlow...")
    try:
        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}")
        print(f"  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        return True
    except ImportError:
        print("  ERROR: TensorFlow not installed!")
        print("  Run: pip install tensorflow")
        return False


def check_huggingface():
    """Check HuggingFace Hub installation and login status."""
    print("\nChecking HuggingFace Hub...")
    try:
        from huggingface_hub import HfApi
        import huggingface_hub
        print(f"  huggingface_hub version: {huggingface_hub.__version__}")
    except ImportError:
        print("  ERROR: huggingface_hub not installed!")
        print("  Run: pip install huggingface_hub")
        return False, None
        
    api = HfApi()
    try:
        user = api.whoami()
        username = user.get('name', user.get('fullname', 'Unknown'))
        print(f"  Logged in as: {username}")
        return True, username
    except Exception:
        print("  NOT logged in to HuggingFace")
        return True, None


def check_model_access():
    """Check if we can access the Path Foundation model."""
    print("\nChecking Path Foundation model access...")
    from huggingface_hub import HfApi, hf_hub_download
    
    api = HfApi()
    
    # Check model info (doesn't require auth)
    try:
        info = api.model_info('google/path-foundation')
        print(f"  Model exists: google/path-foundation")
        print(f"  Gated: {info.gated}")
        print(f"  Downloads: {info.downloads}")
    except Exception as e:
        print(f"  ERROR checking model: {e}")
        return False
        
    # Try to download a small file (requires auth for gated models)
    try:
        # Try downloading just the README to test access
        path = hf_hub_download(
            repo_id='google/path-foundation',
            filename='README.md'
        )
        print(f"  Model access: GRANTED")
        return True
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "restricted" in error_msg.lower():
            print(f"  Model access: DENIED (need to accept license)")
            return False
        else:
            print(f"  Error: {error_msg[:100]}")
            return False


def setup_login():
    """Guide user through HuggingFace login."""
    print("\n" + "="*60)
    print("HuggingFace Login Required")
    print("="*60)
    print("""
To use Path Foundation, you need to:

1. Go to https://huggingface.co/google/path-foundation
2. Click "Acknowledge license" to accept the HAI-DEF terms
3. Create a token at https://huggingface.co/settings/tokens
4. Run the login command below

Option A - Interactive login:
    python -c "from huggingface_hub import login; login()"
    
Option B - With token directly:
    python -c "from huggingface_hub import login; login(token='YOUR_TOKEN_HERE')"

After logging in, run this script again to verify access.
""")


def test_model_loading():
    """Test actually loading the model."""
    print("\nTesting model loading...")
    try:
        import tensorflow as tf
        from huggingface_hub import snapshot_download
        
        print("  Downloading model files...")
        model_dir = snapshot_download(
            repo_id='google/path-foundation',
            allow_patterns=['*.pb', 'variables/*', 'keras_metadata.pb']
        )
        print(f"  Downloaded to: {model_dir}")
        
        print("  Loading SavedModel...")
        model = tf.saved_model.load(model_dir)
        print("  Model loaded successfully!")
        
        # Get inference signature
        infer = model.signatures["serving_default"]
        print(f"  Inference signature: {list(infer.structured_input_signature[1].keys())}")
        
        # Test with dummy input
        print("  Testing inference with dummy input...")
        dummy = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
        output = infer(dummy)
        embedding = output['output_0'].numpy()
        print(f"  Output shape: {embedding.shape}")
        print(f"  Embedding dimension: {embedding.shape[-1]}")
        
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("="*60)
    print("Path Foundation Setup")
    print("="*60)
    print("Model: google/path-foundation")
    print("Part of: Health AI Developer Foundations (HAI-DEF)")
    print("="*60)
    
    # Check TensorFlow
    if not check_tensorflow():
        sys.exit(1)
        
    # Check HuggingFace
    hf_installed, username = check_huggingface()
    if not hf_installed:
        sys.exit(1)
        
    if not username:
        setup_login()
        sys.exit(1)
        
    # Check model access
    if not check_model_access():
        print("\n" + "="*60)
        print("License Acceptance Required")
        print("="*60)
        print("""
You're logged in but haven't accepted the Path Foundation license.

Go to: https://huggingface.co/google/path-foundation
Click "Acknowledge license" button

Then run this script again.
""")
        sys.exit(1)
        
    # Test model loading
    if test_model_loading():
        print("\n" + "="*60)
        print("SUCCESS! Path Foundation is ready to use.")
        print("="*60)
        print("""
You can now run embedding generation:

    source .venv-tf/bin/activate
    python scripts/generate_embeddings_pathfoundation.py \\
        --input data/slides \\
        --output data/embeddings_pathfoundation
""")
    else:
        print("\n" + "="*60)
        print("Model loading failed - check errors above")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
