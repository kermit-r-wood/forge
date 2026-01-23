import subprocess
import sys
import os
import shutil
from pathlib import Path

def clean_build_artifacts():
    """Clean up previous build artifacts."""
    root_dir = Path(__file__).parent.parent
    dist_dir = root_dir / "dist"
    build_dir = root_dir / "build"
    
    if dist_dir.exists():
        print(f"Cleaning {dist_dir}...")
        shutil.rmtree(dist_dir)
    
    if build_dir.exists():
        print(f"Cleaning {build_dir}...")
        shutil.rmtree(build_dir)

def build():
    """Run PyInstaller build."""
    root_dir = Path(__file__).parent.parent
    spec_file = root_dir / "packaging" / "forge.spec"
    
    if not spec_file.exists():
        print(f"Error: Spec file not found at {spec_file}")
        sys.exit(1)
        
    print(f"Building Forge from {spec_file}...")
    
    # We run pyinstaller as a subprocess
    # Using 'uv run' to ensure we use the project's environment
    # or just 'pyinstaller' if we assume it's in the path (which it should be if running this script via uv run)
    
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        str(spec_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=root_dir)
        print("Build completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'pyinstaller' not found. Make sure it is installed in your environment.")
        sys.exit(1)

if __name__ == "__main__":
    clean_build_artifacts()
    build()
