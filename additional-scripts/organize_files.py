import os
import shutil

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'data/raw',
        'models',
        'utils',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def copy_essential_files():
    """Copy only the essential files to their respective directories"""
    # Core Python files
    essential_files = {
        'main.py': '.',
        'utils/data_processor.py': 'utils/',
        'models/cancer_detector.py': 'models/',
        'requirements.txt': '.',
        'README.md': '.',
        'Dockerfile': '.',
        '.gitignore': '.'
    }
    
    for src, dest in essential_files.items():
        if os.path.exists(src):
            shutil.copy2(src, dest)
            print(f"Copied {src} to {dest}")

def main():
    print("Creating directory structure...")
    create_directory_structure()
    
    print("\nCopying essential files...")
    copy_essential_files()
    
    print("\nDone! The repository is now ready for GitHub.")

if __name__ == "__main__":
    main() 