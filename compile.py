import os
import glob
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

# Directories to compile
directories_to_compile = ['rag', 'desktop']

# Exclude entry point scripts or files that shouldn't be compiled
exclude_files = ['desktop/main.py']

extensions = []

for dir_name in directories_to_compile:
    # Find all .py files in the directory
    for py_file in glob.glob(os.path.join(dir_name, '**', '*.py'), recursive=True):
        # Normalize path
        py_file = py_file.replace('\\', '/')
        
        # Skip excluded files
        if any(ex in py_file for ex in exclude_files):
            print(f"Skipping: {py_file}")
            continue

        # Create module name from path (e.g., rag.config)
        module_name = py_file.replace('/', '.').replace('.py', '')
        
        print(f"Adding extension: {module_name} -> {py_file}")
        extensions.append(Extension(module_name, [py_file]))

if not extensions:
    print("No files to compile.")
    exit(0)

# Compile the extensions
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
        build_dir="build",
        annotate=False
    ),
    script_args=["build_ext", "--inplace"]
)

print("Compilation complete. You can now test the .pyd files and optionally remove the original .py files (except main.py) before distributing.")
