# setup.py
from setuptools import setup, find_packages, Extension
import pybind11
import numpy as np

# C++ extension module
forman_module = Extension(
    'zs_ttcs._forman',  # This will be imported as zs_ttcs._forman
    sources=[
        'cpp/forman.cpp',
        'cpp/bindings.cpp'
    ],
    include_dirs=[
        pybind11.get_include(),
        np.get_include(),
        'cpp'  # For forman.h
    ],
    language='c++',
    extra_compile_args=['-std=c++17', '-O3', '-Wall', '-shared', '-fPIC'],
    extra_link_args=['-std=c++17'],
)

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ZS-TTCS",
    version="0.1.0",
    author="Gergo Dioszegi",
    author_email="dijogergo@gmail.com",
    description="ZeroShot-Topological Tree Crown Segmentor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DijoG/ZS-TTCS",
    
    # Package discovery
    packages=find_packages(include=["zs_ttcs", "zs_ttcs.*"]),
    
    # C++ extension
    ext_modules=[forman_module],
    
    # Dependencies
    python_requires=">=3.7",
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "viz": [
            "plotly>=5.0",
            "folium>=0.12",
        ],
    },
    
    # Command-line scripts
    entry_points={
        "console_scripts": [
            "zs-ttcs-step1=zs_ttcs.cli:step1_cli",
            "zs-ttcs-step2=zs_ttcs.cli:step2_cli",
            "zs-ttcs-step3=zs_ttcs.cli:step3_cli",
            "zs-ttcs-pipeline=zs_ttcs.cli:pipeline_cli",
            "zs-ttcs=zs_ttcs.cli:main",  
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        "zs_ttcs": ["py.typed"],  # For type hints
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://github.com/DijoG/ZS-TTCS",
        "Source": "https://github.com/DijoG/ZS-TTCS",
        "Bug Reports": "https://github.com/DijoG/ZS-TTCS/issues",
    },
    
    # Zip safe - set to False because we have C extensions
    zip_safe=False,
)