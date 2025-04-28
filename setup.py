from setuptools import setup, find_packages

with open("docs/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name="bottle-cap-inspection",
    version="1.0.0",
    author="Industry Sponsored Team",
    author_email="team@example.com",
    description="Bottle Cap Inspection System with YOLOv8 Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bottle-cap-inspection",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        'opencv-python>=4.5.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'pandas>=1.2.0',
        # Additional requirements from requirements.txt
        *requirements
    ],
    entry_points={
        'console_scripts': [
            'cap-inspection=detection.integrated_cap_inspection:main',
        ],
    },
)