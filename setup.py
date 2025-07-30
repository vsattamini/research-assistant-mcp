from pathlib import Path
from setuptools import find_packages, setup

# ----------------------------------------------------------------------------
# Package metadata
# ----------------------------------------------------------------------------
PACKAGE_NAME = "research-assistant-mcp"  # PyPI friendly name
SRC_DIRECTORY = "src"  # Root of the importable packages

# Versioning strategy — keep simple for the take-home assignment
VERSION = "0.1.0"

# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()

# Long description from the README (if it exists)
readme_path = HERE / "README.md"
LONG_DESCRIPTION = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Requirements from requirements.txt
requirements_path = HERE / "requirements.txt"
if requirements_path.exists():
    INSTALL_REQUIRES = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    INSTALL_REQUIRES = []

# ----------------------------------------------------------------------------
# Setup declaration
# ----------------------------------------------------------------------------
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="Research assistant that orchestrates multi-step agent workflows (MCP style)",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.8",
    url="https://github.com/your/repo",  # Update if you have a public repo
    package_dir={"": SRC_DIRECTORY},
    packages=find_packages(where=SRC_DIRECTORY),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Optional: expose a console entry-point to launch the Gradio app quickly.
    # A minimal wrapper function is already present in src/main.py under the
    # `if __name__ == "__main__"` guard, so we simply reuse the module.
    # Reviewers can now run `research-assistant` after installation.
    entry_points={
        "console_scripts": [
            # This will execute `python -m src.main` under the hood.
            "research-assistant=src.main:main",  # type: ignore[attr-defined]
        ]
    },
    zip_safe=False,
)
