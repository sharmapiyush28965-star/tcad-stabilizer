from pathlib import Path

from setuptools import setup, find_packages

HERE = Path(__file__).parent
LONG_DESC = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="tcad-stabilizer",
    version="0.6.0",
    description="Drift-diffusion TCAD device solver (1D/2D/3D diodes and MOSFETs) with a Newton stabilizer that converges on hard bias points",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Piyush Sharma",
    author_email="sharmapiyush28965@gmail.com",
    # TODO: point this at the GitHub repo once it exists, e.g.
    # url="https://github.com/<your-github-username>/tcad-stabilizer"
    url="https://pypi.org/project/tcad-stabilizer/",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    extras_require={"plot": ["matplotlib"]},
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="tcad drift-diffusion device-simulation semiconductor diode mosfet newton",
)
