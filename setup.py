from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ecgxai',
      version='0.1.1',
      description='Neatly packaged AI methods for ECG analysis',
      author='Rutger van de Leur',
      author_email='r.r.vandeleur@umcutrecht.nl',
      long_description=long_description,
      long_description_content_type="text/markdown",
      license='GNU AGPLv3',
      packages=find_packages(include=['ecgxai', 'ecgxai.*']),
      url="https://github.com/rutgervandeleur/ecgxai",
      python_requires=">=3.6",
      install_requires=[
          "pytorch_lightning==1.5.10",
          "torchmetrics==0.9.1",
          "torch==1.9.*",
          "numpy",
          "scipy",
          "pandas",
          "scikit-learn",
          "tqdm"
        ]
      )