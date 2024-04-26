from setuptools import setup, find_packages

setup(
    name='spacy_transformers',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'spacy>=3.0.0',
        'transformers',
        'torch'
    ],
    description='spaCy extension for integrating Hugging Face Transformers',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/spacy-transformers',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)