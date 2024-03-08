import setuptools

setuptools.setup(
    name="zebrafish_rtrbm",
    version="0.0.1",
    author="Sebastian Quiroz Monnens, Casper Peters, Luuk Willem Hesselink, Kasper Smeets",
    author_email="luukhesselink@donders.ru.nl",
    description="",
    url="",
    license="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.20.3',
        'scipy>=1.9.1',
        'tqdm',
        'matplotlib>=3.7',
        'pandas'
    ]
)
