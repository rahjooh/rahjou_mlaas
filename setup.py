from setuptools import setup, find_packages

setup(
    name='mlaas',
    description='mlaas description',
    version='0.2',
    url='https://git.tapsell.ir/brain/mlaas',
    packages=find_packages(),
    keywords=['pip','mlaas'],
    install_requires=[
       "numpy==1.18.1",
       "numpydoc==0.9.2",
       "scipy==1.4.1",
       "pydantic==1.5.1",
       "sklearn==0.0",
       "pymongo==3.11.0",
       "dill==0.3.2"
   ],
    )
