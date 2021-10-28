from setuptools import setup

install_requires = []
with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='img_task_toolbox',
    version='1.0',
    description='random code from other great people that being use frequently in my projects',
    author='KKQanT',
    packages=install_requires,
)