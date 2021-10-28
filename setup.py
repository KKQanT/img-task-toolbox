from setuptools import setup

install_requires = []
with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='img_task_toolbox',
    version='1.0.0',
    description='random code from other great people that being use frequently in my projects',
    author='KKQanT',
    author_email="asskarnwin@gmail.com",
    packages=['img_task_toolbox'],
    install_requires=install_requires,
)