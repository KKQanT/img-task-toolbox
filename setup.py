from setuptools import setup, find_packages

install_requires = []
with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

packages = find_packages()

setup(
    name='img_task_toolbox',
    version='1.0.0',
    description='random code from other great people that being use frequently in my projects',
    author='KKQanT',
    author_email="asskarnwin@gmail.com",
    packages=packages,
    install_requires=install_requires,
    url='https://github.com/KKQanT/img-task-toolbox'
)