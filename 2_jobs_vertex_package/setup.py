import setuptools
setuptools.setup(
    name='trainer-model',
    version='1.0.0',
    author='Jose',
    description='job vertex v3 - enviar package custom de entrenamiento',
    install_requires = ['google-cloud-bigquery==3.11.4', 'db-dtypes', 'gcsfs==2023.9.2', 'pandas==2.0.3', 'numpy==1.23.5', 'scikit-learn==1.3.1'],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True
)
