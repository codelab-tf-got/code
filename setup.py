setup(
  name='wide_n_deep_codelab_got',
  version='0.1.0',
  description='A codelab example for predicting Game of Thrones deaths',
  author='Santiago Saavedra, Gema Parreño',
  author_email='santiagosaavedra@gmail.com',
  license='Apache Software License 2',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
  ],
  packages=['wideep'],
  install_requires=[
    'numpy',
    'scipy',
    'git+https://github.com/tflearn/tflearn.git@master#tflearn',
    'pandas',
    'tensorflow>=0.11',
    'sklearn',
  ],
)
