from setuptools import setup, find_packages


setup(
    name='fsm_strip_thickness_damage_analysis',
    version='1.0.1',
    url='https://github.com/petarmaric/fsm_strip_thickness_damage_analysis',
    license='BSD',
    author='Petar Maric',
    author_email='petarmaric@uns.ac.rs',
    description='Console app and Python API for strip thickness-dependent '\
                'damage analysis and visualization of the parametric model of '\
                'buckling and free vibration in prismatic shell structures, '\
                'as computed by the fsm_eigenvalue project.',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    platforms='any',
    py_modules=['fsm_strip_thickness_damage_analysis'],
    entry_points={
        'console_scripts': ['fsm_strip_thickness_damage_analysis=fsm_strip_thickness_damage_analysis:main'],
    },
    install_requires=open('requirements.txt').read().splitlines(),
)
