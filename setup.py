import sys

from setuptools import find_packages, setup


def get_version(filename):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version


version = get_version(filename="src/gym_duckietown/__init__.py")

line = "daffy"

install_requires = [
    "gym",
    "numpy",
    "pyglet==1.5",
    "pyzmq",
    "opencv-python",
    "PyYAML",
    "PyGeometry-z6",
    "carnivalmirror",
    "zuper-commons-z6",
    "typing_extensions",
    "Pillow",
]

system_version = tuple(sys.version_info)[:3]

if system_version < (3, 7):
    install_requires.append("dataclasses")


setup(
    name=f"duckietown-gym-{line}",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    version=version,
    keywords="duckietown, environment, agent, rl, openaigym, openai-gym, gym",
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "dt-check-gpu=gym_duckietown.check_hw:main",
        ],
    },
)
