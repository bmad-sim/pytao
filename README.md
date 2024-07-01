# PyTao

[PyTao](<(https://bmad-sim.github.io/pytao/index.html)>) is Python interface for [Tao](https://www.classe.cornell.edu/bmad/tao.html), which is based on the Bmad subroutine library for relativistic charged–particle and X-ray simulations in accelerators and storage rings.

Documentation for Bmad and Tao, as well as information for downloading the code if needed is given on the [Bmad website](https://www.classe.cornell.edu/bmad).

| **`Documentation`**                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------- |
| [![Documentation](https://img.shields.io/badge/pytao-examples-green.svg)](https://bmad-sim.github.io/pytao/examples/basic/) |
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://bmad-sim.github.io/pytao/api/pytao/)        |

## Installation

See the [PyTao installation instructions](https://bmad-sim.github.io/pytao/user_docs/index.html) for details. The preferred installation method is using conda:

```
conda install -c conda-forge pytao
```

Currently PyTao requires an installed Bmad distribution compiled with the `ACC_ENABLE_SHARED="Y"` flag. This can be set in the `bmad_dist/util/dist_prefs` file.

## Current Build status and Release Info

<table><tr><td>All platforms:</td>
    <td>
      <a href="https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=12517&branchName=master">
        <img src="https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/pytao-feedstock?branchName=master">
      </a>
    </td>
  </tr>
</table>

| Name                                                                                                           | Downloads                                                                                                           | Version                                                                                                           | Platforms                                                                                                           |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-pytao-green.svg)](https://anaconda.org/conda-forge/pytao) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pytao.svg)](https://anaconda.org/conda-forge/pytao) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pytao.svg)](https://anaconda.org/conda-forge/pytao) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pytao.svg)](https://anaconda.org/conda-forge/pytao) |

## Resources

[Bmad website](https://www.classe.cornell.edu/bmad)

## Quickstart

Start a quick pytao session with IPython using:

```
$ pytao -init tao.init
------------------------------------------------------------
Initializing `tao` object with the following: -init tao.init
Type`tao.` and hit tab to see available commands.
------------------------------------------------------------

Python 3.12.0 | packaged by conda-forge | (main, Oct  3 2023, 08:36:57) [Clang 15.0.7 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.18.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: tao
Out[1]: <pytao.interface_commands.Tao at 0x13c43d4f0>
```

Start pytao's GUI from the command-line with:

```
pytao-gui -init tao.init
```

## License

[GNU General Public License](LICENSE)
