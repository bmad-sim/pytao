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

Start a quick PyTao session with IPython using:

```bash
$ pytao -init tao.init
----------------------------------------------------------
Initializing Tao object with the following: -init tao.init
Type`tao.` and hit tab to see available commands.
----------------------------------------------------------

In [1]: tao
Out[1]: <Tao init='-init tao.init' so_lib='/path/to/libtao.so'>
```

To use PyTao's plotting with Matplotlib (`"mpl"`, alternatively Bokeh `"bokeh"`), use the `--pyplot` argument.

```bash
$ pytao --pyplot=mpl -init "$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init"
------------------------------------------------------------------------------------------------------
Initializing Tao object with: -init /Users/klauer/Repos/bmad/bmad-doc/tao_examples/cbeta_cell/tao.init

Type `tao.` and hit tab to see available commands.
------------------------------------------------------------------------------------------------------

* Matplotlib mode configured. Pyplot available as `plt`. *
```

Start pytao's GUI from the command-line with:

```bash
pytao-gui -init tao.init
```

## License

[GNU General Public License](LICENSE)
