


# PyTao

[PyTao]((https://bmad-sim.github.io/pytao/index.html)) is Python interface for [Tao](https://www.classe.cornell.edu/bmad/tao.html), which is based on the Bmad subroutine library for relativistic charged–particle and X-ray simulations in accelerators and storage rings.

Documentation for Bmad and Tao, as well as information for downloading the code if needed is given on the [Bmad website](https://www.classe.cornell.edu/bmad).


**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://bmad-sim.github.io/pytao/dev_docs/api/index.html)  |
[![Documentation](https://img.shields.io/badge/pytao-examples-green.svg)](https://bmad-sim.github.io/pytao/examples/basic.html#)  |




## Installation

See the [PyTao installation instructions](https://bmad-sim.github.io/pytao/user_docs/index.html) for details. The preferred installation method is using conda:

```
$ conda install -c conda-forge pytao
```

Currently PyTao requires an installed Bmad distribution compiled with the `ACC_ENABLE_SHARED="Y"` flag. This can be set in the `bmad_dist/util/dist_prefs` file. 

## Resources

[Bmad website](https://www.classe.cornell.edu/bmad)





## License

[GNU General Public License](LICENSE)
