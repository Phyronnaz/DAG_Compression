# DAG-example

Fork of https://github.com/gegoggigog/DAG-example/tree/compression to work with https://github.com/Phyronnaz/HashDAG

WIP!

Currently it only supports building DAGs up to a resolution of 1024.

No color compression included yet either. Only raw colors.

TODO:
* Add support for larger DAGs by merging DAGs.
* Add color compression.
* Make code make more sense.

Dependencies:
* CUDA
* OpenGL
* GLEW
* glm (https://github.com/g-truc/glm)
* Cereal (https://github.com/USCiLab/cereal) (Optional. For DAGLoader. Storing data to disk.)
